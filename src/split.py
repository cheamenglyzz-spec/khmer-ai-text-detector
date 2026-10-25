"""Step 4: split the cleaned dataset into train / val / test plus an out-of-distribution test set.

Rows are grouped by `human_id`: a human passage and the AI passage(s) written for it always land
in the same split, so the classifier is never tested on a topic it saw in training.

1. test_ood: the AI rows of the held-out generator (a model family the training set never sees)
   plus human rows, built per 200-char length bin so that every bin has as many humans as AI rows.
   Humans linked to a held-out AI row come first, then humans with no AI partner fill the rest of
   the bin. A linked human that finds no free slot in its bin, and an AI row that finds no human
   for its bin, is `quarantined`: it is in no split, so each human_id still sits in one place only.
2. The rest is split 80/10/10 into train/val/test, stratified by group kind (pair / human only /
   AI only) x generator x source as far as the sizes allow, with seed 42.

Input:  data/processed/clean.jsonl   (step 3)
Output: data/processed/splits/{train,val,test,test_ood}.jsonl   same record schema
        reports/step4_split_stats.json

Usage:
    python src/split.py
    python src/split.py --holdout openai/gpt-oss-120b
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `python src/split.py` can import its sibling module as src.clean
from src.clean import (  # noqa: E402
    HUMAN_GENERATOR, N_BINS, bin_label, count_pairs, group_by, ks_by_label, label_name, length_bin,
    length_stats, read_jsonl, sample_by_tier, write_jsonl,
)

IN_PATH = ROOT / "data" / "processed" / "clean.jsonl"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
STATS_PATH = ROOT / "reports" / "step4_split_stats.json"

SEED = 42
DEFAULT_HOLDOUT = "qwen/qwen3.8-27b"
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_NAMES = ("train", "val", "test", "test_ood")
MIN_ROWS_PER_SPLIT = 10  # a generator with fewer rows than this in val or test gets a warning
SAMPLES_PER_SPLIT = 2
SAMPLE_CHARS = 150


# ---------------------------------------------------------------------------
# Grouping and the out-of-distribution set
# ---------------------------------------------------------------------------
def group_rows(records: list[dict]) -> dict[str, list[dict]]:
    """human_id -> its rows. A human row's human_id is its own id, so a group is one human passage
    plus the AI passage(s) written for it (either side may be missing from the cleaned data)."""
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(rec["human_id"], []).append(rec)
    return groups


def has_ai(rows: list[dict]) -> bool:
    return any(r["label"] == 1 for r in rows)


def pick_ood(groups: dict[str, list[dict]], holdout: str, seed: int) -> tuple[list[dict], set[str], list[dict], dict]:
    """Build test_ood bin by bin. Returns (rows, group ids used up, quarantined rows, info for the report).

    In each 200-char length bin, every held-out AI row needs a human row of the same bin:
      1. humans linked to a held-out AI row (same human_id) go first; one that finds no free slot in its
         bin is quarantined, so its human_id is still in one place only (the AI row stays in test_ood);
      2. then humans with no AI partner at all fill the free slots (seeded random pick);
      3. then, if still short, humans whose partner is another generator's AI row (that partner row is
         quarantined);
      4. if the bin has fewer humans than AI rows, the surplus AI rows are quarantined, those whose human
         is not in test_ood first.
    Quarantined rows are in no split. Pools are sorted by id before sampling, so the result depends only on
    the data and the seed.
    """
    rng = random.Random(seed)
    held = sorted(
        (r for rows in groups.values() for r in rows if r["label"] == 1 and r["generator"] == holdout),
        key=lambda r: r["id"],
    )
    if not held:
        have = sorted({r["generator"] for rows in groups.values() for r in rows if r["label"] == 1})
        raise ValueError(f"no AI rows from the held-out generator {holdout!r}; generators present: {have}")
    used = {r["human_id"] for r in held}  # groups that can no longer go to train/val/test
    out = {"other_generator_ai": [], "linked_humans": [], "partner_ai": [], "surplus_ai": []}  # quarantined, by reason
    for hid in sorted(used):  # a second AI row for the same passage would put the human_id in two splits
        out["other_generator_ai"] += [r for r in groups[hid] if r["label"] == 1 and r["generator"] != holdout]

    ai_by_bin: dict[int, list[dict]] = {b: [] for b in range(N_BINS)}
    for r in held:
        ai_by_bin[length_bin(r["length"])].append(r)
    pools = {name: {b: [] for b in range(N_BINS)} for name in ("linked", "unpaired", "paired")}
    for hid in sorted(groups):
        for h in groups[hid]:
            if h["label"] != 0:
                continue
            name = "linked" if hid in used else ("paired" if has_ai(groups[hid]) else "unpaired")
            pools[name][length_bin(h["length"])].append(h)

    humans_in: list[dict] = []
    bins = []
    for b in range(N_BINS):
        slots = len(ai_by_bin[b])
        row = {"bin": bin_label(b), "ai": slots, "linked_available": len(pools["linked"][b])}
        chosen = sample_by_tier(pools["linked"][b], slots, lambda r: 0, rng)
        row["linked_kept"] = len(chosen)
        chosen_ids = {h["id"] for h in chosen}
        out["linked_humans"] += [h for h in pools["linked"][b] if h["id"] not in chosen_ids]
        row["linked_quarantined"] = len(pools["linked"][b]) - len(chosen)
        for name in ("unpaired", "paired"):
            extra = sample_by_tier(pools[name][b], slots - len(chosen), lambda r: 0, rng) if slots > len(chosen) else []
            row[f"from_{name}"] = len(extra)
            chosen += extra
            used.update(h["human_id"] for h in extra)
            if name == "paired":
                for h in extra:
                    out["partner_ai"] += [r for r in groups[h["human_id"]] if r["label"] == 1]
        row["humans"] = len(chosen)
        humans_in += chosen
        bins.append(row)

    human_ids_in = {h["human_id"] for h in humans_in}
    ai_in: list[dict] = []
    for b in range(N_BINS):
        keep = sample_by_tier(ai_by_bin[b], bins[b]["humans"], lambda r: 0 if r["human_id"] in human_ids_in else 1, rng)
        kept_ids = {r["id"] for r in keep}
        out["surplus_ai"] += [r for r in ai_by_bin[b] if r["id"] not in kept_ids]
        bins[b]["ai_kept"] = len(keep)
        bins[b]["ai_quarantined"] = len(ai_by_bin[b]) - len(keep)
        ai_in += keep

    rows = humans_in + ai_in
    quarantined = [r for reason_rows in out.values() for r in reason_rows]
    info = {
        "holdout": holdout, "ai_rows": len(held), "ai_rows_kept": len(ai_in), "human_rows": len(humans_in),
        "pairs": count_pairs(rows), "bins": bins,
        "humans_from": {name: sum(b[key] for b in bins) for name, key in
                        (("linked", "linked_kept"), ("unpaired", "from_unpaired"), ("paired", "from_paired"))},
        "quarantined": {reason: len(reason_rows) for reason, reason_rows in out.items()},
    }
    return rows, used, quarantined, info


# ---------------------------------------------------------------------------
# In-distribution split
# ---------------------------------------------------------------------------
def stratum(rows: list[dict]) -> tuple[str, str, str]:
    """(kind, AI generator(s), source) of one group."""
    labels = {r["label"] for r in rows}
    kind = "pair" if labels == {0, 1} else ("ai_only" if labels == {1} else "human_only")
    generators = sorted({r["generator"] for r in rows if r["label"] == 1})
    return kind, "+".join(generators) or "-", rows[0]["source"]


def assign_groups(strata: dict[tuple, list[str]], fractions: dict[str, float], seed: int) -> dict[str, str]:
    """group id -> 'train' | 'val' | 'test', per stratum.

    Each stratum is shuffled and cut by the fractions. The rounding error is carried into the next
    stratum, so small strata (a few rows of one generator) do not all round down to nothing and
    the overall shares still come out at the requested fractions.
    """
    rng = random.Random(seed)
    carry = {"val": 0.0, "test": 0.0}
    assignment: dict[str, str] = {}
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        n = len(ids)
        counts = {"val": 0, "test": 0}
        for name in ("val", "test"):
            want = fractions[name] * n + carry[name]
            counts[name] = min(max(math.floor(want + 0.5), 0), n - sum(counts.values()))
            carry[name] = want - counts[name]
        cuts = {"val": ids[: counts["val"]], "test": ids[counts["val"]: counts["val"] + counts["test"]]}
        for name, chosen in cuts.items():
            for gid in chosen:
                assignment[gid] = name
        for gid in ids[counts["val"] + counts["test"]:]:
            assignment[gid] = "train"
    return assignment


def make_splits(records: list[dict], holdout: str = DEFAULT_HOLDOUT, seed: int = SEED) -> tuple[dict[str, list[dict]], dict]:
    """Return ({split name: rows}, info). Every row of `records` is in exactly one split or quarantined."""
    groups = group_rows(records)
    ood_rows, used, quarantined, ood_info = pick_ood(groups, holdout, seed)
    quarantined_ids = {r["id"] for r in quarantined}
    rest = {hid: [r for r in rows if r["id"] not in quarantined_ids] for hid, rows in groups.items() if hid not in used}
    strata: dict[tuple, list[str]] = {}
    for hid, rows in rest.items():
        strata.setdefault(stratum(rows), []).append(hid)
    assignment = assign_groups(strata, SPLIT_FRACTIONS, seed)

    splits: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for hid, rows in rest.items():
        splits[assignment[hid]] += rows
    splits["test_ood"] = ood_rows
    rng = random.Random(seed)
    for name in SPLIT_NAMES:
        splits[name].sort(key=lambda r: r["id"])
        rng.shuffle(splits[name])
    check_splits(splits, quarantined, records)
    return splits, {"ood": ood_info, "quarantined": quarantined}


def check_splits(splits: dict[str, list[dict]], quarantined: list[dict], records: list[dict]) -> None:
    """Raise AssertionError if the splits leak or lose rows."""
    owner: dict[str, str] = {}
    for name, rows in splits.items():
        for rec in rows:
            first = owner.setdefault(rec["human_id"], name)
            assert first == name, f"human_id {rec['human_id']} is in both {first} and {name}"
    every = [r for rows in splits.values() for r in rows]
    assert len({r["id"] for r in every}) == len(every), "a row is in more than one split"
    assert len({r["text"] for r in every}) == len(every), "the same text appears twice"
    assert len(every) + len(quarantined) == len(records), "rows were lost or invented"


# ---------------------------------------------------------------------------
# Statistics and reporting
# ---------------------------------------------------------------------------
def split_summary(rows: list[dict]) -> dict:
    by_label = group_by(rows, label_name)
    return {
        "n": len(rows),
        "by_label": {k: len(v) for k, v in by_label.items()},
        "by_generator": {k: len(v) for k, v in group_by(rows, lambda r: r["generator"]).items()},
        "by_source": {k: len(v) for k, v in group_by(rows, lambda r: r["source"]).items()},
        "length_by_label": {k: length_stats([r["length"] for r in v]) for k, v in by_label.items()},
        "ks_length_human_vs_ai": ks_by_label(rows),
    }


def small_generator_warnings(summaries: dict[str, dict], holdout: str) -> list[str]:
    """Generators with too few rows in val or test (or in test_ood, for the held-out one) to say anything."""
    notes = []
    generators = sorted({g for s in summaries.values() for g in s["by_generator"] if g != HUMAN_GENERATOR})
    for gen in generators:
        if gen == holdout:
            n = summaries["test_ood"]["by_generator"].get(gen, 0)
            if n < MIN_ROWS_PER_SPLIT:
                notes.append(f"{gen}: only {n} rows in test_ood (held out); fewer than {MIN_ROWS_PER_SPLIT}")
            continue
        counts = {name: summaries[name]["by_generator"].get(gen, 0) for name in ("train", "val", "test")}
        if min(counts["val"], counts["test"]) < MIN_ROWS_PER_SPLIT:
            notes.append(
                f"{gen}: train {counts['train']}, val {counts['val']}, test {counts['test']}; "
                f"fewer than {MIN_ROWS_PER_SPLIT} rows in val or test, so its per-generator numbers mean little"
            )
    return notes


def print_split_tables(summaries: dict[str, dict]) -> None:
    names = list(summaries)
    print("\n=== Rows per split ===")
    print(f"{'':<34}" + "".join(f"{n:>10}" for n in names))
    print(f"{'total':<34}" + "".join(f"{summaries[n]['n']:>10}" for n in names))
    for title, key in (("label", "by_label"), ("generator", "by_generator"), ("source", "by_source")):
        keys = sorted({k for s in summaries.values() for k in s[key]})
        for k in keys:
            print(f"{title + ' ' + k:<34}" + "".join(f"{summaries[n][key].get(k, 0):>10}" for n in names))
    print("\n--- Length (chars): n / min / median / mean / max ---")
    for name in names:
        for label, st in summaries[name]["length_by_label"].items():
            if st["n"]:
                print(f"{name + ' ' + label:<18}{st['n']:>6}{st['min']:>7}{st['median']:>8.0f}{st['mean']:>8.0f}{st['max']:>7}")
        ks = summaries[name]["ks_length_human_vs_ai"]
        print(f"{name + ' KS human-vs-ai':<18}{'n/a' if ks is None else format(ks, '.3f'):>6}")


def print_ood(info: dict) -> None:
    o = info["ood"]
    q = o["quarantined"]
    h = o["humans_from"]
    print(f"\n=== test_ood (held out: {o['holdout']}) ===")
    print(f"AI rows {o['ai_rows']} -> kept {o['ai_rows_kept']}; human rows {o['human_rows']} "
          f"(linked to a held-out AI row {h['linked']}, no AI partner {h['unpaired']}, partner is another generator {h['paired']}); "
          f"pairs kept {o['pairs']}")
    print(f"quarantined (in no split): {sum(q.values())} = linked humans without a free slot {q['linked_humans']}, "
          f"surplus AI rows {q['surplus_ai']}, partner AI rows of used humans {q['partner_ai']}, "
          f"other-generator AI rows sharing a passage {q['other_generator_ai']}")
    print(f"{'bin':>12}{'AI':>5}{'AI kept':>9}{'linked':>8}{'kept':>6}{'+free':>7}{'+paired':>9}{'humans':>8}")
    for r in o["bins"]:
        if r["ai"] or r["linked_available"]:
            print(f"{r['bin']:>12}{r['ai']:>5}{r['ai_kept']:>9}{r['linked_available']:>8}{r['linked_kept']:>6}"
                  f"{r['from_unpaired']:>7}{r['from_paired']:>9}{r['humans']:>8}")


def print_samples(splits: dict[str, list[dict]], seed: int = SEED) -> None:
    print(f"\n=== {SAMPLES_PER_SPLIT} random samples per split (seed {seed}), first {SAMPLE_CHARS} chars ===")
    for name, rows in splits.items():
        for rec in random.Random(seed).sample(rows, min(SAMPLES_PER_SPLIT, len(rows))):
            print(f"[{name} | {label_name(rec)} | {rec['generator']} | {rec['source']} | {rec['length']} chars]")
            print("  " + rec["text"][:SAMPLE_CHARS].replace("\n", " | "))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split the cleaned dataset into train/val/test/test_ood.")
    p.add_argument("--input", type=Path, default=IN_PATH, help="cleaned JSONL (default: %(default)s)")
    p.add_argument("--out-dir", type=Path, default=SPLITS_DIR, help="folder for the split files (default: %(default)s)")
    p.add_argument("--stats", type=Path, default=STATS_PATH, help="stats JSON (default: %(default)s)")
    p.add_argument("--holdout", default=DEFAULT_HOLDOUT, help="generator held out for test_ood (default: %(default)s)")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Khmer must not crash a cp1252 console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)

    records = read_jsonl(args.input)
    if not records:
        print(f"error: no records in {args.input}", file=sys.stderr)
        return 2
    print(f"Input: {len(records)} rows from {args.input.name}")
    try:
        splits, info = make_splits(records, args.holdout, args.seed)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summaries = {name: split_summary(rows) for name, rows in splits.items()}
    print_split_tables(summaries)
    print_ood(info)
    notes = small_generator_warnings(summaries, args.holdout)
    for note in notes:
        print(f"NOTE: {note}")
    print_samples(splits, args.seed)

    for name, rows in splits.items():
        write_jsonl(args.out_dir / f"{name}.jsonl", rows)
    quarantined = info["quarantined"]
    stats = {
        "config": {"seed": args.seed, "holdout": args.holdout, "fractions": SPLIT_FRACTIONS, "input": str(args.input)},
        "input_rows": len(records),
        "splits": summaries,
        "test_ood": info["ood"],
        "quarantined": {
            "rows": len(quarantined),
            "by_reason": info["ood"]["quarantined"],
            "by_label": dict(sorted(Counter(label_name(r) for r in quarantined).items())),
            "by_generator": dict(sorted(Counter(r["generator"] for r in quarantined).items())),
        },
        "notes": notes,
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {', '.join(f'{n} {len(r)}' for n, r in splits.items())} to {args.out_dir}\nWrote stats to {args.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
