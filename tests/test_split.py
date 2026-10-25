"""Tests for src/split.py on a small synthetic dataset (run: python -m pytest tests)."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import clean, split  # noqa: E402

HOLDOUT = "g/holdout"
GENERATORS = ["g/a", "g/b", "g/c", HOLDOUT]
SOURCES = ["wikipedia", "mc4_web"]


def row(id_, label, length, human_id, generator, source, text=None):
    return {
        "id": id_, "text": text or f"ខ្ញុំស្រលាញ់ {id_}។", "label": label, "source": source, "generator": generator,
        "url": "", "date": "", "length": length, "human_id": human_id, "title": "", "prompt_type": "",
    }


def make_records(n=400, p_human=0.85, p_ai=0.7, seed=0):
    """n topics; each has a human row and an AI row (generators in rotation), either may be missing."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        hid, source = f"h{i:04d}", SOURCES[i % 2]
        if rng.random() < p_human:
            records.append(row(hid, 0, rng.randint(300, 2500), hid, "human", source))
        if rng.random() < p_ai:
            records.append(row(f"ai_{hid}", 1, rng.randint(300, 2500), hid, GENERATORS[i % 4], source))
    return records


def human_ids(rows):
    return {r["human_id"] for r in rows}


def test_splits_do_not_leak_and_keep_groups_together():
    records = make_records()
    splits, info = split.make_splits(records, HOLDOUT, seed=42)
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not human_ids(splits[a]) & human_ids(splits[b]), (a, b)
    in_splits = {r["id"] for rows in splits.values() for r in rows}
    assert in_splits | {r["id"] for r in info["quarantined"]} == {r["id"] for r in records}
    assert len(in_splits) == sum(len(rows) for rows in splits.values())
    owner = {r["id"]: name for name, rows in splits.items() for r in rows}
    for rec in records:  # an AI row and its human row are always in the same split
        if rec["label"] == 1 and rec["human_id"] in owner and rec["id"] in owner:
            assert owner[rec["id"]] == owner[rec["human_id"]]


def test_check_splits_catches_leaks():
    a, b = row("h1", 0, 400, "h1", "human", "wikipedia"), row("ai_h1", 1, 400, "h1", "g/a", "wikipedia")
    with pytest.raises(AssertionError, match="both"):
        split.check_splits({"train": [a], "val": [b]}, [], [a, b])
    c = row("h2", 0, 400, "h2", "human", "wikipedia", text=a["text"])
    with pytest.raises(AssertionError, match="twice"):
        split.check_splits({"train": [a], "val": [c]}, [], [a, c])
    with pytest.raises(AssertionError, match="lost"):
        split.check_splits({"train": [a]}, [], [a, c])
    split.check_splits({"train": [a, b]}, [], [a, b])  # a clean split passes


def split_of(splits, quarantined=()):
    owner = {r["id"]: name for name, rows in splits.items() for r in rows}
    owner.update({r["id"]: "quarantined" for r in quarantined})
    return owner


def test_holdout_generator_is_only_in_test_ood_or_quarantined():
    records = make_records()
    splits, info = split.make_splits(records, HOLDOUT, seed=42)
    holdout_ids = {r["id"] for r in records if r["generator"] == HOLDOUT}
    ood_ai = {r["id"] for r in splits["test_ood"] if r["label"] == 1}
    quarantined_ai = {r["id"] for r in info["quarantined"] if r["generator"] == HOLDOUT}
    assert ood_ai | quarantined_ai == holdout_ids and not ood_ai & quarantined_ai  # every held-out row is one or the other
    assert {r["generator"] for r in splits["test_ood"] if r["label"] == 1} == {HOLDOUT}
    for name in ("train", "val", "test"):
        assert HOLDOUT not in {r["generator"] for r in splits[name]}
    linked = {r["human_id"] for r in records if r["generator"] == HOLDOUT}
    owner = split_of(splits, info["quarantined"])
    present = {r["id"] for r in records if r["label"] == 0 and r["id"] in linked}
    assert {owner[i] for i in present} <= {"test_ood", "quarantined"}  # a linked human never lands in train/val/test


def test_test_ood_has_equal_counts_in_every_length_bin():
    records = make_records()
    splits, info = split.make_splits(records, HOLDOUT, seed=42)
    ood = splits["test_ood"]
    labels = Counter(r["label"] for r in ood)
    assert labels[0] == labels[1] > 0
    assert all(n_human == n_ai for n_human, n_ai in counts_per_bin(ood).values())
    assert info["ood"]["ai_rows_kept"] == labels[1] and info["ood"]["human_rows"] == labels[0]
    assert info["ood"]["pairs"] == clean.count_pairs(ood) > 0
    assert clean.ks_by_label(ood) < 0.3


def counts_per_bin(rows):
    counts = {}
    for r in rows:
        counts.setdefault(clean.length_bin(r["length"]), [0, 0])[r["label"]] += 1
    return counts


def test_linked_humans_come_first_then_humans_without_a_partner():
    # bin 0 (400 chars) has 4 held-out AI rows; two of their humans are in bin 0, the other two are in bin 4
    records = [row(f"ai_l{i}", 1, 400, f"l{i}", HOLDOUT, "wikipedia") for i in range(4)]
    records += [row(f"l{i}", 0, 410, f"l{i}", "human", "wikipedia") for i in range(2)]  # linked, same bin
    records += [row(f"l{i}", 0, 1200, f"l{i}", "human", "wikipedia") for i in range(2, 4)]  # linked, bin 4: no free slot
    records += [row(f"u{i}", 0, 420 + i, f"u{i}", "human", "wikipedia") for i in range(30)]  # no AI partner
    splits, info = split.make_splits(records, HOLDOUT, seed=42)
    ood_ids = {r["id"] for r in splits["test_ood"]}
    assert {"l0", "l1"} <= ood_ids  # the linked humans with a slot are used first
    assert len([i for i in ood_ids if i.startswith("u")]) == 2  # the other two slots go to unpartnered humans
    quarantined = {r["id"] for r in info["quarantined"]}
    assert quarantined == {"l2", "l3"}  # linked, but their bin has no slot: in no split
    assert not quarantined & set(split_of(splits)) and info["ood"]["pairs"] == 2
    assert info["ood"]["quarantined"]["linked_humans"] == 2


def test_surplus_ai_rows_are_quarantined_when_a_bin_has_too_few_humans():
    # bin 3 (1000 chars): 5 held-out AI rows and only 2 humans in the whole data for that bin
    records = [row(f"ai_x{i}", 1, 1000, f"x{i}", HOLDOUT, "wikipedia") for i in range(5)]
    records += [row(f"hh{i}", 0, 1010, f"hh{i}", "human", "wikipedia") for i in range(2)]
    records += [row(f"o{i}", 0, 400 + 10 * i, f"o{i}", "human", "wikipedia") for i in range(40)]  # other bins only
    splits, info = split.make_splits(records, HOLDOUT, seed=42)
    ood = splits["test_ood"]
    assert Counter(r["label"] for r in ood) == {0: 2, 1: 2}
    assert info["ood"]["quarantined"]["surplus_ai"] == 3
    assert len([r for r in info["quarantined"] if r["label"] == 1]) == 3
    assert all(n_human == n_ai for n_human, n_ai in counts_per_bin(ood).values())


def test_surplus_ai_rows_whose_human_is_in_test_ood_are_kept():
    # bin 3: 4 held-out AI rows, 2 humans available: the two linked ones. The AI rows with a human survive.
    records = [row(f"ai_p{i}", 1, 1000, f"p{i}", HOLDOUT, "wikipedia") for i in range(2)]
    records += [row(f"ai_q{i}", 1, 1000, f"q{i}", HOLDOUT, "wikipedia") for i in range(2)]  # no human in the data
    records += [row(f"p{i}", 0, 1010, f"p{i}", "human", "wikipedia") for i in range(2)]
    splits, info = split.make_splits(records, HOLDOUT, seed=42)
    kept_ai = {r["id"] for r in splits["test_ood"] if r["id"].startswith("ai_") and r["id"][3] in "pq"}
    assert kept_ai == {"ai_p0", "ai_p1"}  # the two with a human in test_ood
    assert {"ai_q0", "ai_q1"} <= {r["id"] for r in info["quarantined"]}


def test_quarantine_when_only_paired_humans_are_left():
    records = []
    for i in range(30):  # holdout AI rows whose human passages are not in the data
        records.append(row(f"ai_x{i}", 1, 400 + 60 * i, f"x{i}", HOLDOUT, "wikipedia"))
    for i in range(120):  # every other human has an AI partner, so top-up must take paired humans
        hid = f"h{i:03d}"
        records.append(row(hid, 0, 300 + 17 * i, hid, "human", "wikipedia"))
        records.append(row(f"ai_{hid}", 1, 300 + 17 * i, hid, "g/a", "wikipedia"))
    splits, info = split.make_splits(records, HOLDOUT, seed=42)
    ood_labels = Counter(r["label"] for r in splits["test_ood"])
    assert ood_labels[0] == ood_labels[1] == 30
    assert info["ood"]["humans_from"] == {"linked": 0, "unpaired": 0, "paired": 30}
    quarantined = {r["id"] for r in info["quarantined"]}
    assert len(quarantined) == 30
    assert not quarantined & {r["id"] for rows in splits.values() for r in rows}  # in no split
    assert all(r["label"] == 1 and r["generator"] == "g/a" for r in info["quarantined"])


def test_partner_rows_of_another_generator_in_a_holdout_group_are_set_aside():
    records = [
        row("h1", 0, 500, "h1", "human", "wikipedia"),
        row("ai_h1", 1, 500, "h1", HOLDOUT, "wikipedia"),
        row("ai2_h1", 1, 520, "h1", "g/a", "wikipedia"),  # second AI row linked to the same human
    ] + make_records(60, seed=3)
    splits, info = split.make_splits(records, HOLDOUT, seed=42)
    assert "ai2_h1" in {r["id"] for r in info["quarantined"]}
    assert "ai2_h1" not in {r["id"] for rows in splits.values() for r in rows}
    assert info["ood"]["quarantined"]["other_generator_ai"] >= 1


def test_in_distribution_split_sizes_and_stratification():
    records = make_records(n=1200)
    splits, _ = split.make_splits(records, HOLDOUT, seed=42)
    sizes = {name: len(splits[name]) for name in ("train", "val", "test")}
    total = sum(sizes.values())
    assert abs(sizes["val"] / total - 0.1) < 0.02 and abs(sizes["test"] / total - 0.1) < 0.02
    for name in ("train", "val", "test"):
        gens = Counter(r["generator"] for r in splits[name] if r["label"] == 1)
        assert set(gens) == {"g/a", "g/b", "g/c"}
    def ai_share(name):
        return sum(r["label"] == 1 for r in splits[name]) / len(splits[name])
    for name in ("val", "test"):  # the synthetic data has fewer AI than human rows; every split keeps that ratio
        assert abs(ai_share(name) - ai_share("train")) < 0.06
    # each generator's share of AI rows is about the same in every split
    def shares(name):
        gens = Counter(r["generator"] for r in splits[name] if r["label"] == 1)
        return {g: n / sum(gens.values()) for g, n in gens.items()}
    for g in ("g/a", "g/b", "g/c"):
        assert abs(shares("train")[g] - shares("val")[g]) < 0.1
        assert abs(shares("train")[g] - shares("test")[g]) < 0.1


def test_splits_are_deterministic_for_a_seed():
    records = make_records()
    ids = lambda s: {name: [r["id"] for r in rows] for name, rows in s.items()}  # noqa: E731
    first = ids(split.make_splits(records, HOLDOUT, seed=42)[0])
    assert first == ids(split.make_splits(list(reversed(records)), HOLDOUT, seed=42)[0])  # input order is irrelevant
    assert first == ids(split.make_splits(records, HOLDOUT, seed=42)[0])
    assert first != ids(split.make_splits(records, HOLDOUT, seed=7)[0])


def test_missing_holdout_generator_raises():
    with pytest.raises(ValueError, match="held-out generator"):
        split.make_splits(make_records(), "g/nope", seed=42)


def test_assign_groups_carries_rounding_across_small_strata():
    strata = {("pair", f"g{i}", "wikipedia"): [f"s{i}_{j}" for j in range(3)] for i in range(40)}  # 120 groups
    assignment = split.assign_groups(strata, split.SPLIT_FRACTIONS, seed=42)
    counts = Counter(assignment.values())
    assert counts["val"] == 12 and counts["test"] == 12 and counts["train"] == 96
    assert split.assign_groups(strata, split.SPLIT_FRACTIONS, seed=42) == assignment


def test_main_writes_splits_and_stats(tmp_path, capsys):
    records = make_records(200)
    src = tmp_path / "clean.jsonl"
    clean.write_jsonl(src, records)
    out_dir, stats = tmp_path / "splits", tmp_path / "stats.json"
    code = split.main(["--input", str(src), "--out-dir", str(out_dir), "--stats", str(stats), "--holdout", HOLDOUT])
    assert code == 0
    written = {name: clean.read_jsonl(out_dir / f"{name}.jsonl") for name in split.SPLIT_NAMES}
    assert all(written[name] for name in written)
    assert all(set(r) == set(clean.OUTPUT_FIELDS) for rows in written.values() for r in rows)
    data = json.loads(stats.read_text(encoding="utf-8"))
    assert data["splits"]["test_ood"]["n"] == len(written["test_ood"])
    assert data["config"]["holdout"] == HOLDOUT
    assert "samples per split" in capsys.readouterr().out
    assert split.main(["--input", str(src), "--out-dir", str(out_dir), "--stats", str(stats), "--holdout", "g/nope"]) == 2
