"""Step 3: clean, deduplicate and length-match human vs AI Khmer text into one labelled dataset.

Both classes go through exactly the same normalization, so the detector cannot learn
formatting artifacts (e.g. gpt-oss puts U+200B between Khmer words, human text does not).

Pipeline: normalize -> truncation / Khmer-ratio / length filters -> copied-hint check
          -> deduplication -> length matching -> shuffle -> write.

Inputs:
    data/human.csv                   human passages (step 1)
    data/generated/ai_raw.jsonl      AI passages (step 2); --ai takes one or more files
Outputs:
    data/processed/clean.jsonl       one record per line, shuffled with seed 42
    reports/step3_clean_stats.json   counts, drops by reason, length stats, KS, Latin-word rates

Usage:
    python src/clean.py
    python src/clean.py --ai data/generated/test100.jsonl      # dev run on a small sample
    python src/clean.py --no-length-match                      # keep every cleaned row
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HUMAN_CSV = ROOT / "data" / "human.csv"
AI_PATH = ROOT / "data" / "generated" / "ai_raw.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "clean.jsonl"
STATS_PATH = ROOT / "reports" / "step3_clean_stats.json"

SEED = 42
LABEL_NAMES = {0: "human", 1: "ai"}
HUMAN_GENERATOR = "human"
PROMPT_WEB = "web_hint"  # AI rows whose prompt showed the start of the linked human passage

# ---------------------------------------------------------------------------
# Thresholds (edit here)
# ---------------------------------------------------------------------------
MIN_CHARS = 300
MAX_CHARS = 2500
MIN_KHMER_RATIO = 0.7
BIN_WIDTH = 200  # length-matching bins: 300-499, 500-699, ..., 2300-2500 (last bin includes 2500)
N_BINS = (MAX_CHARS - MIN_CHARS) // BIN_WIDTH
COPY_CHECK_CHARS = 300  # copy check looks at the first N chars of a web_hint AI text...
COPY_MIN_SPAN = 60  # ...and drops it if a span this long also appears in the linked human text
NEAR_DUP_CHARS = 150  # texts with identical first N chars count as near-duplicates
TRUNCATED_FINISH_REASONS = {"length"}  # the API cut these off at the token cap, mid-word
SENTENCE_ENDS = "។៕?!"  # Khmer khan and bariyoosan, plus ? and !
CLOSERS = "\"'”’»)]"  # closing quotes/brackets that may follow a sentence end

# Invisible characters, removed from BOTH classes. First five: zero-width space/non-joiner/
# joiner, word joiner, BOM. The rest only showed up in the human text (soft hyphen, LRM/RLM,
# bidi embeddings, invisible operators); gone from both so they cannot become a feature.
INVISIBLE_CHARS = (
    "​‌‍⁠﻿"
    "­‎‏‪‫‬‭‮⁡⁢⁣⁤"
)

DROP_REASONS = (
    "truncated_no_sentence_end",
    "low_khmer_ratio",
    "too_short",
    "too_long",
    "copied_hint",
    "duplicate_id",
    "duplicate",
    "near_duplicate",
)
OUTPUT_FIELDS = (
    "id", "text", "label", "source", "generator", "url", "date",
    "length", "human_id", "title", "prompt_type",
)

# ---------------------------------------------------------------------------
# Text cleaning (pure functions)
# ---------------------------------------------------------------------------
_REMOVE_INVISIBLE = str.maketrans("", "", INVISIBLE_CHARS)
_LINE_BREAKS = re.compile(r"\r\n|[\r\x0b\x0c\x85  ]")
_CONTROLS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]")
_INLINE_SPACE = re.compile(r"[^\S\n]+")  # any Unicode whitespace except the newline
# "()", "( )" and Wikipedia template leftovers like "(, ;)": nothing but spaces and separators.
_EMPTY_PARENS = re.compile(r"\([\s,;:]*\)")
_SENTENCE_END = re.compile(f"[{SENTENCE_ENDS}]+[{re.escape(CLOSERS)}]*")
_KHMER_CHAR = re.compile("[ក-៿᧠-᧿]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def normalize_text(text: str) -> str:
    """Apply the same formatting rules to every text, human or AI."""
    text = unicodedata.normalize("NFC", text.translate(_REMOVE_INVISIBLE))
    text = _LINE_BREAKS.sub("\n", text)
    text = _CONTROLS.sub(" ", text)
    text = _INLINE_SPACE.sub(" ", text)  # NBSP and friends become a normal space, runs collapse
    while True:  # loop so that "(())" is gone too
        text, n = _EMPTY_PARENS.subn("", text)
        if not n:
            break
    text = _INLINE_SPACE.sub(" ", text)  # removing "()" can leave a double space
    lines = (line.strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line).strip()  # one "\n" between paragraphs


def trim_to_sentence_end(text: str, limit: int | None = None) -> str:
    """Cut `text` after its last sentence end (within the first `limit` chars). '' if there is none."""
    window = text if limit is None else text[:limit]
    last = None
    for last in _SENTENCE_END.finditer(window):
        pass
    return window[: last.end()].rstrip() if last else ""


def ends_with_sentence_end(text: str) -> bool:
    return bool(text) and trim_to_sentence_end(text) == text


def khmer_ratio(text: str) -> float:
    """Share of non-space characters that are Khmer script (U+1780-17FF, U+19E0-19FF)."""
    chars = "".join(text.split())
    return len(_KHMER_CHAR.findall(chars)) / len(chars) if chars else 0.0


def latin_word_count(text: str) -> int:
    return len(_LATIN_WORD.findall(text))


def flatten(text: str) -> str:
    """Collapse all whitespace (incl. newlines) to single spaces; the web_hint prompt did the same."""
    return " ".join(text.split())


def copies_hint(ai_text: str, human_text: str) -> bool:
    """True if the AI text's opening shares a verbatim span of COPY_MIN_SPAN chars with the human text.

    Any common substring of that length is also a COPY_MIN_SPAN-long window of the opening,
    so checking each window with `in` is exact (and much cheaper than a full alignment).
    """
    head = flatten(ai_text)[:COPY_CHECK_CHARS]
    human = flatten(human_text)
    return any(head[i : i + COPY_MIN_SPAN] in human for i in range(len(head) - COPY_MIN_SPAN + 1))


@dataclass
class Filtered:
    text: str
    reason: str | None = None  # a DROP_REASONS entry, or None if the text stays
    trimmed_truncated: bool = False  # AI text cut at the token cap, trimmed to its last sentence end
    trimmed_long: bool = False  # AI text over MAX_CHARS, trimmed to a sentence end


def filter_text(text: str, is_ai: bool, truncated: bool = False) -> Filtered:
    """Truncation, Khmer-ratio and length rules for one already-normalized text."""
    out = Filtered(text)
    if truncated:
        trimmed = trim_to_sentence_end(text)
        if not trimmed:
            out.reason = "truncated_no_sentence_end"
            return out
        out.trimmed_truncated = trimmed != text
        out.text = text = trimmed
    if khmer_ratio(text) < MIN_KHMER_RATIO:
        out.reason = "low_khmer_ratio"
        return out
    if is_ai and len(text) > MAX_CHARS:
        trimmed = trim_to_sentence_end(text, MAX_CHARS)
        if trimmed:  # no sentence end inside the limit: leave it, it is dropped as too_long below
            out.trimmed_long = True
            out.text = text = trimmed
    if len(text) < MIN_CHARS:
        out.reason = "too_short"
    elif len(text) > MAX_CHARS:
        out.reason = "too_long"
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_human(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = {"id", "text", "title", "source", "url", "date"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return [
        {
            "id": r["id"], "text": r["text"], "label": 0, "source": r["source"],
            "generator": HUMAN_GENERATOR, "url": r["url"], "date": r["date"], "length": 0,
            "human_id": r["id"], "title": r["title"], "prompt_type": "", "finish_reason": "",
        }
        for r in df.to_dict("records")
    ]


def read_jsonl(path: Path) -> list[dict]:
    """Read records, skipping blank or unparseable lines (a live writer may leave a half line)."""
    records: list[dict] = []
    bad = 0
    with open(path, "rb") as f:  # decode per line: one bad line must not hide the rest
        for raw in f:
            if not raw.strip():
                continue
            try:
                records.append(json.loads(raw.decode("utf-8")))
            except ValueError:  # includes JSONDecodeError and UnicodeDecodeError
                bad += 1
    if bad:
        print(f"WARNING: skipped {bad} unparseable line(s) in {path.name}", file=sys.stderr)
    return records


def load_ai(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        for r in read_jsonl(path):
            human_id = str(r.get("human_id") or "")
            records.append(
                {
                    "id": str(r.get("id") or f"ai_{human_id}"), "text": str(r.get("text") or ""), "label": 1,
                    "source": str(r.get("source") or ""), "generator": str(r.get("generator") or ""),
                    "url": str(r.get("url") or ""), "date": str(r.get("date") or ""), "length": 0,
                    "human_id": human_id, "title": str(r.get("title") or ""),
                    "prompt_type": str(r.get("prompt_type") or ""),
                    "finish_reason": str(r.get("finish_reason") or "stop"),
                }
            )
    return records


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def clean_and_filter(human: list[dict], ai: list[dict]) -> tuple[list[dict], list[dict], Counter]:
    """Normalize every text, then apply the filters and the copy check.

    Returns (kept records, drops as {id, label, reason}, trim counters). Records are updated in place.
    """
    records = human + ai
    for rec in records:
        rec["text"] = normalize_text(rec["text"])
    human_text = {r["id"]: r["text"] for r in human}  # normalized, whether or not the row survives
    kept: list[dict] = []
    drops: list[dict] = []
    trims: Counter = Counter()
    for rec in records:
        is_ai = rec["label"] == 1
        res = filter_text(rec["text"], is_ai, truncated=is_ai and rec["finish_reason"] in TRUNCATED_FINISH_REASONS)
        if res.reason is None and rec["prompt_type"] == PROMPT_WEB:
            if rec["human_id"] not in human_text:
                trims["web_hint_without_human_text"] += 1  # cannot be checked
            elif copies_hint(res.text, human_text[rec["human_id"]]):
                res.reason = "copied_hint"
        if res.reason:
            drops.append({"id": rec["id"], "label": rec["label"], "reason": res.reason})
            continue
        trims["ai_truncated_to_sentence_end"] += res.trimmed_truncated
        trims["ai_long_to_sentence_end"] += res.trimmed_long
        rec["text"] = res.text
        rec["length"] = len(res.text)
        kept.append(rec)
    return kept, drops, trims


def drop_duplicates(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the first of each id / exact text / identical-opening group (input order: human before AI)."""
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    seen_heads: set[str] = set()
    kept: list[dict] = []
    drops: list[dict] = []
    for rec in records:
        head = rec["text"][:NEAR_DUP_CHARS]
        if rec["id"] in seen_ids:
            reason = "duplicate_id"  # same AI row passed in twice (e.g. overlapping --ai files)
        elif rec["text"] in seen_texts:
            reason = "duplicate"
        elif head in seen_heads:
            reason = "near_duplicate"
        else:
            seen_ids.add(rec["id"])
            seen_texts.add(rec["text"])
            seen_heads.add(head)
            kept.append(rec)
            continue
        drops.append({"id": rec["id"], "label": rec["label"], "reason": reason})
    return kept, drops


def length_bin(length: int) -> int:
    return min(max(length - MIN_CHARS, 0) // BIN_WIDTH, N_BINS - 1)


def bin_label(index: int) -> str:
    low = MIN_CHARS + index * BIN_WIDTH
    return f"{low}-{low + BIN_WIDTH - (0 if index == N_BINS - 1 else 1)}"


def length_match(records: list[dict], seed: int = SEED) -> list[dict]:
    """Per length bin, downsample the larger class to the smaller class's count.

    Ids are sorted before sampling, so the result depends only on the data and the seed.
    """
    rng = random.Random(seed)
    by_bin: dict[int, dict[int, list[dict]]] = {}
    for rec in records:
        by_bin.setdefault(length_bin(rec["length"]), {0: [], 1: []})[rec["label"]].append(rec)
    kept: list[dict] = []
    for index in sorted(by_bin):
        groups = {label: sorted(recs, key=lambda r: r["id"]) for label, recs in by_bin[index].items()}
        n = min(len(groups[0]), len(groups[1]))
        for label in (0, 1):
            group = groups[label]
            kept += rng.sample(group, n) if len(group) > n else group
    return kept


def ks_statistic(a, b) -> float | None:
    """Two-sample Kolmogorov-Smirnov statistic: max gap between the two empirical CDFs."""
    a, b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    if not len(a) or not len(b):
        return None
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def ks_by_label(records: list[dict]) -> float | None:
    return ks_statistic(
        [r["length"] for r in records if r["label"] == 0], [r["length"] for r in records if r["label"] == 1]
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def group_by(records: list[dict], key) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(key(rec), []).append(rec)
    return dict(sorted(groups.items()))


def label_name(rec: dict) -> str:
    return LABEL_NAMES[rec["label"]]


def length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"n": 0}
    return {
        "n": len(lengths), "min": min(lengths), "median": statistics.median(lengths),
        "mean": round(statistics.mean(lengths), 1), "max": max(lengths),
    }


def latin_rate(records: list[dict]) -> float:
    """Latin-letter words per 1,000 characters."""
    chars = sum(len(r["text"]) for r in records)
    return round(1000 * sum(latin_word_count(r["text"]) for r in records) / chars, 3) if chars else 0.0


def summarize(records: list[dict]) -> dict:
    """Counts, length stats, Latin-word rates and sentence-end share of one set of records."""
    by_label, by_gen = group_by(records, label_name), group_by(records, lambda r: r["generator"])
    return {
        "n": len(records),
        "by_label": {k: len(v) for k, v in by_label.items()},
        "by_source": dict(sorted(Counter(r["source"] for r in records).items())),
        "by_label_source": {k: dict(sorted(Counter(r["source"] for r in v).items())) for k, v in by_label.items()},
        "by_generator": {k: len(v) for k, v in by_gen.items()},
        "length_by_label": {k: length_stats([r["length"] for r in v]) for k, v in by_label.items()},
        "length_by_generator": {k: length_stats([r["length"] for r in v]) for k, v in by_gen.items()},
        "latin_words_per_1000_chars": {
            "by_label": {k: latin_rate(v) for k, v in by_label.items()},
            "by_generator": {k: latin_rate(v) for k, v in by_gen.items()},
        },
        "ends_with_sentence_end": {
            k: round(sum(ends_with_sentence_end(r["text"]) for r in v) / len(v), 3) for k, v in by_label.items()
        },
    }


def count_drops(drops: list[dict]) -> dict:
    counts = {reason: {"human": 0, "ai": 0, "total": 0} for reason in DROP_REASONS}
    for d in drops:
        counts[d["reason"]][LABEL_NAMES[d["label"]]] += 1
        counts[d["reason"]]["total"] += 1
    return counts


def bin_table(before: list[dict], after: list[dict]) -> list[dict]:
    """Per length bin: human/AI counts before matching and the count kept per class."""
    rows = []
    for index in range(N_BINS):
        row = {"bin": bin_label(index)}
        for name, recs in (("before", before), ("after", after)):
            in_bin = [r for r in recs if length_bin(r["length"]) == index]
            row[f"{name}_human"] = sum(r["label"] == 0 for r in in_bin)
            row[f"{name}_ai"] = sum(r["label"] == 1 for r in in_bin)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_counts(title: str, s: dict) -> None:
    print(f"\n=== {title}: {s['n']} rows ===")
    print("per label:     " + ", ".join(f"{k} {v}" for k, v in s["by_label"].items()))
    print("per source:    " + ", ".join(f"{k} {v}" for k, v in s["by_source"].items()))
    for label, sources in s["by_label_source"].items():
        print(f"    {label:<8}   " + ", ".join(f"{k} {v}" for k, v in sources.items()))
    print("per generator: " + ", ".join(f"{k} {v}" for k, v in s["by_generator"].items()))


def print_lengths(title: str, s: dict) -> None:
    print(f"\n--- Length (chars), {title} ---")
    print(f"{'':<34}{'n':>7}{'min':>7}{'median':>8}{'mean':>8}{'max':>7}")
    rows = list(s["length_by_label"].items()) + [
        (k, v) for k, v in s["length_by_generator"].items() if k != HUMAN_GENERATOR
    ]
    for name, st in rows:
        if st["n"]:
            print(f"{name:<34}{st['n']:>7}{st['min']:>7}{st['median']:>8.0f}{st['mean']:>8.0f}{st['max']:>7}")


def print_drops(drop_counts: dict, n_in: dict) -> None:
    print("\n=== Drops by reason and class ===")
    print(f"{'reason':<28}{'human':>8}{'ai':>8}{'total':>8}")
    for reason, c in drop_counts.items():
        print(f"{reason:<28}{c['human']:>8}{c['ai']:>8}{c['total']:>8}")
    totals = {k: sum(c[k] for c in drop_counts.values()) for k in ("human", "ai", "total")}
    print(f"{'total dropped':<28}{totals['human']:>8}{totals['ai']:>8}{totals['total']:>8}")
    print(f"{'input rows':<28}{n_in['human']:>8}{n_in['ai']:>8}{n_in['human'] + n_in['ai']:>8}")


def print_bins(rows: list[dict]) -> None:
    print("\n--- Length bins (human / ai), before -> after matching ---")
    for r in rows:
        print(
            f"{r['bin']:>10}   {r['before_human']:>6} / {r['before_ai']:<6} -> {r['after_human']:>5} / {r['after_ai']:<5}"
        )


def print_samples(records: list[dict], seed: int = SEED, k: int = 3, chars: int = 200) -> None:
    print(f"\n=== {k} random samples (seed {seed}), first {chars} chars ===")
    for rec in random.Random(seed).sample(records, min(k, len(records))):
        print(f"[{label_name(rec)} | {rec['generator']} | {rec['source']} | {rec['length']} chars]")
        print("  " + rec["text"][:chars].replace("\n", " | "))


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps({k: rec[k] for k in OUTPUT_FIELDS}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean, deduplicate and length-match human vs AI Khmer text.")
    p.add_argument("--human", type=Path, default=HUMAN_CSV, help="human passages CSV (default: %(default)s)")
    p.add_argument("--ai", type=Path, nargs="+", default=[AI_PATH], help="AI JSONL file(s) (default: %(default)s)")
    p.add_argument("--out", type=Path, default=OUT_PATH, help="output JSONL (default: %(default)s)")
    p.add_argument("--stats", type=Path, default=STATS_PATH, help="stats JSON (default: %(default)s)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-length-match", action="store_true", help="skip the per-bin downsampling")
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Khmer must not crash a cp1252 console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)

    human = load_human(args.human)
    ai = load_ai(args.ai)
    if not ai:
        print("error: no AI records found in " + ", ".join(str(p) for p in args.ai), file=sys.stderr)
        return 2
    n_in = {"human": len(human), "ai": len(ai)}
    print(f"Input: {len(human)} human rows, {len(ai)} AI rows from {', '.join(p.name for p in args.ai)}")
    finish_reasons = dict(sorted(Counter(r["finish_reason"] for r in ai).items()))
    print("AI finish_reason: " + ", ".join(f"{k} {v}" for k, v in finish_reasons.items()))

    kept, drops, trims = clean_and_filter(human, ai)
    kept, dup_drops = drop_duplicates(kept)
    drops += dup_drops
    drop_counts = count_drops(drops)
    print_drops(drop_counts, n_in)
    print(
        f"\nAI trimmed to a sentence end: {trims['ai_truncated_to_sentence_end']} cut-off by the token cap, "
        f"{trims['ai_long_to_sentence_end']} longer than {MAX_CHARS} chars"
    )
    if trims["web_hint_without_human_text"]:
        print(f"WARNING: {trims['web_hint_without_human_text']} web_hint rows have no linked human text; copy check skipped")

    after_cleaning = summarize(kept)
    ks_before = ks_by_label(kept)
    if args.no_length_match:
        final = kept
    else:
        final = length_match(kept, args.seed)
    ks_after = ks_by_label(final)
    bins = bin_table(kept, final)
    summary = summarize(final)

    print_counts("After cleaning and dedup, before length matching", after_cleaning)
    print_lengths("before length matching", after_cleaning)
    if args.no_length_match:
        print("\nLength matching skipped (--no-length-match).")
    else:
        print_bins(bins)
        print_counts("After length matching (final)", summary)
        print_lengths("after length matching (final)", summary)
    print(f"\nKS statistic, human vs ai length: before {ks_before}, after {ks_after}")
    print("Latin-letter words per 1,000 chars: " + ", ".join(
        f"{stage} " + "/".join(f"{k} {v}" for k, v in s["latin_words_per_1000_chars"]["by_label"].items())
        for stage, s in (("before matching", after_cleaning), ("final", summary))
    ))
    print("Share of texts ending in a sentence end: " + ", ".join(
        f"{k} {v}" for k, v in summary["ends_with_sentence_end"].items()
    ))
    if not summary["by_label"].get("human") or not summary["by_label"].get("ai"):
        print("WARNING: one class is empty in the final set", file=sys.stderr)

    output = sorted(final, key=lambda r: r["id"])  # sort first so the shuffle only depends on the data
    random.Random(args.seed).shuffle(output)
    write_jsonl(args.out, output)

    stats = {
        "config": {
            "seed": args.seed, "min_chars": MIN_CHARS, "max_chars": MAX_CHARS, "min_khmer_ratio": MIN_KHMER_RATIO,
            "bin_width": BIN_WIDTH, "copy_check_chars": COPY_CHECK_CHARS, "copy_min_span": COPY_MIN_SPAN,
            "near_dup_chars": NEAR_DUP_CHARS, "length_match": not args.no_length_match,
            "human_file": str(args.human), "ai_files": [str(p) for p in args.ai],
        },
        "input": {
            **n_in,
            "ai_by_generator": dict(sorted(Counter(r["generator"] for r in ai).items())),
            "ai_by_prompt_type": dict(sorted(Counter(r["prompt_type"] for r in ai).items())),
            "ai_by_finish_reason": finish_reasons,
        },
        "trimmed": {k: v for k, v in trims.items() if k.startswith("ai_")},
        "drops": drop_counts,
        "after_cleaning": after_cleaning,
        "length_match_bins": bins,
        "final": summary,
        "ks_length": {"before_matching": ks_before, "after_matching": ks_after},
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print_samples(output, args.seed)
    print(f"\nWrote {len(output)} records to {args.out}\nWrote stats to {args.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
