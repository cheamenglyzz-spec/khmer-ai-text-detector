"""Step 3: clean, deduplicate and length-match human vs AI Khmer text into one labelled dataset.

Both classes go through exactly the same normalization and structure cleaning, so the
detector cannot learn formatting artifacts (e.g. gpt-oss puts U+200B between Khmer words,
human text does not; Wikipedia passages carry "See also" footers and one-word headings).

Pipeline: normalize -> cut footer sections -> strip list markers -> drop residue lines (site
          titles, URLs, wikitext) -> drop short fragment lines -> trim to the last sentence end
          -> Khmer-ratio / length filters -> copied-hint check -> deduplication -> length
          matching -> shuffle -> write.
The per-text part (everything up to the length filters) lives in `process_text`; step 2
(generate_ai.py) calls `clean_human_text` so it only prompts for passages that survive it.

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
MIN_LINE_CHARS = 40  # a line shorter than this that does not end a sentence is a fragment (heading, nav, list item)
TRUNCATED_FINISH_REASONS = {"length"}  # the API cut these off at the token cap, mid-word
SENTENCE_ENDS = "។៕?!"  # Khmer khan and bariyoosan, plus ? and !
CLOSERS = "\"'”’»)]"  # closing quotes/brackets that may follow a sentence end

# Invisible characters, removed from BOTH classes. First five: zero-width space/non-joiner/
# joiner, word joiner, BOM. The rest only showed up in the human text (soft hyphen, LRM/RLM,
# bidi embeddings, invisible operators); gone from both so they cannot become a feature.
INVISIBLE_CHARS = (
    "\u200b\u200c\u200d\u2060\ufeff"
    "\u00ad\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2061\u2062\u2063\u2064"
)

# A short line (< MIN_LINE_CHARS) starting with one of these (case-insensitive) is a footer
# heading: it and everything after it is cut. Spelling variants are listed on purpose.
FOOTER_MARKERS = (
    "មើលផងដែរ", "សូមមើលផងដែរ", "See also",  # See also
    "ឯកសារយោង", "ឯកសារពិគ្រោះ",  # References / consulted documents
    "គន្ថនិទ្ទេស", "អានបន្ថែម", "ការអានបន្ថែម",  # Bibliography / further reading
    "តំណភ្ជាប់", "តំនភ្ជាប់", "តំណរភ្ជាប់", "តំណខាងក្រៅ", "តំណក្រៅ", "តំនក្រៅ",  # External links
    "កំណត់សម្គាល់", "កំណត់សំគាល់", "កំណត់ចំណាំ", "Notes",  # Notes
    "ប្រភពឯកសារ",  # Sources
    "ចំណាត់ថ្នាក់ក្រុម", "Category:",  # Category
)

# Residue lines, dropped from both classes: web/wiki leftovers an AI text never contains.
SITE_SEPARATOR = " | "  # "headline | site-name" title lines
WIKITEXT_STARTS = ("{|", "|}", "|-", "{{", "!", "Short description")  # table / template / maintenance lines

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
_LINE_BREAKS = re.compile(r"\r\n|[\r\x0b\x0c\x85\u2028\u2029]")
_CONTROLS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]")
_INLINE_SPACE = re.compile(r"[^\S\n]+")  # any Unicode whitespace except the newline
# "()", "( )" and Wikipedia template leftovers like "(, ;)": nothing but spaces and separators.
_EMPTY_PARENS = re.compile(r"\([\s,;:]*\)")
_SENTENCE_END = re.compile(f"[{SENTENCE_ENDS}]+[{re.escape(CLOSERS)}]*")
# Leading bullet or numbering marker. A dash or a number needs a space after it, so that
# "-5" and "2.5 លាន" stay intact; "*", "•" and "·" may sit right against the text.
_LIST_MARKER = re.compile(r"(?:[*•·]+\s*|-+\s+|[0-9០-៩]+[.)]\s+)")
_FOOTER_PREFIXES = tuple(m.lower() for m in FOOTER_MARKERS)
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_TEMPLATE_PARAM = re.compile(r"\|[^=|]{1,40}=")  # "| name = value", matched at the start of a line
_KHMER_CHAR = re.compile("[\u1780-\u17ff\u19e0-\u19ff]")
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


def is_footer_heading(line: str) -> bool:
    return len(line) < MIN_LINE_CHARS and line.lower().startswith(_FOOTER_PREFIXES)


def cut_footer(text: str) -> tuple[str, bool]:
    """Cut the text at its first footer heading (See also, References, External links, ...)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if is_footer_heading(line):
            return "\n".join(lines[:i]), True
    return text, False


def strip_list_markers(text: str) -> tuple[str, int]:
    """Remove a leading bullet / "1." marker from each line; returns (text, lines changed)."""
    lines: list[str] = []
    changed = 0
    for line in text.split("\n"):
        stripped = _LIST_MARKER.match(line)
        if stripped:
            changed += 1
            line = line[stripped.end():]
        if line:  # a line that held only a marker disappears
            lines.append(line)
    return "\n".join(lines), changed


def is_residue_line(line: str) -> bool:
    """A site-name title (" | "), a URL, or a wikitext table / template line."""
    return (
        SITE_SEPARATOR in line
        or bool(_URL.search(line))
        or line.startswith(WIKITEXT_STARTS)
        or bool(_TEMPLATE_PARAM.match(line))
    )


def drop_residue_lines(text: str) -> tuple[str, int]:
    """Drop residue lines; returns (text, lines dropped)."""
    lines = text.split("\n")
    kept = [line for line in lines if not is_residue_line(line)]
    return "\n".join(kept), len(lines) - len(kept)


def drop_short_lines(text: str) -> tuple[str, int]:
    """Drop lines under MIN_LINE_CHARS that do not end a sentence; returns (text, lines dropped)."""
    lines = text.split("\n")
    kept = [line for line in lines if len(line) >= MIN_LINE_CHARS or ends_with_sentence_end(line)]
    return "\n".join(kept), len(lines) - len(kept)


@dataclass
class Structured:
    text: str
    footer_cut: bool = False
    markers_stripped: int = 0  # lines that lost a list marker
    residue_dropped: int = 0  # site-title / URL / wikitext lines removed
    lines_dropped: int = 0  # fragment lines removed


def clean_structure(text: str) -> Structured:
    """Footer, list-marker, residue-line and fragment-line rules for one already-normalized text."""
    text, footer_cut = cut_footer(text)
    text, markers = strip_list_markers(text)
    text, residue = drop_residue_lines(text)
    text, dropped = drop_short_lines(text)
    return Structured(text, footer_cut, markers, residue, dropped)


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
    trimmed_end: bool = False  # text cut back to its last sentence end
    trimmed_truncated: bool = False  # ...and it was an AI text the API cut off at the token cap
    trimmed_long: bool = False  # AI text over MAX_CHARS, trimmed to a sentence end


def filter_text(text: str, is_ai: bool, truncated: bool = False) -> Filtered:
    """Sentence-end, Khmer-ratio and length rules for one structure-cleaned text.

    Every text is trimmed to its last sentence end (a trailing fragment is a chunk-boundary or
    token-cap artifact in both classes). `truncated` only labels the stats.
    """
    out = Filtered(text)
    trimmed = trim_to_sentence_end(text)
    if not trimmed:
        out.reason = "truncated_no_sentence_end"
        return out
    out.trimmed_end = trimmed != text
    out.trimmed_truncated = truncated and out.trimmed_end
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


@dataclass
class Processed:
    normalized: str  # after normalize_text, before any structure rule
    structure: Structured
    filtered: Filtered  # .text is the final cleaned text; .reason is set if the text is dropped


def process_text(text: str, is_ai: bool, truncated: bool = False) -> Processed:
    """The whole per-text rule set, in order: normalize -> structure cleaning -> sentence-end trim and filters.

    Step 3 runs this on every row; generate_ai.py runs it (via clean_human_text) on the human passages.
    """
    normalized = normalize_text(text)
    structure = clean_structure(normalized)
    return Processed(normalized, structure, filter_text(structure.text, is_ai, truncated))


def clean_human_text(text: str) -> str | None:
    """The cleaned text of a human passage, or None if step 3 would drop it (no sentence end, not Khmer, bad length).

    Cross-row rules (copy check, deduplication, length matching) are not applied here.
    """
    res = process_text(text, is_ai=False).filtered
    return None if res.reason else res.text


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
@dataclass
class CleanRun:
    kept: list[dict]
    drops: list[dict]  # {id, label, reason}
    trims: Counter  # AI trim counters and the web_hint-without-human warning
    events: dict[str, Counter]  # structure-cleaning events per class name
    before: dict  # structure_stats of the normalized input, before structure cleaning


def clean_and_filter(human: list[dict], ai: list[dict]) -> CleanRun:
    """Run process_text on every record, then the copy check. Updates records in place."""
    records = human + ai
    processed = [
        process_text(r["text"], r["label"] == 1, truncated=r["finish_reason"] in TRUNCATED_FINISH_REASONS)
        for r in records
    ]
    before = structure_stats([{"label": r["label"], "text": p.normalized} for r, p in zip(records, processed)])
    # step 2 builds the web_hint prompt from the cleaned human text, so the copy check compares against
    # that text too (kept for every human row, also those step 3 drops, since an AI row may link to it)
    human_text = {r["id"]: p.filtered.text for r, p in zip(records, processed) if r["label"] == 0}
    kept: list[dict] = []
    drops: list[dict] = []
    trims: Counter = Counter()
    events = {name: Counter() for name in LABEL_NAMES.values()}
    for rec, proc in zip(records, processed):
        ev = events[label_name(rec)]
        st, res = proc.structure, proc.filtered
        ev["texts"] += 1
        ev["footer_cut"] += st.footer_cut
        ev["texts_markers_stripped"] += st.markers_stripped > 0
        ev["markers_stripped"] += st.markers_stripped
        ev["texts_residue_dropped"] += st.residue_dropped > 0
        ev["residue_dropped"] += st.residue_dropped
        ev["texts_lines_dropped"] += st.lines_dropped > 0
        ev["lines_dropped"] += st.lines_dropped
        ev["end_trimmed"] += res.trimmed_end
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
    return CleanRun(kept, drops, trims, events, before)


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


def structure_stats(records: list[dict]) -> dict:
    """Per class: share of texts ending in a sentence end, share of lines under MIN_LINE_CHARS, mean lines."""
    out = {}
    for name, recs in group_by(records, label_name).items():
        lines = [line for r in recs for line in r["text"].split("\n")]
        out[name] = {
            "n": len(recs),
            "ends_with_sentence_end": round(sum(ends_with_sentence_end(r["text"]) for r in recs) / len(recs), 3),
            "short_line_share": round(sum(len(line) < MIN_LINE_CHARS for line in lines) / len(lines), 3),
            "mean_lines": round(len(lines) / len(recs), 2),
        }
    return out


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
        "structure": structure_stats(records),
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


STRUCTURE_EVENT_ROWS = (
    ("texts in", "texts"),
    ("footer section cut", "footer_cut"),
    ("texts with fragment lines dropped", "texts_lines_dropped"),
    ("fragment lines dropped", "lines_dropped"),
    ("texts with residue lines dropped", "texts_residue_dropped"),
    ("residue lines dropped (| / URL / wikitext)", "residue_dropped"),
    ("texts with list markers stripped", "texts_markers_stripped"),
    ("lines with list marker stripped", "markers_stripped"),
    ("texts trimmed to last sentence end", "end_trimmed"),
)
STRUCTURE_METRIC_ROWS = (
    ("share ending in a sentence end", "ends_with_sentence_end"),
    (f"share of lines under {MIN_LINE_CHARS} chars", "short_line_share"),
    ("mean lines per text", "mean_lines"),
)


def print_structure(events: dict, before: dict, after: dict) -> None:
    classes = list(LABEL_NAMES.values())
    print("\n=== Structure cleaning (same rules for both classes) ===")
    print(f"{'':<46}" + "".join(f"{c:>10}" for c in classes))
    for title, key in STRUCTURE_EVENT_ROWS:
        print(f"{title:<46}" + "".join(f"{events[c][key]:>10}" for c in classes))
    print(f"\n{'before (all input) -> after (kept rows)':<46}" + "".join(f"{c:>20}" for c in classes))
    for title, key in STRUCTURE_METRIC_ROWS:
        cells = [f"{before.get(c, {}).get(key, '-')} -> {after.get(c, {}).get(key, '-')}" for c in classes]
        print(f"{title:<46}" + "".join(f"{cell:>20}" for cell in cells))


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

    run = clean_and_filter(human, ai)
    trims = run.trims
    kept, dup_drops = drop_duplicates(run.kept)
    drop_counts = count_drops(run.drops + dup_drops)
    after_cleaning = summarize(kept)
    print_structure(run.events, run.before, after_cleaning["structure"])
    print_drops(drop_counts, n_in)
    print(
        f"\nAI trimmed to a sentence end: {trims['ai_truncated_to_sentence_end']} cut-off by the token cap, "
        f"{trims['ai_long_to_sentence_end']} longer than {MAX_CHARS} chars"
    )
    if trims["web_hint_without_human_text"]:
        print(f"WARNING: {trims['web_hint_without_human_text']} web_hint rows have no linked human text; copy check skipped")

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
    if not summary["by_label"].get("human") or not summary["by_label"].get("ai"):
        print("WARNING: one class is empty in the final set", file=sys.stderr)

    output = sorted(final, key=lambda r: r["id"])  # sort first so the shuffle only depends on the data
    random.Random(args.seed).shuffle(output)
    write_jsonl(args.out, output)

    stats = {
        "config": {
            "seed": args.seed, "min_chars": MIN_CHARS, "max_chars": MAX_CHARS, "min_khmer_ratio": MIN_KHMER_RATIO,
            "bin_width": BIN_WIDTH, "copy_check_chars": COPY_CHECK_CHARS, "copy_min_span": COPY_MIN_SPAN,
            "near_dup_chars": NEAR_DUP_CHARS, "min_line_chars": MIN_LINE_CHARS,
            "footer_markers": list(FOOTER_MARKERS), "length_match": not args.no_length_match,
            "human_file": str(args.human), "ai_files": [str(p) for p in args.ai],
        },
        "input": {
            **n_in,
            "ai_by_generator": dict(sorted(Counter(r["generator"] for r in ai).items())),
            "ai_by_prompt_type": dict(sorted(Counter(r["prompt_type"] for r in ai).items())),
            "ai_by_finish_reason": finish_reasons,
        },
        "trimmed": {k: v for k, v in trims.items() if k.startswith("ai_")},
        "structure_cleaning": {
            "events": {name: dict(ev) for name, ev in run.events.items()},
            "before_all_input": run.before,
            "after_kept_rows": after_cleaning["structure"],
        },
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
