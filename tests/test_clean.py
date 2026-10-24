"""Tests for the pure functions in src/clean.py (run: python -m pytest tests)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import clean  # noqa: E402

SENTENCE = "ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា។"  # 25 chars, ends with the Khmer khan
LONG_TEXT = " ".join([SENTENCE] * 20)  # 519 chars, within 300-2500


def rec(id_, label, length, text=None, finish_reason="stop", prompt_type=""):
    return {
        "id": id_, "label": label, "length": length, "text": text or id_, "source": "wikipedia",
        "generator": "human" if label == 0 else "m", "human_id": id_, "prompt_type": prompt_type,
        "finish_reason": finish_reason,
    }


# -- normalize_text ---------------------------------------------------------
def test_normalize_removes_invisible_chars():
    text = "ខ្ញុំ​ស្រលាញ់‌‍ប្រទេស⁠កម្ពុជា﻿។"
    assert clean.normalize_text(text) == "ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា។"


def test_normalize_removes_empty_parentheses():
    assert clean.normalize_text("() ស្ថិតនៅក្នុងខេត្ត") == "ស្ថិតនៅក្នុងខេត្ត"
    assert clean.normalize_text("ភ្នំពេញ ( ) ជារាជធានី") == "ភ្នំពេញ ជារាជធានី"
    assert clean.normalize_text("ភ្នំពេញ (, ; ) ជារាជធានី") == "ភ្នំពេញ ជារាជធានី"
    assert clean.normalize_text("ក (()) ខ") == "ក ខ"


def test_normalize_keeps_non_empty_parentheses():
    assert clean.normalize_text("ភ្នំពេញ (Phnom Penh) ជា") == "ភ្នំពេញ (Phnom Penh) ជា"


def test_normalize_collapses_whitespace():
    text = "  ក  ខ\t\tគ   ឃ  \n\n\n  ង  \r\n\r\n  ច ឆ \n"
    assert clean.normalize_text(text) == "ក ខ គ ឃ\nង\nច\nឆ"


def test_normalize_is_idempotent_and_nfc():
    text = "á​ ( ) x \n\n y"
    once = clean.normalize_text(text)
    assert once == "á x\ny"  # NFC composes a + combining acute
    assert clean.normalize_text(once) == once


def test_normalize_same_rules_for_both_classes():
    # the gpt-oss artifact: U+200B between words must leave nothing behind
    assert clean.normalize_text("ក​ខ​គ") == clean.normalize_text("កខគ")


# -- trim_to_sentence_end ---------------------------------------------------
def test_trim_cuts_after_last_sentence_end():
    assert clean.trim_to_sentence_end("ក។ ខ។ គឃ") == "ក។ ខ។"
    assert clean.trim_to_sentence_end("ក។ ខ៕ គ") == "ក។ ខ៕"
    assert clean.trim_to_sentence_end("ក។ ខ? គ!  ឃ") == "ក។ ខ? គ!"
    assert clean.trim_to_sentence_end("ក។ «ខ។» គ") == "ក។ «ខ។»"  # keeps the closing quote


def test_trim_returns_empty_without_sentence_end():
    assert clean.trim_to_sentence_end("កខគ ឃង") == ""
    assert clean.trim_to_sentence_end("") == ""


def test_trim_respects_limit():
    text = "កកក។ ខខខ។ គគគ។"
    assert clean.trim_to_sentence_end(text, 10) == "កកក។ ខខខ។"
    assert clean.trim_to_sentence_end(text, 8) == "កកក។"  # the second end is cut off by the limit
    assert clean.trim_to_sentence_end(text, 3) == ""


def test_ends_with_sentence_end():
    assert clean.ends_with_sentence_end("កខ។")
    assert not clean.ends_with_sentence_end("កខ។ គ")
    assert not clean.ends_with_sentence_end("")


# -- khmer_ratio ------------------------------------------------------------
def test_khmer_ratio():
    assert clean.khmer_ratio("កខគ ឃង") == 1.0
    assert clean.khmer_ratio("hello world") == 0.0
    assert clean.khmer_ratio("") == 0.0
    assert clean.khmer_ratio("   \n") == 0.0
    assert clean.khmer_ratio("កខ ab") == pytest.approx(0.5)  # spaces are not counted
    assert clean.khmer_ratio("កក᧠᧡ab") == pytest.approx(4 / 6)  # Khmer symbols block counts


# -- filter_text ------------------------------------------------------------
def test_filter_keeps_good_text():
    res = clean.filter_text(LONG_TEXT, is_ai=True)
    assert res.reason is None and res.text == LONG_TEXT


def test_filter_drops_low_khmer_ratio_and_short():
    assert clean.filter_text("hello " * 100, is_ai=False).reason == "low_khmer_ratio"
    assert clean.filter_text(SENTENCE, is_ai=False).reason == "too_short"


def test_filter_truncated_ai_is_trimmed_or_dropped():
    cut = LONG_TEXT + " ខ្ញុំស្រ"  # cut off mid-word
    res = clean.filter_text(cut, is_ai=True, truncated=True)
    assert res.reason is None and res.text == LONG_TEXT and res.trimmed_truncated
    assert clean.filter_text("ក" * 400, is_ai=True, truncated=True).reason == "truncated_no_sentence_end"
    short = clean.filter_text(SENTENCE + " ខ្ញុំស្រ", is_ai=True, truncated=True)
    assert short.reason == "too_short"  # trimmed to one sentence, below the minimum


def test_filter_long_ai_trimmed_back_but_long_human_dropped():
    text = " ".join([SENTENCE] * 120)  # 3,119 chars
    res = clean.filter_text(text, is_ai=True)
    assert res.reason is None and res.trimmed_long
    assert res.text.endswith("។") and clean.MIN_CHARS <= len(res.text) <= clean.MAX_CHARS
    assert clean.filter_text(text, is_ai=False).reason == "too_long"
    assert clean.filter_text("ក" * 3000, is_ai=True).reason == "too_long"  # no sentence end to trim to


# -- copies_hint ------------------------------------------------------------
def test_copies_hint():
    human = "ក្រុមហ៊ុនបានប្រកាសថា " + LONG_TEXT
    copied = "សរុបមក៖ " + human[:150]  # contains a 60+ char verbatim span
    assert clean.copies_hint(copied, human)
    assert not clean.copies_hint("ខ្មែរ " * 80, human)
    assert not clean.copies_hint("", human)


def test_copies_hint_ignores_newline_vs_space_and_late_copies():
    human = "ក" * 30 + " " + "ខ" * 40 + " " + "គ" * 100
    assert clean.copies_hint("ក" * 30 + "\n" + "ខ" * 40 + "\n" + "គ" * 10, human)  # newline == space
    late = "ឃ" * 300 + human  # the copy starts after the first 300 chars: not checked
    assert not clean.copies_hint(late, human)


# -- drop_duplicates --------------------------------------------------------
def test_drop_duplicates_keeps_first():
    head = "ក" * 150
    records = [
        rec("h1", 0, 300, text="ក" * 300),
        rec("h2", 0, 300, text="ក" * 300),  # exact duplicate of h1
        rec("h3", 0, 300, text=head + "ខ" * 150),  # same first 150 chars as h1: near-duplicate
        rec("a1", 1, 300, text="គ" * 300),
        rec("a1", 1, 300, text="ឃ" * 300),  # same id passed twice
        rec("a2", 1, 300, text="ង" * 300),
    ]
    kept, drops = clean.drop_duplicates(records)
    assert [r["id"] for r in kept] == ["h1", "a1", "a2"]
    assert [(d["id"], d["reason"]) for d in drops] == [
        ("h2", "duplicate"), ("h3", "near_duplicate"), ("a1", "duplicate_id"),
    ]


# -- length matching --------------------------------------------------------
def test_length_bin_edges():
    assert clean.length_bin(300) == 0
    assert clean.length_bin(499) == 0
    assert clean.length_bin(500) == 1
    assert clean.length_bin(2299) == clean.N_BINS - 2
    assert clean.length_bin(2300) == clean.N_BINS - 1
    assert clean.length_bin(2500) == clean.N_BINS - 1  # the last bin includes MAX_CHARS
    assert clean.bin_label(0) == "300-499" and clean.bin_label(clean.N_BINS - 1) == "2300-2500"


def make_pool():
    humans = [rec(f"h{i:03d}", 0, 300 + (i * 37) % 2200) for i in range(400)]
    ais = [rec(f"a{i:03d}", 1, 1200 + (i * 53) % 1300) for i in range(150)]
    return humans + ais


def counts_per_bin(records):
    counts = {}
    for r in records:
        counts.setdefault(clean.length_bin(r["length"]), [0, 0])[r["label"]] += 1
    return counts


def test_length_match_gives_equal_counts_per_bin():
    pool = make_pool()
    matched = clean.length_match(pool, seed=42)
    assert matched
    for index, (n_human, n_ai) in counts_per_bin(matched).items():
        assert n_human == n_ai, index
    before = counts_per_bin(pool)
    after = counts_per_bin(matched)
    for index, (n_human, n_ai) in after.items():
        assert n_human == min(before[index])  # the smaller class is kept whole
    # bins that only one class occupies are emptied
    assert set(after) == {i for i, (h, a) in before.items() if h and a}


def test_length_match_is_deterministic_and_order_independent():
    pool = make_pool()
    first = clean.length_match(pool, seed=42)
    again = clean.length_match(list(reversed(pool)), seed=42)
    assert {r["id"] for r in first} == {r["id"] for r in again}
    other = clean.length_match(pool, seed=7)
    assert {r["id"] for r in other} != {r["id"] for r in first}
    assert {r["id"] for r in first if r["label"] == 1} == {
        r["id"] for r in other if r["label"] == 1
    }  # AI is the smaller class in the shared bins, so it is never sampled


def test_length_match_handles_one_empty_class():
    assert clean.length_match([rec("h1", 0, 500), rec("h2", 0, 900)]) == []


# -- ks_statistic -----------------------------------------------------------
def test_ks_statistic():
    assert clean.ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0
    assert clean.ks_statistic([1, 2, 3], [10, 11, 12]) == 1.0
    assert clean.ks_statistic([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)
    assert clean.ks_statistic([], [1, 2]) is None


# -- latin_word_count -------------------------------------------------------
def test_latin_word_count():
    assert clean.latin_word_count("ក្រុមហ៊ុន Apple និង Google ១២៣ COVID-19") == 3
    assert clean.latin_word_count("ភាសាខ្មែរសុទ្ធ") == 0
