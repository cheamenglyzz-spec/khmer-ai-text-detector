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
    text = "ខ្ញុំ\u200bស្រលាញ់\u200c\u200dប្រទេស\u2060កម្ពុជា\ufeff។"
    assert clean.normalize_text(text) == "ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា។"


def test_normalize_removes_empty_parentheses():
    assert clean.normalize_text("() ស្ថិតនៅក្នុងខេត្ត") == "ស្ថិតនៅក្នុងខេត្ត"
    assert clean.normalize_text("ភ្នំពេញ ( ) ជារាជធានី") == "ភ្នំពេញ ជារាជធានី"
    assert clean.normalize_text("ភ្នំពេញ (, ; ) ជារាជធានី") == "ភ្នំពេញ ជារាជធានី"
    assert clean.normalize_text("ក (()) ខ") == "ក ខ"


def test_normalize_keeps_non_empty_parentheses():
    assert clean.normalize_text("ភ្នំពេញ (Phnom Penh) ជា") == "ភ្នំពេញ (Phnom Penh) ជា"


def test_normalize_collapses_whitespace():
    text = "  ក\u00a0\u00a0ខ\t\tគ \u202f ឃ  \n\n\n  ង  \r\n\r\n  ច\u2028ឆ \n"
    assert clean.normalize_text(text) == "ក ខ គ ឃ\nង\nច\nឆ"


def test_normalize_is_idempotent_and_nfc():
    text = "a\u0301\u200b ( ) x \n\n y"
    once = clean.normalize_text(text)
    assert once == "á x\ny"  # NFC composes a + combining acute
    assert clean.normalize_text(once) == once


def test_normalize_same_rules_for_both_classes():
    # the gpt-oss artifact: U+200B between words must leave nothing behind
    assert clean.normalize_text("ក\u200bខ\u200bគ") == clean.normalize_text("កខគ")


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
    assert clean.filter_text("hello " * 100 + "!", is_ai=False).reason == "low_khmer_ratio"
    assert clean.filter_text(SENTENCE, is_ai=False).reason == "too_short"


def test_filter_trims_every_text_to_its_last_sentence_end():
    tail = LONG_TEXT + " ខ្ញុំស្រ"  # a trailing fragment, no token cap involved
    human = clean.filter_text(tail, is_ai=False)
    assert human.reason is None and human.text == LONG_TEXT
    assert human.trimmed_end and not human.trimmed_truncated
    ai = clean.filter_text(tail, is_ai=True, truncated=True)
    assert ai.trimmed_end and ai.trimmed_truncated
    assert not clean.filter_text(LONG_TEXT, is_ai=False).trimmed_end
    assert clean.filter_text("ក" * 400, is_ai=False).reason == "truncated_no_sentence_end"


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
    assert clean.filter_text("ក" * 3000 + "។", is_ai=True).reason == "too_long"  # no sentence end within 2,500


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


# -- structure cleaning -----------------------------------------------------
def test_footer_headings_cover_every_marker():
    for marker in clean.FOOTER_MARKERS:
        assert clean.is_footer_heading(marker), marker
        assert clean.is_footer_heading(marker.upper() + " x"), marker  # case-insensitive, extra words allowed


def test_cut_footer_cuts_at_first_footer_heading():
    body = LONG_TEXT + "\n" + SENTENCE
    text = body + "\nមើលផងដែរ\nខេត្តស្វាយរៀង\nឯកសារយោង\n" + SENTENCE
    assert clean.cut_footer(text) == (body, True)
    assert clean.cut_footer(body + "\nតំណភ្ជាប់ខាងក្រៅ\nhttp://x") == (body, True)
    assert clean.cut_footer(body + "\nSee also\nកម្ពុជា") == (body, True)
    assert clean.cut_footer(body + "\nCategory:Foo") == (body, True)


def test_cut_footer_leaves_other_lines_alone():
    assert clean.cut_footer(LONG_TEXT) == (LONG_TEXT, False)
    inline = "ខ្ញុំ មើលផងដែរ ទៅលើសៀវភៅ។"  # the marker is not at the start of the line
    assert clean.cut_footer(LONG_TEXT + "\n" + inline) == (LONG_TEXT + "\n" + inline, False)
    long_heading = "មើលផងដែរ " + "ក" * clean.MIN_LINE_CHARS  # not short: a real line, not a heading
    assert clean.cut_footer(LONG_TEXT + "\n" + long_heading)[1] is False
    assert clean.cut_footer("មើលផងដែរ\n" + LONG_TEXT) == ("", True)  # footer first: nothing left


@pytest.mark.parametrize(
    "line, expected",
    [
        ("* ក", "ក"), ("*ក", "ក"), ("** ក", "ក"), ("- ក", "ក"), ("• ក", "ក"), ("· ក", "ក"),
        ("១. ក", "ក"), ("១២) ក", "ក"), ("3. ក", "ក"), ("10) ក", "ក"),
        ("2.5 លាននាក់", "2.5 លាននាក់"), ("-5 ដឺក្រេ", "-5 ដឺក្រេ"), ("ក * ខ", "ក * ខ"),
        ("1990 ក", "1990 ក"), ("ក - ខ", "ក - ខ"), ("ក 1. ខ", "ក 1. ខ"),
    ],
)
def test_strip_list_markers_line(line, expected):
    assert clean.strip_list_markers(line)[0] == expected


def test_strip_list_markers_counts_and_drops_empty_lines():
    text, changed = clean.strip_list_markers("* ក\n1. ខ\n*\nគ")
    assert (text, changed) == ("ក\nខ\nគ", 3)  # the lone "*" line disappears
    assert clean.strip_list_markers("ក\nខ") == ("ក\nខ", 0)


def test_drop_short_lines():
    long_line = "ក" * clean.MIN_LINE_CHARS
    text = "\n".join(["ប្រវត្តិ", "ខ្ញុំស្រលាញ់។", long_line, "ខេត្តកណ្តាល", "ឆ្នាំ ២០២០?"])
    kept, dropped = clean.drop_short_lines(text)
    assert kept == "\n".join(["ខ្ញុំស្រលាញ់។", long_line, "ឆ្នាំ ២០២០?"])  # short line with a sentence end stays
    assert dropped == 2
    assert clean.drop_short_lines("ក" * (clean.MIN_LINE_CHARS - 1))[1] == 1
    assert clean.drop_short_lines("ក" * clean.MIN_LINE_CHARS)[1] == 0


def test_clean_structure_applies_rules_in_order():
    para = "ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា ហើយខ្ញុំក៏ស្រលាញ់ប្រជាជនខ្មែរទាំងអស់។"
    assert len(para) >= clean.MIN_LINE_CHARS
    text = "\n".join(["ប្រវត្តិ", "* " + para, "១. " + para, "ខេត្តកណ្តាល", "មើលផងដែរ", "ខេត្តកណ្តាល", "ឯកសារយោង"])
    res = clean.clean_structure(text)
    assert res.text == para + "\n" + para
    assert res.footer_cut and res.markers_stripped == 2 and res.lines_dropped == 2  # footer lines are not counted
    same = clean.clean_structure(para)
    assert same.text == para and not same.footer_cut and same.markers_stripped == same.lines_dropped == 0


def test_structure_stats():
    short = "ខ្ញុំ"
    recs = [
        {"label": 0, "text": LONG_TEXT},  # 1 line, ends in a sentence end
        {"label": 0, "text": LONG_TEXT + "\n" + short + "\nមើលផងដែរ"},  # 3 lines, 2 short, no sentence end
        {"label": 1, "text": SENTENCE + " " + LONG_TEXT},
    ]
    out = clean.structure_stats(recs)
    assert out["human"] == {"n": 2, "ends_with_sentence_end": 0.5, "short_line_share": 0.5, "mean_lines": 2.0}
    assert out["ai"] == {"n": 1, "ends_with_sentence_end": 1.0, "short_line_share": 0.0, "mean_lines": 1.0}


# -- residue lines ----------------------------------------------------------
@pytest.mark.parametrize(
    "line",
    [
        "headline | posts-society", "ព័ត៌មាន | ទំព័រ", "ក |  ខ",  # site-name separator
        "មើលបន្ថែមនៅ http://example.com/a", "មើលនៅ https://x.y", "សូមចូលទៅ www.example.com", "WWW.Example.com ក",  # URL
        '{| class="wikitable"', "|}", "|}ប្រធានាធិបតី", "|-", '|- style="vertical-align:top;"', "! ក", "{{Infobox ស៊ីហ្គេម |",
        "| image_caption = ប្រជាជន", "|population=៧", "| name = ក",  # table / template parameter
        "Short description is different from Wikidata", "Short description",
    ],
)
def test_is_residue_line_true(line):
    assert clean.is_residue_line(line)


@pytest.mark.parametrize(
    "line",
    [
        "«សួស្តី» ជាពាក្យ", "ក → ខ", "ក ← ខ", "ក » ខ",  # quotation marks and arrows are prose
        "Apple ជាក្រុមហ៊ុន", "ក|ខ", "ក |ខ", "| ក ខ គ", "ការបោះឆ្នោត! ជា", "ក Short description", "ក = ខ",
    ],
)
def test_is_residue_line_false(line):
    assert not clean.is_residue_line(line)


def test_drop_residue_lines_counts():
    text = "\n".join([LONG_TEXT, "headline | posts-society", SENTENCE, "http://x.com/ក", "|-"])
    assert clean.drop_residue_lines(text) == (LONG_TEXT + "\n" + SENTENCE, 3)
    assert clean.drop_residue_lines(LONG_TEXT) == (LONG_TEXT, 0)


def test_clean_structure_counts_residue_apart_from_fragments():
    title = "ព័ត៌មានថ្មីៗនៅកម្ពុជា ថ្ងៃនេះ | ស្រុកខ្មែរ"  # long enough to survive the fragment rule on its own
    assert len(title) >= clean.MIN_LINE_CHARS
    text = "\n".join(["ប្រវត្តិ", title, LONG_TEXT, "ទស្សនាតាម http://example.com"])
    res = clean.clean_structure(text)
    assert res.text == LONG_TEXT
    assert res.residue_dropped == 2 and res.lines_dropped == 1


# -- shared per-text rules (step 2 imports clean_human_text) ----------------
def test_clean_human_text_returns_cleaned_text_or_none():
    assert clean.clean_human_text("ប្រវត្តិ\n" + LONG_TEXT + "\nមើលផងដែរ\nខេត្តស្វាយរៀង") == LONG_TEXT
    assert clean.clean_human_text(LONG_TEXT + " ខ្ញុំស្រ") == LONG_TEXT  # trailing fragment trimmed
    assert clean.clean_human_text("ក" * 400) is None  # no sentence end
    assert clean.clean_human_text(SENTENCE) is None  # too short
    assert clean.clean_human_text("hello " * 100 + "!") is None  # not Khmer
    assert clean.clean_human_text("") is None


def test_process_text_exposes_each_stage():
    proc = clean.process_text(LONG_TEXT.replace(" ", "  ") + "\n\nមើលផងដែរ\nគ", is_ai=False)
    assert proc.normalized == LONG_TEXT + "\nមើលផងដែរ\nគ"
    assert proc.structure.footer_cut and proc.structure.text == LONG_TEXT
    assert proc.filtered.reason is None and proc.filtered.text == LONG_TEXT
    assert clean.process_text("ក" * 400, is_ai=False).filtered.reason == "truncated_no_sentence_end"


# -- clean_and_filter (whole pipeline up to dedup) ---------------------------
def pipeline_rec(id_, label, text, **extra):
    base = {
        "id": id_, "text": text, "label": label, "source": "wikipedia", "generator": "m", "url": "", "date": "",
        "length": 0, "human_id": id_, "title": "", "prompt_type": "", "finish_reason": "stop",
    }
    return {**base, **extra}


def test_clean_and_filter_end_to_end():
    other = " ".join(["ភ្នំពេញជារាជធានីនៃព្រះរាជាណាចក្រកម្ពុជា និងជាទីក្រុងធំជាងគេ។"] * 10)
    heading = "ក្រុមខេត្ត"
    k = clean.COPY_MIN_SPAN - len(heading)  # heading + space + k chars is 61 chars; without the heading only k
    raw_human = {
        "h1": heading + "\n" + LONG_TEXT + "\n* " + SENTENCE + "\nមើលផងដែរ\nខេត្តស្វាយរៀង",
        "h2": "ក" * 400,  # no sentence end at all
    }
    expected = {hid: clean.clean_human_text(text) for hid, text in raw_human.items()}
    human = [pipeline_rec(hid, 0, text) for hid, text in raw_human.items()]
    ai = [
        # copies the opening of the (cleaned) human text
        pipeline_rec("ai_h1", 1, LONG_TEXT[:120] + " " + other, human_id="h1", prompt_type="web_hint"),
        # shares only heading + k chars with it; the heading is not in the cleaned text the prompt was built from
        pipeline_rec("ai_h4", 1, heading + " " + LONG_TEXT[:k] + " " + other, human_id="h1", prompt_type="web_hint"),
        pipeline_rec("ai_h3", 1, other + " ខ្ញុំស្រ", human_id="h3", prompt_type="wiki_title"),
    ]
    run = clean.clean_and_filter(human, ai)
    kept = {r["id"]: r for r in run.kept}
    assert set(kept) == {"h1", "ai_h3", "ai_h4"}
    assert kept["h1"]["text"] == expected["h1"] == LONG_TEXT + "\n" + SENTENCE  # step 3 and step 2 agree
    assert expected["h2"] is None
    assert kept["h1"]["length"] == len(kept["h1"]["text"])
    assert kept["ai_h3"]["text"] == other  # trailing fragment trimmed
    assert {(d["id"], d["reason"]) for d in run.drops} == {
        ("h2", "truncated_no_sentence_end"), ("ai_h1", "copied_hint"),
    }
    assert run.events["human"]["footer_cut"] == 1
    assert run.events["human"]["markers_stripped"] == 1
    assert run.events["human"]["lines_dropped"] == 1  # the heading in h1; h2's single line is long enough
    assert run.events["human"]["residue_dropped"] == 0
    assert run.events["ai"]["end_trimmed"] == 1
    assert run.before["human"]["ends_with_sentence_end"] == 0.0
