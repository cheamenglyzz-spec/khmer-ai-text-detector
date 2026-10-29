"""Tests for src/evaluate.py and the step-7 additions to src/make_colab_bundle.py (run: python -m pytest tests).

The pure functions are tested on hand-made records. The scorers are tested with a stub, with the step-5 baseline
trained on synthetic splits, and with a tiny random XLM-R plus a character-level tokenizer built in a temp folder,
so everything runs offline on CPU in seconds."""
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baseline, clean, evaluate, make_colab_bundle  # noqa: E402

HUMAN_LETTERS = "កខគឃងចឆជ"
AI_LETTERS = "តថទធនបផព"


def sentence(rng, letters):
    return " ".join("".join(rng.choice(letters) for _ in range(rng.randint(3, 6))) for _ in range(4)) + "។"


def make_rows(split, n_per_class, seed):
    """Synthetic Khmer-looking text (8 sentences, ~190 chars): human rows use one set of letters, AI rows another."""
    rng = random.Random(seed)
    rows = []
    for label, letters in ((0, HUMAN_LETTERS), (1, AI_LETTERS)):
        for i in range(n_per_class):
            text = " ".join(sentence(rng, letters) for _ in range(8))
            hid = f"{split}{i:03d}"
            source = "wikipedia" if i % 2 else "mc4_web"
            generator = "human" if label == 0 else ("g/a" if i % 3 else "g/b")
            rows.append({
                "id": hid if label == 0 else f"ai_{hid}", "text": text, "label": label, "source": source, "generator": generator,
                "url": "", "date": "", "length": len(text), "human_id": hid, "title": "",
                "prompt_type": "" if label == 0 else ("wiki_title" if source == "wikipedia" else "web_hint"),
            })
    return rows


SIZES = {"train": 40, "val": 12, "test": 12, "test_ood": 8}


def write_splits(folder, sizes=SIZES):
    for i, (name, n) in enumerate(sizes.items()):
        clean.write_jsonl(folder / f"{name}.jsonl", make_rows(name, n, seed=i))


def rec(id, label, pred, p_ai, generator="g", source="s", prompt_type="", length=500, text="ក", split="test"):
    return {"id": id, "split": split, "label": label, "pred": pred, "p_ai": p_ai, "generator": generator, "source": source,
            "prompt_type": prompt_type, "length": length, "text": text}


class StubScorer:
    """p_ai from a function of the text; remembers what it was asked to score."""

    device = "cpu"
    tokenizer = None

    def __init__(self, fn):
        self.fn, self.seen = fn, []

    def score(self, texts):
        self.seen.append(list(texts))
        return np.array([self.fn(t) for t in texts], dtype=np.float64)


class FakeTokenizer:
    """One token per character plus <s> and </s>."""

    model_max_length = 512

    def __call__(self, texts, truncation=False, max_length=None, padding=False, add_special_tokens=True, **kwargs):
        ids = [[0] + [ord(c) % 90 + 4 for c in t] + [2] for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------
def test_length_bin_label_edges():
    assert [evaluate.length_bin_label(x) for x in (299, 300, 699, 700, 1099, 1100, 1899, 1900, 2500, 2501, None)] == [
        "<300", "300-699", "300-699", "700-1099", "700-1099", "1100-1499", "1500-1899", "1900-2500", "1900-2500", ">2500", "unknown",
    ]
    labels = {evaluate.length_bin_label(x) for x in range(250, 2600, 7)} | {"unknown"}
    assert labels == set(evaluate.LENGTH_BIN_ORDER)  # the order list covers every label


def test_cut_text_unchanged_when_short_enough():
    assert evaluate.cut_text("កខគ។", 4) == ("កខគ។", "unchanged")
    assert evaluate.cut_text("កខគ។", 100) == ("កខគ។", "unchanged")


def test_cut_text_goes_back_to_the_last_sentence_end():
    text = "កកកកកកកកក។ ខខខខខខខខខ។ គគគគគគគគគ។"
    assert evaluate.cut_text(text, 25) == ("កកកកកកកកក។ ខខខខខខខខខ។", "sentence")  # the window ends inside the third sentence
    assert evaluate.cut_text(text, 22) == ("កកកកកកកកក។ ខខខខខខខខខ។", "sentence")  # a mark right at the end of the window counts
    assert evaluate.cut_text("កកកកកកកកក៕ ខខខខខខខខខ", 15) == ("កកកកកកកកក៕", "sentence")  # bariyoosan too
    assert evaluate.cut_text("កកកកកកកកក។» ខខខខខខខខខ", 15) == ("កកកកកកកកក។»", "sentence")  # a closing quote stays with the sentence
    assert evaluate.cut_text("កក? ខខខខខខខខខខខខខខខខ", 16)[1] == "hard"  # only Khmer marks count, not "?"


def test_cut_text_cuts_hard_when_no_good_sentence_end_exists():
    assert evaluate.cut_text("ក" * 40, 25) == ("ក" * 25, "hard")
    early = "កក។" + "ខ" * 40  # the only mark is at 3 of 25 chars: keeping 12% of N is not worth it
    assert evaluate.cut_text(early, 25) == (early[:25], "hard")
    assert evaluate.cut_text(early, 25, min_keep=0.1) == ("កក។", "sentence")
    assert evaluate.cut_text("ក" * 24 + " " + "ខ" * 10, 25) == ("ក" * 24, "hard")  # a trailing space is removed


def test_cut_text_keeps_khmer_intact():
    text = sentence(random.Random(1), HUMAN_LETTERS) * 5
    cut, kind = evaluate.cut_text(text, 60)
    assert text.startswith(cut) and len(cut) <= 60 and kind in ("sentence", "hard")
    assert cut.encode("utf-8").decode("utf-8") == cut


def test_metrics_of_adds_human_accuracy():
    recs = [rec("a", 1, 1, 0.9), rec("b", 1, 0, 0.4), rec("c", 0, 0, 0.1), rec("d", 0, 0, 0.2), rec("e", 0, 1, 0.8)]
    m = evaluate.metrics_of(recs)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert m["human_accuracy"] == pytest.approx(2 / 3, abs=1e-4) and m["recall"] == 0.5 and m["n"] == 5
    assert 0.5 < m["roc_auc"] <= 1.0
    only_ai = evaluate.metrics_of([rec("a", 1, 1, 0.9)])
    assert only_ai["human_accuracy"] is None and only_ai["roc_auc"] is None


def test_group_metrics_per_group_rates_and_order():
    recs = [
        rec("a", 1, 1, 0.9, generator="g/a", length=400), rec("b", 1, 0, 0.3, generator="g/a", length=800),
        rec("c", 1, 1, 0.8, generator="g/b", length=800), rec("d", 0, 0, 0.1, generator="human", length=400),
        rec("e", 0, 1, 0.7, generator="human", length=800),
    ]
    by_gen = evaluate.group_metrics(recs, lambda r: r["generator"])
    assert list(by_gen) == ["g/a", "g/b", "human"]
    assert by_gen["g/a"] == {"n": 2, "n_ai": 2, "n_human": 0, "accuracy": 0.5, "ai_recall": 0.5, "human_accuracy": None, "mean_p_ai": 0.6, "roc_auc": None}
    assert by_gen["human"]["human_accuracy"] == 0.5 and by_gen["human"]["ai_recall"] is None
    by_len = evaluate.group_metrics(recs, lambda r: evaluate.length_bin_label(r["length"]), evaluate.LENGTH_BIN_ORDER)
    assert list(by_len) == ["300-699", "700-1099"]
    assert by_len["300-699"]["n"] == 2 and by_len["300-699"]["roc_auc"] == 1.0  # both classes: AUC is defined
    assert by_len["700-1099"]["roc_auc"] == pytest.approx(0.5)  # ai p 0.3 / 0.8 against human p 0.7


def test_breakdowns_of_keys_and_token_truncation_only_when_known():
    recs = [rec("a", 1, 1, 0.9, prompt_type="web_hint"), rec("b", 0, 0, 0.1)]
    b = evaluate.breakdowns_of(recs, 512)
    assert set(b) == {"by_generator", "by_source", "by_prompt_type", "by_length_bin"}
    assert set(b["by_prompt_type"]) == {"web_hint", "(human)"}
    recs[0].update(truncated=True, n_tokens=900)
    recs[1].update(truncated=False, n_tokens=200)
    b = evaluate.breakdowns_of(recs, 512)
    assert set(b["by_token_truncation"]) == {"over 512 tokens", "within 512 tokens"}
    assert b["by_token_truncation"]["over 512 tokens"]["n_ai"] == 1


def test_error_rows_sorted_by_confidence_and_capped():
    recs = [
        rec("fp_low", 0, 1, 0.6), rec("fn_high", 1, 0, 0.02, text="  ក\n ខ  " + "គ" * 400), rec("ok", 1, 1, 0.99),
        rec("fp_high", 0, 1, 0.97), rec("fn_mid", 1, 0, 0.3),
    ]
    e = evaluate.error_rows(recs)
    assert [x["id"] for x in e["listed"]] == ["fn_high", "fp_high", "fn_mid", "fp_low"]  # 0.98, 0.97, 0.70, 0.60
    assert e["n_errors"] == 4 and e["false_positives"] == 2 and e["false_negatives"] == 2 and e["omitted"] == 0
    assert e["listed"][0]["confidence"] == 0.98 and e["listed"][0]["p_ai"] == 0.02
    head = e["listed"][0]["text_head"]
    assert head.startswith("ក ខ គ") and len(head) == evaluate.HEAD_CHARS  # whitespace flattened, cut to ~150 chars
    capped = evaluate.error_rows(recs, limit=2)
    assert [x["id"] for x in capped["listed"]] == ["fn_high", "fp_high"] and capped["omitted"] == 2 and capped["n_errors"] == 4
    assert evaluate.error_rows([rec("ok", 1, 1, 0.9)]) == {"n_errors": 0, "false_positives": 0, "false_negatives": 0, "listed": [], "omitted": 0}


def test_records_from_predictions_joins_text_and_counts_problems():
    splits = {"val": make_rows("val", 2, 1), "test": make_rows("test", 2, 2)}
    val = splits["val"]
    preds = [
        {"id": r["id"], "split": "val", "label": r["label"], "pred": r["label"], "generator": r["generator"], "source": r["source"], "p_ai": 0.5 + r["label"] / 4}
        for r in val
    ]
    preds.append({"id": "nowhere", "split": "test", "label": 1, "pred": 1, "generator": "g/a", "source": "wikipedia", "p_ai": 0.9})
    preds.append({**preds[0], "split": "train"})
    preds.append({**preds[1], "label": 1 - preds[1]["label"]})  # same id, other label than the split file says
    records, info = evaluate.records_from_predictions(preds, splits)
    assert set(records) == {"val", "test"} and len(records["val"]) == 5 and len(records["test"]) == 1
    assert info == {"rows": 7, "unmatched": 1, "label_mismatch": 1, "other_splits": 1}
    joined = records["val"][2]
    assert joined["text"] == val[2]["text"] and joined["length"] == val[2]["length"] and joined["prompt_type"] == val[2]["prompt_type"]
    lonely = records["test"][0]
    assert lonely["text"] == "" and lonely["length"] is None
    records, info = evaluate.records_from_predictions(preds[:2], None)  # no split files at all
    assert info["unmatched"] == 2 and records["val"][0]["text"] == ""
    with pytest.raises(ValueError, match="missing"):
        evaluate.records_from_predictions([{"id": "x", "split": "val"}], None)


def test_add_token_info_flags_and_skips_rows_without_text():
    recs = {"val": [rec("a", 1, 1, 0.9, text="ក" * 10), rec("b", 0, 0, 0.1, text="ក" * 100), rec("c", 0, 0, 0.1, text="")]}
    stats = evaluate.add_token_info(recs, FakeTokenizer(), max_length=50)
    assert [r.get("truncated") for r in recs["val"]] == [False, True, None]
    assert stats["val"]["n"] == 2 and stats["val"]["truncated"] == 1 and stats["val"]["fraction_by_label"] == {"human": 1.0, "ai": 0.0}


def test_default_name():
    ns = lambda **kw: type("A", (), {"name": None, "model": None, "predictions": None, **kw})()  # noqa: E731
    assert evaluate.default_name(ns(model="baseline")) == "baseline"
    assert evaluate.default_name(ns(model="xlmr", name="xlmr_v2")) == "xlmr_v2"
    assert evaluate.default_name(ns(predictions=Path("reports/step6_xlmr_predictions.jsonl"))) == "xlmr"
    assert evaluate.default_name(ns(predictions=Path("mine.jsonl"))) == "mine"


# ---------------------------------------------------------------------------
# Robustness with a stub scorer
# ---------------------------------------------------------------------------
def test_char_truncation_cuts_both_classes_the_same_way_and_rescores():
    rows = {"test": make_rows("test", 4, 1), "test_ood": make_rows("ood", 3, 2), "val": make_rows("val", 2, 3)}
    records = {k: [evaluate.make_record(r, k, 0.9 if r["label"] else 0.1) for r in v] for k, v in rows.items()}
    scorer = StubScorer(lambda t: 0.9 if len(t) > 60 and set(t) & set(AI_LETTERS) else 0.1)  # an AI text is only recognised while it is longer than 60 chars
    block = evaluate.char_truncation(scorer, records, [50, 100, 10_000])
    assert block["chars"] == [50, 100, 10_000] and set(block["results"]) == {"full", "50", "100", "10000"}
    assert set(block["results"]["full"]) == {"test", "test_ood", "test+test_ood"}  # val is not part of the robustness test
    assert len(scorer.seen) == 6  # 3 cuts x 2 splits
    assert all(len(t) <= 50 for t in scorer.seen[0] + scorer.seen[1])
    assert all(len(t) <= 100 for t in scorer.seen[2] + scorer.seen[3])
    assert scorer.seen[4] == [r["text"] for r in rows["test"]]  # N beyond every text: nothing changes
    r50 = block["results"]["50"]["test"]
    assert r50["recall"] == 0.0 and r50["human_accuracy"] == 1.0 and r50["n"] == 8  # every AI text now looks short
    assert block["results"]["50"]["test+test_ood"]["n"] == 14
    assert block["results"]["10000"]["test"]["accuracy"] == 1.0
    s = block["cut_stats"]["50"]["test"]
    assert s["n"] == 8 and s["unchanged"] + s["sentence"] + s["hard"] == 8 and s["unchanged"] == 0
    assert s["mean_chars"]["human"] <= 50 and s["mean_chars"]["ai"] <= 50
    assert block["cut_stats"]["10000"]["test"]["unchanged"] == 8
    assert block["results"]["full"]["test"]["accuracy"] == 1.0  # from the existing records, no extra scoring


def test_char_truncation_with_one_robustness_split_only():
    records = {"test": [evaluate.make_record(r, "test", 0.5) for r in make_rows("t", 2, 1)]}
    block = evaluate.char_truncation(StubScorer(lambda t: 0.9), records, [80])
    assert set(block["results"]["80"]) == {"test"}  # no combined row for a single split


def test_paraphrase_robustness_compares_with_the_original_prediction():
    rows = make_rows("test", 4, 5)  # human: test000-003, AI: ai_test000-003
    records = {"test": [evaluate.make_record(r, "test", 0.9 if r["label"] else 0.1) for r in rows]}
    records["test"][5]["p_ai"], records["test"][5]["pred"] = 0.3, 0  # ai_test001 was already missed
    para = [
        {"id": "ai_test000", "label": 1, "text": "PARA-STAYS-AI", "model": "m", "ngram_overlap": 0.4},
        {"id": "ai_test001", "label": 1, "text": "PARA-BECOMES-AI", "model": "m", "ngram_overlap": 0.6},
        {"id": "ai_test002", "label": 1, "text": "PARA-ESCAPES", "model": "m"},
        {"id": "test000", "label": 0, "text": "PARA-BECOMES-AI", "model": "m", "ngram_overlap": 0.2},
        {"id": "test001", "label": 0, "text": "PARA-HUMAN", "model": "m"},
        {"id": "not-in-the-splits", "label": 1, "text": "x", "model": "m"},
    ]
    scorer = StubScorer(lambda t: {"PARA-STAYS-AI": 0.95, "PARA-BECOMES-AI": 0.8, "PARA-ESCAPES": 0.1, "PARA-HUMAN": 0.2}[t] if t != "x" else 0.5)
    block = evaluate.paraphrase_robustness(scorer, para, records)
    assert (block["rows_in_file"], block["rows_scored"], block["rows_unknown_id"], block["models"]) == (6, 5, 1, ["m"])
    ai = block["ai"]
    assert ai["n"] == 3 and ai["recall_original"] == pytest.approx(2 / 3, abs=1e-4) and ai["recall_paraphrased"] == pytest.approx(2 / 3, abs=1e-4)
    assert ai["changed_to_human"] == 1 and ai["changed_to_ai"] == 1  # ai_test002 escaped, ai_test001 was won back
    assert ai["mean_ngram_overlap"] == 0.5 and ai["mean_p_ai_paraphrased"] == pytest.approx((0.95 + 0.8 + 0.1) / 3, abs=1e-4)
    assert set(ai["by_generator"]) == {"g/a", "g/b"}
    human = block["human"]
    assert human["n"] == 2 and human["false_positive_rate_original"] == 0.0 and human["false_positive_rate_paraphrased"] == 0.5
    assert human["changed_to_ai"] == 1 and set(human["by_source"]) == {"mc4_web", "wikipedia"} and "note" in human
    no_human = evaluate.paraphrase_robustness(scorer, para, records, include_human=False)
    assert no_human["human"] is None and no_human["ai"]["n"] == 3


def test_load_paraphrases_keeps_the_first_record_per_id_and_skips_empty_text(tmp_path):
    path = tmp_path / "p.jsonl"
    clean.write_jsonl(path, [])  # empty file is fine
    assert evaluate.load_paraphrases(path) == []
    path.write_text(
        '{"id": "a", "text": "ក", "label": 1}\n{"id": "a", "text": "ខ", "label": 1}\n{"id": "b", "text": "", "label": 0}\n{"id": "c", "text": "គ"}\n',
        encoding="utf-8",
    )
    assert [(p["id"], p["text"]) for p in evaluate.load_paraphrases(path)] == [("a", "ក"), ("c", "គ")]


# ---------------------------------------------------------------------------
# main: predictions file
# ---------------------------------------------------------------------------
def write_predictions(path, splits_dir):
    """p_ai 0.9 for AI and 0.1 for human, except: one human of val called AI (0.95), one AI of val missed (0.2)."""
    preds = []
    for name in ("val", "test", "test_ood"):
        for r in clean.read_jsonl(splits_dir / f"{name}.jsonl"):
            p = 0.9 if r["label"] else 0.1
            if name == "val" and r["id"] == "val000":
                p = 0.95
            if name == "val" and r["id"] == "ai_val001":
                p = 0.2
            preds.append({"id": r["id"], "split": name, "label": r["label"], "pred": int(p > 0.5), "generator": r["generator"], "source": r["source"], "p_ai": p})
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return preds


def test_main_predictions_mode(tmp_path, capsys):
    splits_dir = tmp_path / "splits"
    write_splits(splits_dir)
    preds_path = tmp_path / "step6_demo_predictions.jsonl"
    write_predictions(preds_path, splits_dir)
    out = tmp_path / "rep" / "r.json"
    code = evaluate.main(["--predictions", str(preds_path), "--splits-dir", str(splits_dir), "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_bytes().decode("utf-8"))
    assert report["name"] == "demo" and report["mode"] == "predictions" and report["rows"] == {"val": 24, "test": 24, "test_ood": 16}
    assert report["rows_by_label"]["val"] == {"human": 12, "ai": 12}
    assert report["config"]["predictions_join"]["unmatched"] == 0
    val = report["metrics"]["val"]
    assert val["confusion_matrix"] == {"tn": 11, "fp": 1, "fn": 1, "tp": 11} and val["accuracy"] == pytest.approx(22 / 24, abs=1e-4)
    assert report["metrics"]["test"]["accuracy"] == 1.0 and report["metrics"]["test_ood"]["f1"] == 1.0
    assert set(report["breakdowns"]["test"]) == {"by_generator", "by_source", "by_prompt_type", "by_length_bin"}
    assert set(report["breakdowns"]["test"]["by_generator"]) == {"g/a", "g/b", "human"}
    assert report["token_truncation"] is None
    errs = report["errors"]["val"]
    assert [e["id"] for e in errs["listed"]] == ["val000", "ai_val001"]  # 0.95 sure before 0.80 sure
    assert errs["listed"][0]["text_head"].startswith(clean.read_jsonl(splits_dir / "val.jsonl")[0]["text"][:5])
    assert errs["false_positives"] == 1 and errs["false_negatives"] == 1
    assert report["errors"]["test"]["n_errors"] == 0
    assert report["robustness"] == {"skipped": "predictions-file mode: re-scoring truncated or paraphrased texts needs a model"}
    raw = out.read_bytes().decode("utf-8")
    assert "\\u1780" not in raw and "ក" in raw  # Khmer stored as UTF-8, not escapes
    printed = capsys.readouterr().out
    assert "Metrics (ai = positive class)" in printed and "robustness tests skipped" in printed and "val: 2 errors" in printed


def test_main_predictions_mode_with_a_tokenizer_and_without_split_files(tmp_path, monkeypatch, capsys):
    splits_dir = tmp_path / "splits"
    write_splits(splits_dir)
    preds_path = tmp_path / "p.jsonl"
    write_predictions(preds_path, splits_dir)
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda name: FakeTokenizer())
    out = tmp_path / "r.json"
    assert evaluate.main(["--predictions", str(preds_path), "--splits-dir", str(splits_dir), "--tokenizer", "fake", "--max-length", "100", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["token_truncation"]["val"]["fraction"] == 1.0  # ~190 chars = ~190 tokens > 100
    assert report["config"]["max_length"] == 100 and report["config"]["tokenizer"] == "fake"
    assert "over 100 tokens" in report["breakdowns"]["test"]["by_token_truncation"]
    assert "Texts longer than 100 tokens" in capsys.readouterr().out

    def broken(name):
        raise OSError("offline")

    monkeypatch.setattr(evaluate, "load_tokenizer", broken)
    out2 = tmp_path / "r2.json"
    code = evaluate.main(["--predictions", str(preds_path), "--splits-dir", str(tmp_path / "none"), "--tokenizer", "x", "--out", str(out2)])
    assert code == 0  # split files and tokenizer are optional in this mode
    err = capsys.readouterr().err
    assert "run src/split.py first" in err and "not available" in err
    report2 = json.loads(out2.read_text(encoding="utf-8"))
    assert report2["token_truncation"] is None and report2["config"]["predictions_join"]["unmatched"] == 64
    assert report2["metrics"]["val"]["n"] == 24 and report2["errors"]["val"]["listed"][0]["text_head"] == ""


def test_main_predictions_mode_errors(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "split": "val"}\n', encoding="utf-8")
    assert evaluate.main(["--predictions", str(bad), "--splits-dir", str(tmp_path), "--out", str(tmp_path / "r.json")]) == 2
    assert "missing" in capsys.readouterr().err
    only_train = tmp_path / "t.jsonl"
    only_train.write_text('{"id": "a", "split": "train", "label": 1, "pred": 1, "generator": "g", "source": "s", "p_ai": 0.9}\n', encoding="utf-8")
    assert evaluate.main(["--predictions", str(only_train), "--splits-dir", str(tmp_path), "--out", str(tmp_path / "r.json")]) == 2
    assert "no val/test/test_ood rows" in capsys.readouterr().err
    assert evaluate.main(["--predictions", str(tmp_path / "missing.jsonl"), "--splits-dir", str(tmp_path), "--out", str(tmp_path / "r.json")]) == 2


def test_arguments_need_exactly_one_source():
    with pytest.raises(SystemExit):
        evaluate.parse_args([])
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--model", "baseline", "--predictions", "x.jsonl"])
    args = evaluate.parse_args(["--model", "xlmr"])
    assert (args.max_length, args.truncate_chars, args.seed, args.batch_size) == (512, [300, 600, 1000], 42, 32)


# ---------------------------------------------------------------------------
# main: baseline model
# ---------------------------------------------------------------------------
@pytest.fixture
def baseline_setup(tmp_path):
    splits_dir, model_path = tmp_path / "splits", tmp_path / "m" / "model.joblib"
    write_splits(splits_dir)
    assert baseline.main(["--splits-dir", str(splits_dir), "--model-out", str(model_path), "--metrics-out", str(tmp_path / "bm.json"),
                          "--ngram-range", "1", "2", "--c-grid", "1", "--top", "3"]) == 0
    return splits_dir, model_path


def write_paraphrases(path, splits_dir):
    """AI paraphrases that stay AI-looking, one that looks human, a human paraphrase that looks AI."""
    test_ids = {r["id"] for r in clean.read_jsonl(splits_dir / "test.jsonl")}
    ai_text = make_rows("x", 1, 99)[1]["text"]
    human_text = make_rows("x", 1, 98)[0]["text"]
    recs = [
        {"id": "ai_test000", "split": "test", "label": 1, "text": ai_text, "model": "stub", "ngram_overlap": 0.3},
        {"id": "ai_test001", "split": "test", "label": 1, "text": ai_text, "model": "stub", "ngram_overlap": 0.4},
        {"id": "ai_test002", "split": "test", "label": 1, "text": human_text, "model": "stub", "ngram_overlap": 0.5},
        {"id": "test000", "split": "test", "label": 0, "text": human_text, "model": "stub", "ngram_overlap": 0.3},
        {"id": "test001", "split": "test", "label": 0, "text": ai_text, "model": "stub", "ngram_overlap": 0.3},
    ]
    assert all(r["id"] in test_ids for r in recs)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_main_baseline_end_to_end(baseline_setup, tmp_path, capsys):
    splits_dir, model_path = baseline_setup
    para = tmp_path / "eval" / "paraphrased.jsonl"
    para.parent.mkdir()
    write_paraphrases(para, splits_dir)
    out = tmp_path / "rep" / "step7.json"
    code = evaluate.main([
        "--model", "baseline", "--model-path", str(model_path), "--splits-dir", str(splits_dir), "--out", str(out),
        "--truncate-chars", "60", "120", "--paraphrased", str(para),
    ])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["name"] == "baseline" and report["mode"] == "model" and report["rows"] == {"val": 24, "test": 24, "test_ood": 16}
    for m in report["metrics"].values():
        assert {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "n", "human_accuracy"} <= set(m)
        assert m["accuracy"] > 0.9
    assert report["token_truncation"] is None and report["config"]["device"] == "cpu"
    trunc = report["robustness"]["char_truncation"]
    assert set(trunc["results"]) == {"full", "60", "120"} and set(trunc["results"]["60"]) == {"test", "test_ood", "test+test_ood"}
    assert trunc["results"]["120"]["test"]["n"] == 24 and trunc["cut_stats"]["60"]["test_ood"]["n"] == 16
    assert all(v["mean_chars"]["human"] <= 60 for v in trunc["cut_stats"]["60"].values())
    para_block = report["robustness"]["paraphrase"]
    assert para_block["rows_scored"] == 5 and para_block["ai"]["n"] == 3 and para_block["human"]["n"] == 2
    assert para_block["ai"]["recall_original"] == 1.0 and para_block["ai"]["recall_paraphrased"] == pytest.approx(2 / 3, abs=1e-4)
    assert para_block["ai"]["changed_to_human"] == 1  # the AI text rewritten with human-looking letters
    assert para_block["human"]["false_positive_rate_paraphrased"] == 0.5 and para_block["human"]["changed_to_ai"] == 1
    assert para_block["file"] == str(para)
    printed = capsys.readouterr().out
    assert "Texts cut to the first N chars" in printed and "Paraphrased texts" in printed and "evasion test" in printed

    out2 = tmp_path / "rep" / "no_human.json"
    assert evaluate.main(["--model", "baseline", "--model-path", str(model_path), "--splits-dir", str(splits_dir), "--out", str(out2),
                          "--truncate-chars", "--paraphrased", str(para), "--no-human-paraphrase"]) == 0
    r2 = json.loads(out2.read_text(encoding="utf-8"))
    assert "char_truncation" not in r2["robustness"] and r2["robustness"]["paraphrase"]["human"] is None
    out3 = tmp_path / "rep" / "none.json"
    assert evaluate.main(["--model", "baseline", "--model-path", str(model_path), "--splits-dir", str(splits_dir), "--out", str(out3),
                          "--truncate-chars", "80", "--paraphrased", str(tmp_path / "missing.jsonl")]) == 0
    assert "paraphrase" not in json.loads(out3.read_text(encoding="utf-8"))["robustness"]
    assert "no paraphrase test" in capsys.readouterr().out


def test_main_baseline_default_output_name_and_limit(baseline_setup, tmp_path, monkeypatch):
    splits_dir, model_path = baseline_setup
    monkeypatch.setattr(evaluate, "REPORT_TEMPLATE", tmp_path / "reports" / "step7_eval_{name}.json")
    assert evaluate.main(["--model", "baseline", "--model-path", str(model_path), "--splits-dir", str(splits_dir), "--limit", "6",
                          "--no-paraphrase", "--truncate-chars", "100"]) == 0
    report = json.loads((tmp_path / "reports" / "step7_eval_baseline.json").read_text(encoding="utf-8"))
    assert report["rows"] == {"val": 6, "test": 6, "test_ood": 6} and report["rows_by_label"]["val"] == {"human": 3, "ai": 3}


def test_main_fails_cleanly_without_model_or_splits(baseline_setup, tmp_path, capsys):
    splits_dir, model_path = baseline_setup
    assert evaluate.main(["--model", "baseline", "--model-path", str(tmp_path / "no.joblib"), "--splits-dir", str(splits_dir)]) == 2
    assert "run src/baseline.py first" in capsys.readouterr().err
    assert evaluate.main(["--model", "baseline", "--model-path", str(model_path), "--splits-dir", str(tmp_path / "nothing")]) == 2
    assert "run src/split.py first" in capsys.readouterr().err
    assert evaluate.main(["--model", "xlmr", "--model-path", str(tmp_path / "no_such_model"), "--splits-dir", str(splits_dir)]) == 2
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main: XLM-R backend with a tiny offline model
# ---------------------------------------------------------------------------
def make_tiny_model(folder, chars, max_length=64):
    """A 1-layer random XLM-R plus a character-level tokenizer, saved in folder: usable as --model-path offline."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification

    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for c in sorted(set(chars) | {" "}):
        vocab.setdefault(c, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")  # one token per character
    tok.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", model_max_length=max_length,
    ).save_pretrained(folder)
    config = XLMRobertaConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=max_length + 2, pad_token_id=1, bos_token_id=0, eos_token_id=2, id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1}, num_labels=2,
    )
    XLMRobertaForSequenceClassification(config).save_pretrained(folder)


@pytest.fixture
def tiny_model(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    folder = tmp_path / "tiny"
    make_tiny_model(folder, HUMAN_LETTERS + AI_LETTERS + "។")
    return folder


def test_xlmr_scorer_is_independent_of_batching_and_order(tiny_model):
    texts = [r["text"] for r in make_rows("a", 3, 4)] + ["កខ", "គ" * 300]  # the last one is longer than 64 tokens
    one = evaluate.XlmrScorer(tiny_model, max_length=64, batch_size=1).score(texts)
    many = evaluate.XlmrScorer(tiny_model, max_length=64, batch_size=4).score(texts)
    assert one.shape == (len(texts),) and np.all((one >= 0) & (one <= 1))
    assert np.allclose(one, many, atol=1e-5)  # sorting by length inside must not scramble the order
    rev = evaluate.XlmrScorer(tiny_model, max_length=64, batch_size=4).score(texts[::-1])
    assert np.allclose(one, rev[::-1], atol=1e-5)
    assert evaluate.XlmrScorer(tiny_model, max_length=64).score([]).shape == (0,)
    scorer = evaluate.XlmrScorer(tiny_model, max_length=64)
    assert scorer.score(["គ" * 300])[0] == pytest.approx(scorer.score(["គ" * 62])[0], abs=1e-5)  # both reach 64 tokens: the rest is cut off


def test_main_xlmr_end_to_end_with_a_tiny_model(tiny_model, tmp_path, capsys):
    splits_dir = tmp_path / "splits"
    write_splits(splits_dir, {"train": 4, "val": 6, "test": 6, "test_ood": 4})
    para = tmp_path / "paraphrased.jsonl"
    write_paraphrases(para, splits_dir)
    out = tmp_path / "rep" / "xlmr.json"
    code = evaluate.main([
        "--model", "xlmr", "--model-path", str(tiny_model), "--splits-dir", str(splits_dir), "--out", str(out), "--max-length", "64",
        "--batch-size", "4", "--truncate-chars", "40", "100", "--paraphrased", str(para),
    ])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["name"] == "xlmr" and report["rows"] == {"val": 12, "test": 12, "test_ood": 8} and report["config"]["max_length"] == 64
    assert report["config"]["device"].startswith(("cpu", "cuda"))
    assert report["token_truncation"]["val"]["fraction"] == 1.0  # ~190 chars is far over 64 tokens
    assert set(report["token_truncation"]["test"]["fraction_by_label"]) == {"human", "ai"}
    assert set(report["breakdowns"]["val"]["by_token_truncation"]) == {"over 64 tokens"}
    assert report["breakdowns"]["test_ood"]["by_generator"]["g/a"]["n_ai"] + report["breakdowns"]["test_ood"]["by_generator"]["g/b"]["n_ai"] == 4
    trunc = report["robustness"]["char_truncation"]
    assert set(trunc["results"]) == {"full", "40", "100"} and trunc["results"]["40"]["test"]["n"] == 12
    assert report["robustness"]["paraphrase"]["rows_scored"] == 5
    assert all(0 <= e["p_ai"] <= 1 for split in report["errors"].values() for e in split["listed"])
    printed = capsys.readouterr().out
    assert "Texts longer than 64 tokens" in printed and "Texts cut to the first N chars" in printed


def test_main_xlmr_rejects_a_max_length_the_model_cannot_take(tiny_model, tmp_path, capsys):
    splits_dir = tmp_path / "splits"
    write_splits(splits_dir, {"train": 2, "val": 2, "test": 2, "test_ood": 2})
    code = evaluate.main(["--model", "xlmr", "--model-path", str(tiny_model), "--splits-dir", str(splits_dir), "--max-length", "128", "--out", str(tmp_path / "r.json")])
    assert code == 2 and "more than the model supports" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The Colab bundle must carry what evaluate.py needs
# ---------------------------------------------------------------------------
def fake_repo(tmp_path, with_paraphrases):
    for name in make_colab_bundle.SPLIT_NAMES:
        clean.write_jsonl(tmp_path / "data" / "processed" / "splits" / f"{name}.jsonl", make_rows(name, 2, 1))
    (tmp_path / "src").mkdir()
    for name in ("evaluate.py", "finetune.py", "baseline.py", "clean.py", "paraphrase.py"):
        (tmp_path / "src" / name).write_text("# stub\n", encoding="utf-8")
    if with_paraphrases:
        (tmp_path / "data" / "eval").mkdir(parents=True)
        (tmp_path / "data" / "eval" / "paraphrased.jsonl").write_text('{"id": "a", "text": "ក"}\n{"id": "b", "text": "ខ"}\n', encoding="utf-8")


@pytest.mark.parametrize("with_paraphrases", [True, False])
def test_colab_bundle_has_the_evaluation_inputs(tmp_path, monkeypatch, capsys, with_paraphrases):
    fake_repo(tmp_path, with_paraphrases)
    monkeypatch.setattr(make_colab_bundle, "ROOT", tmp_path)
    monkeypatch.setattr(make_colab_bundle, "SPLITS_DIR", tmp_path / "data" / "processed" / "splits")
    monkeypatch.setattr(make_colab_bundle, "EVAL_DIR", tmp_path / "data" / "eval")
    monkeypatch.setattr(make_colab_bundle, "OUT_PATH", tmp_path / "bundle.zip")
    assert make_colab_bundle.main() == 0
    with zipfile.ZipFile(tmp_path / "bundle.zip") as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("MANIFEST.json").decode("utf-8"))
    assert {"src/evaluate.py", "src/finetune.py", "src/baseline.py", "src/clean.py"} <= names
    assert {f"data/processed/splits/{n}.jsonl" for n in make_colab_bundle.SPLIT_NAMES} <= names
    assert ("data/eval/paraphrased.jsonl" in names) == with_paraphrases
    assert manifest["rows"].get("paraphrased") == (2 if with_paraphrases else None)
    assert manifest["eval_files"] == (["paraphrased.jsonl"] if with_paraphrases else [])
    assert ("no paraphrased.jsonl" in capsys.readouterr().out) == (not with_paraphrases)
