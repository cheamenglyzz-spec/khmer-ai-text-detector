"""Tests for src/finetune.py without downloading anything (run: python -m pytest tests).

The pure functions are tested with a fake tokenizer. One end-to-end test trains a tiny random
XLM-R model with a character-level tokenizer that is built in a temp folder, so the whole script
(Trainer, evaluation, saving, JSON outputs) runs offline on CPU in a few seconds."""
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import clean, finetune  # noqa: E402

HUMAN_LETTERS = "កខគឃងចឆជ"
AI_LETTERS = "តថទធនបផព"


class FakeTokenizer:
    """One token per character plus <s> and </s>; truncation keeps the final </s>, like the real ones."""

    model_max_length = 512

    def __call__(self, texts, truncation=False, max_length=None, padding=False, add_special_tokens=True, **kwargs):
        ids = [[0] + [ord(c) % 90 + 4 for c in t] + [2] for t in texts]
        if truncation:
            ids = [x if len(x) <= max_length else x[: max_length - 1] + [2] for x in ids]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


def make_rows(split, n_per_class, seed):
    """Synthetic Khmer-looking text: human rows use one set of letters, AI rows another."""
    rng = random.Random(seed)
    rows = []
    for label, letters, generator in ((0, HUMAN_LETTERS, "human"), (1, AI_LETTERS, "g/a")):
        for i in range(n_per_class):
            text = " ".join("".join(rng.choice(letters) for _ in range(rng.randint(3, 7))) for _ in range(30))
            hid = f"{split}{i:03d}"
            rows.append({
                "id": hid if label == 0 else f"ai_{hid}", "text": text, "label": label, "source": "wikipedia" if i % 2 else "mc4_web",
                "generator": generator, "url": "", "date": "", "length": len(text), "human_id": hid, "title": "", "prompt_type": "",
            })
    return rows


def write_splits(folder, sizes):
    for i, (name, n) in enumerate(sizes.items()):
        clean.write_jsonl(folder / f"{name}.jsonl", make_rows(name, n, seed=i))


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------
def test_label_maps_follow_the_dataset_convention():
    assert finetune.ID2LABEL == {0: "human", 1: "ai"}
    assert finetune.LABEL2ID == {"human": 0, "ai": 1}


def test_limit_rows_is_balanced_seeded_and_never_invents_rows():
    rows = make_rows("t", 20, seed=1)
    a, b = finetune.limit_rows(rows, 10, seed=42), finetune.limit_rows(rows, 10, seed=42)
    assert [r["id"] for r in a] == [r["id"] for r in b]  # deterministic
    assert [r["id"] for r in a] != [r["id"] for r in finetune.limit_rows(rows, 10, seed=7)]
    assert len(a) == 10 and sum(r["label"] for r in a) == 5
    assert len(finetune.limit_rows(rows, 7, seed=42)) == 7  # odd n: labels differ by at most one
    assert len(finetune.limit_rows(rows, 400)) == 40 and len(finetune.limit_rows(rows, 40)) == 40
    assert finetune.limit_rows(rows, 400) is not rows  # a copy
    skewed = [r for r in rows if r["label"] == 0] + [r for r in rows if r["label"] == 1][:2]
    out = finetune.limit_rows(skewed, 10)
    assert len(out) == 10 and sum(r["label"] for r in out) == 2  # the scarce label runs out, the other fills up
    assert len({r["id"] for r in out}) == 10


def test_token_lengths_and_truncation_stats():
    tok = FakeTokenizer()
    lengths = finetune.token_lengths(["ab", "abcdefgh", "abcdefghij"], tok)
    assert lengths == [4, 10, 12]  # characters + <s> + </s>
    assert finetune.token_lengths([], tok) == []
    stats = finetune.truncation_stats(lengths, max_length=10, labels=[0, 1, 1])
    assert stats["n"] == 3 and stats["truncated"] == 1 and stats["fraction"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["tokens_max"] == 12 and stats["tokens_median"] == 10.0
    assert stats["fraction_by_label"] == {"human": 0.0, "ai": 0.5}
    assert finetune.truncation_stats([10, 10], max_length=10)["truncated"] == 0  # exactly max_length still fits
    empty = finetune.truncation_stats([], max_length=10)
    assert empty["n"] == 0 and empty["fraction"] is None


def test_encode_rows_truncates_and_keeps_labels():
    rows = make_rows("t", 2, seed=3)
    enc = finetune.encode_rows(rows, FakeTokenizer(), max_length=16)
    assert set(enc) == {"input_ids", "attention_mask", "labels"}
    assert enc["labels"] == [0, 0, 1, 1]
    assert all(len(x) == 16 and x[0] == 0 and x[-1] == 2 for x in enc["input_ids"])  # <s> ... </s> survives truncation
    assert enc["attention_mask"] == [[1] * 16] * 4
    short = finetune.encode_rows([{"text": "កខ", "label": 1}], FakeTokenizer(), max_length=16)
    assert len(short["input_ids"][0]) == 4  # not padded: the collator pads per batch


def test_prepare_data_builds_datasets_and_truncation_per_split():
    splits = {"train": make_rows("a", 3, seed=1), "val": make_rows("b", 2, seed=2)}
    datasets, truncation = finetune.prepare_data(splits, FakeTokenizer(), max_length=64)
    assert {k: len(v) for k, v in datasets.items()} == {"train": 6, "val": 4}
    assert datasets["train"].column_names == ["input_ids", "attention_mask", "labels"]
    assert datasets["train"]["labels"] == [0, 0, 0, 1, 1, 1]
    assert truncation["train"]["fraction"] == 1.0 and truncation["train"]["fraction_by_label"] == {"human": 1.0, "ai": 1.0}
    assert finetune.prepare_data(splits, FakeTokenizer(), max_length=512)[1]["val"]["truncated"] == 0


def test_khmer_text_is_not_mangled_by_encoding():
    text = "ភាសាខ្មែរ"
    enc = finetune.encode_rows([{"text": text, "label": 0}], FakeTokenizer(), max_length=64)
    assert enc["input_ids"][0][1:-1] == [ord(c) % 90 + 4 for c in text]


def test_softmax_p_ai():
    p = finetune.softmax_p_ai([[0.0, 0.0], [0.0, 2.0], [3.0, 0.0], [1000.0, 1001.0]])
    assert p[0] == pytest.approx(0.5) and p[1] == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert p[2] < 0.05 and 0.7 < p[3] < 0.75  # no overflow for large logits
    assert np.all((p >= 0) & (p <= 1))


def test_trainer_metrics_is_flat_and_uses_ai_as_positive():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    m = finetune.trainer_metrics(SimpleNamespace(predictions=logits, label_ids=np.array([0, 1, 0, 1, 1])))
    assert set(m) == {"f1", "accuracy", "precision", "recall"} and all(isinstance(v, float) for v in m.values())
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5) and m["f1"] == pytest.approx(2 / 3)


def test_prediction_records_carry_what_step_7_needs():
    rows = make_rows("t", 1, seed=1)
    recs = finetune.prediction_records(rows, "test_ood", np.array([0.1234567891, 0.9]), np.array([0, 1]))
    assert recs == [
        {"id": "t000", "split": "test_ood", "label": 0, "pred": 0, "generator": "human", "source": "mc4_web", "p_ai": 0.123457},
        {"id": "ai_t000", "split": "test_ood", "label": 1, "pred": 1, "generator": "g/a", "source": "mc4_web", "p_ai": 0.9},
    ]


def test_epoch_log_keeps_only_evaluation_entries():
    history = [
        {"loss": 0.7, "epoch": 0.5, "step": 10},
        {"eval_loss": 0.61234, "eval_f1": 0.81234, "eval_accuracy": 0.8, "epoch": 1.0, "step": 20},
        {"eval_loss": 0.5, "eval_f1": 0.9, "eval_accuracy": 0.9, "epoch": 2.0, "step": 40},
        {"train_loss": 0.6, "train_runtime": 3.0, "epoch": 2.0, "step": 40},
    ]
    log = finetune.epoch_log(history)
    assert [e["epoch"] for e in log] == [1.0, 2.0] and log[0]["val_f1"] == 0.8123 and log[0]["val_loss"] == 0.6123


def test_check_max_length():
    finetune.check_max_length(512, FakeTokenizer())
    with pytest.raises(ValueError, match="more than the model supports"):
        finetune.check_max_length(513, FakeTokenizer())
    with pytest.raises(ValueError, match="too small"):
        finetune.check_max_length(2, FakeTokenizer())


def test_remove_checkpoints_only_touches_checkpoint_folders(tmp_path):
    folder = tmp_path / "ck"
    (folder / "checkpoint-8").mkdir(parents=True)
    (folder / "checkpoint-8" / "model.safetensors").write_bytes(b"x")
    (folder / "checkpoint-16").mkdir()
    finetune.remove_checkpoints(folder)
    assert not folder.exists()  # empty afterwards, so it is removed too
    (folder / "keep").mkdir(parents=True)
    (folder / "checkpoint-4").mkdir()
    finetune.remove_checkpoints(folder)
    assert (folder / "keep").exists() and not (folder / "checkpoint-4").exists()  # foreign content survives
    finetune.remove_checkpoints(tmp_path / "never_created")  # no error


def test_write_jsonl_round_trips_khmer(tmp_path):
    path = tmp_path / "sub" / "p.jsonl"
    finetune.write_jsonl(path, [{"id": "ក", "p_ai": 0.5}])
    assert path.read_text(encoding="utf-8") == '{"id": "ក", "p_ai": 0.5}\n'


def test_training_args_follow_the_brief():
    args = finetune.parse_args([])
    assert (args.model, args.max_length, args.lr, args.epochs, args.batch_size) == ("xlm-roberta-base", 512, 2e-5, 3, 16)
    assert (args.warmup_ratio, args.weight_decay, args.seed, args.max_steps, args.limit) == (0.1, 0.01, 42, -1, 0)
    ta = finetune.build_training_args(args, Path("unused"), use_cuda=False)
    assert ta.eval_strategy == "epoch" and ta.save_strategy == "epoch" and ta.save_total_limit == 1
    assert ta.load_best_model_at_end and ta.metric_for_best_model == "f1" and ta.greater_is_better
    assert ta.learning_rate == 2e-5 and ta.num_train_epochs == 3 and ta.per_device_train_batch_size == 16
    assert ta.weight_decay == 0.01 and ta.seed == 42 and not ta.fp16
    warmup = getattr(ta, "warmup_ratio", None) or ta.warmup_steps  # transformers 5 keeps the ratio in warmup_steps
    assert warmup == 0.1
    assert finetune.build_training_args(args, Path("unused"), use_cuda=True).fp16 is True


# ---------------------------------------------------------------------------
# End to end with a tiny offline model
# ---------------------------------------------------------------------------
def make_tiny_model(folder, chars, max_length=64):
    """A 1-layer random XLM-R plus a character-level tokenizer, saved in folder: usable as --model offline."""
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
        max_position_embeddings=max_length + 2, pad_token_id=1, bos_token_id=0, eos_token_id=2,
    )
    XLMRobertaForSequenceClassification(config).save_pretrained(folder)


def test_main_end_to_end_with_a_tiny_model(tmp_path, capsys):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    model_dir, splits_dir = tmp_path / "tiny", tmp_path / "splits"
    make_tiny_model(model_dir, HUMAN_LETTERS + AI_LETTERS)
    write_splits(splits_dir, {"train": 16, "val": 6, "test": 6, "test_ood": 4})
    out_dir, metrics_path, preds_path = tmp_path / "out" / "model", tmp_path / "rep" / "m.json", tmp_path / "rep" / "p.jsonl"
    code = finetune.main([
        "--splits-dir", str(splits_dir), "--model", str(model_dir), "--out-dir", str(out_dir),
        "--metrics", str(metrics_path), "--predictions", str(preds_path),
        "--max-length", "64", "--epochs", "2", "--batch-size", "8", "--lr", "1e-3",
    ])
    assert code == 0
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(report["metrics"]) == {"val", "test", "test_ood"}  # same structure as step 5
    for m in report["metrics"].values():
        assert {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "n"} <= set(m)
    assert report["metrics"]["val"]["n"] == 12 and report["metrics"]["test_ood"]["n"] == 8
    assert set(report["breakdowns"]["test"]["ai_recall_by_generator"]) == {"g/a"}
    assert set(report["breakdowns"]["test_ood"]["human_accuracy_by_source"]) == {"wikipedia", "mc4_web"}
    assert set(report["truncation"]) == {"train", "val", "test", "test_ood"}
    assert report["truncation"]["train"]["fraction"] == 1.0  # ~200-character texts, 64 tokens
    assert report["rows"] == {"train": 32, "val": 12, "test": 12, "test_ood": 8}
    assert [e["epoch"] for e in report["training"]["val_by_epoch"]] == [1.0, 2.0]
    assert report["training"]["steps"] == 8 and report["training"]["best_checkpoint"].startswith("checkpoint-")
    assert report["environment"]["device"] == "cpu" and report["config"]["learning_rate"] == 1e-3
    preds = [json.loads(line) for line in preds_path.read_text(encoding="utf-8").splitlines()]
    assert len(preds) == 12 + 12 + 8
    assert {p["split"] for p in preds} == {"val", "test", "test_ood"}
    assert all(set(p) == {"id", "split", "label", "pred", "generator", "source", "p_ai"} and 0 <= p["p_ai"] <= 1 for p in preds)
    assert {p["id"] for p in preds if p["split"] == "val"} == {r["id"] for r in clean.read_jsonl(splits_dir / "val.jsonl")}
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert config["id2label"] == {"0": "human", "1": "ai"} and config["label2id"] == {"human": 0, "ai": 1}
    assert (out_dir / "tokenizer.json").exists()
    assert not (out_dir / finetune.CHECKPOINT_SUBDIR).exists()  # only the final model is kept
    out = capsys.readouterr().out
    assert "Metrics (ai = positive class)" in out and "Texts longer than 64 tokens" in out and "Trained in" in out


def test_main_limit_flag_and_max_steps_smoke(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    model_dir, splits_dir = tmp_path / "tiny", tmp_path / "splits"
    make_tiny_model(model_dir, HUMAN_LETTERS + AI_LETTERS)
    write_splits(splits_dir, {"train": 20, "val": 10, "test": 10, "test_ood": 10})
    metrics_path = tmp_path / "m.json"
    code = finetune.main([
        "--splits-dir", str(splits_dir), "--model", str(model_dir), "--out-dir", str(tmp_path / "o"),
        "--metrics", str(metrics_path), "--predictions", str(tmp_path / "p.jsonl"),
        "--limit", "8", "--max-steps", "3", "--max-length", "32", "--batch-size", "4",
        "--checkpoint-dir", str(tmp_path / "ck"),
    ])
    assert code == 0
    assert not (tmp_path / "ck").exists() and not (tmp_path / "o" / finetune.CHECKPOINT_SUBDIR).exists()
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert report["rows"] == {"train": 8, "val": 8, "test": 8, "test_ood": 8}
    assert report["rows_by_label"]["train"] == {"human": 4, "ai": 4}
    assert report["training"]["steps"] == 3 and report["config"]["limit"] == 8 and report["config"]["max_steps"] == 3


def test_main_fails_cleanly_without_splits(tmp_path, capsys):
    assert finetune.main(["--splits-dir", str(tmp_path / "nothing")]) == 2
    assert "run src/split.py first" in capsys.readouterr().err


def test_main_rejects_a_max_length_the_model_cannot_take(tmp_path, capsys):
    pytest.importorskip("transformers")
    model_dir, splits_dir = tmp_path / "tiny", tmp_path / "splits"
    make_tiny_model(model_dir, HUMAN_LETTERS + AI_LETTERS)
    write_splits(splits_dir, {"train": 4, "val": 2, "test": 2, "test_ood": 2})
    code = finetune.main(["--splits-dir", str(splits_dir), "--model", str(model_dir), "--max-length", "128", "--out-dir", str(tmp_path / "o")])
    assert code == 2 and "more than the model supports" in capsys.readouterr().err
