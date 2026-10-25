"""Step 6: fine-tune xlm-roberta-base as the detector (Hugging Face Trainer, runs on CPU or one GPU).

Texts are tokenized with truncation at --max-length (512 is the model's limit). Khmer can need many
tokens per character, so the metrics JSON reports, for every split, the fraction of texts that got
truncated (and the token-length distribution), also split by label: a label gap there would be a
length shortcut. Batches are padded dynamically, fp16 is used only when CUDA is available.

Training runs for --epochs epochs with an evaluation and a checkpoint on val after each one; the
checkpoint with the best val F1 (ai = positive class) is loaded at the end. That model is then scored
on val, test and test_ood (the held-out generator) with exactly the metrics of step 5 (the functions
are imported from baseline.py), including recall per AI generator and accuracy per human source.

Input:  data/processed/splits/{train,val,test,test_ood}.jsonl   (step 4)
Output: models/xlmr/                              final model + tokenizer, gitignored
        reports/step6_xlmr_metrics.json           same structure as step 5, plus truncation and training info
        reports/step6_xlmr_predictions.jsonl      one row per val/test/test_ood text: id, split, label, pred,
                                                  generator, source, p_ai (probability of "ai"), for step 7

Usage:
    python src/finetune.py                                          # full run; needs a GPU to be practical
    python src/finetune.py --limit 32 --max-steps 4 --max-length 128    # CPU smoke test
    python src/finetune.py --lr 3e-5 --epochs 4 --batch-size 8
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `python src/finetune.py` can import its sibling modules as src.baseline / src.clean
from src.baseline import (  # noqa: E402
    EVAL_SPLITS, breakdowns, compute_metrics, labels_of, load_splits, print_breakdowns, print_metrics, texts_of,
)
from src.clean import LABEL_NAMES, group_by  # noqa: E402

SPLITS_DIR = ROOT / "data" / "processed" / "splits"
OUT_DIR = ROOT / "models" / "xlmr"
METRICS_PATH = ROOT / "reports" / "step6_xlmr_metrics.json"
PREDICTIONS_PATH = ROOT / "reports" / "step6_xlmr_predictions.jsonl"
CHECKPOINT_SUBDIR = "checkpoints"  # default checkpoint folder, inside --out-dir; emptied after the final model is saved

MODEL_NAME = "xlm-roberta-base"
SEED = 42
MAX_LENGTH = 512
LEARNING_RATE = 2e-5
EPOCHS = 3
BATCH_SIZE = 16
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
ID2LABEL = dict(LABEL_NAMES)  # {0: "human", 1: "ai"}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


# ---------------------------------------------------------------------------
# Data preparation (pure functions; the tokenizer is passed in)
# ---------------------------------------------------------------------------
def limit_rows(rows: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """A seeded sample of n rows with the labels as balanced as the data allows (for smoke tests).

    Each label's rows are shuffled, then the labels are dealt out in turn, so a small n gives
    n/2 human and n/2 ai rows. All rows are returned if n >= len(rows)."""
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    queues = []
    for _, group in sorted(group_by(rows, lambda r: r["label"]).items()):
        group = list(group)
        rng.shuffle(group)
        queues.append(group)
    out: list[dict] = []
    while len(out) < n and any(queues):
        for queue in queues:
            if queue and len(out) < n:
                out.append(queue.pop())
    return out


def check_max_length(max_length: int, tokenizer) -> None:
    if max_length < 8:
        raise ValueError(f"--max-length {max_length} is too small")
    if max_length > tokenizer.model_max_length:
        raise ValueError(f"--max-length {max_length} is more than the model supports ({tokenizer.model_max_length})")


def token_lengths(texts: list[str], tokenizer) -> list[int]:
    """Tokens per text including special tokens, with no truncation."""
    if not texts:
        return []
    enc = tokenizer(texts, truncation=False, padding=False, add_special_tokens=True, verbose=False)  # verbose=False: no "longer than max" warning
    return [len(ids) for ids in enc["input_ids"]]


def truncation_stats(lengths: list[int], max_length: int, labels=None) -> dict:
    """How many texts need more than max_length tokens, overall and per label if labels are given."""
    arr = np.asarray(lengths, dtype=np.int64)
    over = arr > max_length
    stats = {
        "n": int(arr.size),
        "truncated": int(over.sum()),
        "fraction": round(float(over.mean()), 4) if arr.size else None,
        "tokens_mean": round(float(arr.mean()), 1) if arr.size else None,
        "tokens_median": float(np.median(arr)) if arr.size else None,
        "tokens_max": int(arr.max()) if arr.size else None,
    }
    if labels is not None:
        lab = np.asarray(labels)
        stats["fraction_by_label"] = {
            LABEL_NAMES[k]: round(float(over[lab == k].mean()), 4) for k in sorted(LABEL_NAMES) if (lab == k).any()
        }
    return stats


def encode_rows(rows: list[dict], tokenizer, max_length: int) -> dict[str, list]:
    """input_ids / attention_mask (truncated to max_length, not padded: the collator pads per batch) and labels."""
    enc = tokenizer(texts_of(rows), truncation=True, max_length=max_length, padding=False)
    return {
        "input_ids": [list(ids) for ids in enc["input_ids"]],
        "attention_mask": [list(m) for m in enc["attention_mask"]],
        "labels": labels_of(rows).tolist(),
    }


def prepare_data(splits: dict[str, list[dict]], tokenizer, max_length: int):
    """Return ({split: datasets.Dataset}, {split: truncation stats})."""
    from datasets import Dataset

    datasets, truncation = {}, {}
    for name, rows in splits.items():
        truncation[name] = truncation_stats(token_lengths(texts_of(rows), tokenizer), max_length, labels_of(rows))
        datasets[name] = Dataset.from_dict(encode_rows(rows, tokenizer, max_length))
    return datasets, truncation


# ---------------------------------------------------------------------------
# Metrics and predictions
# ---------------------------------------------------------------------------
def softmax_p_ai(logits) -> np.ndarray:
    """Probability of the 'ai' class (label 1) from logits of shape (n, 2)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def trainer_metrics(eval_pred) -> dict:
    """Flat metrics for the Trainer (it logs and compares them): F1 with ai as the positive class drives model selection."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, y = np.asarray(eval_pred.predictions), np.asarray(eval_pred.label_ids)
    pred = logits.argmax(axis=-1)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


def prediction_records(rows: list[dict], split: str, p_ai: np.ndarray, pred: np.ndarray) -> list[dict]:
    return [
        {
            "id": r["id"], "split": split, "label": int(r["label"]), "pred": int(p), "generator": r["generator"],
            "source": r["source"], "p_ai": round(float(s), 6),
        }
        for r, s, p in zip(rows, p_ai, pred)
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def remove_checkpoints(folder: Path) -> None:
    """Delete the Trainer's checkpoint-* folders (and the folder itself if that leaves it empty), nothing else."""
    for ckpt in folder.glob("checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)
    try:
        folder.rmdir()
    except OSError:
        pass


def epoch_log(log_history: list[dict]) -> list[dict]:
    """One entry per evaluation (= per epoch) from the Trainer's log."""
    return [
        {
            "epoch": round(float(e["epoch"]), 2), "step": int(e["step"]), "val_loss": round(float(e["eval_loss"]), 4),
            "val_f1": round(float(e["eval_f1"]), 4), "val_accuracy": round(float(e["eval_accuracy"]), 4),
        }
        for e in log_history if "eval_f1" in e
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_training_args(args: argparse.Namespace, output_dir: Path, use_cuda: bool):
    from transformers import TrainingArguments

    names = {f.name for f in dataclasses.fields(TrainingArguments)}
    # transformers 5 folds warmup_ratio into warmup_steps (a float below 1 is a ratio); older versions have both
    warmup = {"warmup_ratio": args.warmup_ratio} if "warmup_ratio" in names else {"warmup_steps": args.warmup_ratio}
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2 * args.batch_size,
        weight_decay=args.weight_decay,
        fp16=use_cuda,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,  # no optimizer state: keeps checkpoints at ~1 GB (we never resume)
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=LOGGING_STEPS,
        dataloader_pin_memory=use_cuda,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        **warmup,
    )


def checkpoint_dir(args: argparse.Namespace) -> Path:
    return args.checkpoint_dir or args.out_dir / CHECKPOINT_SUBDIR


def train_model(args: argparse.Namespace, tokenizer, datasets: dict, use_cuda: bool):
    """Fine-tune and return (trainer with the best-val-F1 model loaded, seconds spent training, train metrics)."""
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, set_seed

    set_seed(args.seed)  # before the model is built: the classification head is initialized randomly
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    trainer = Trainer(
        model=model,
        args=build_training_args(args, checkpoint_dir(args), use_cuda),
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_cuda else None),
        compute_metrics=trainer_metrics,
        processing_class=tokenizer,
    )
    t0 = time.monotonic()
    result = trainer.train()
    return trainer, time.monotonic() - t0, result.metrics


def evaluate_splits(trainer, datasets: dict, splits: dict[str, list[dict]]):
    """Score val, test and test_ood with the loaded model; return (metrics, breakdowns, prediction rows)."""
    metrics, parts, predictions = {}, {}, []
    for name in EVAL_SPLITS:
        logits = trainer.predict(datasets[name]).predictions
        pred, p_ai = np.asarray(logits).argmax(axis=-1), softmax_p_ai(logits)
        metrics[name] = compute_metrics(labels_of(splits[name]), pred, p_ai)
        parts[name] = breakdowns(splits[name], pred)
        predictions += prediction_records(splits[name], name, p_ai, pred)
    return metrics, parts, predictions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_truncation(truncation: dict, max_length: int) -> None:
    print(f"\n=== Texts longer than {max_length} tokens (cut off) ===")
    print(f"{'split':<10}{'n':>6}{'cut':>7}{'share':>8}{'median':>8}{'max':>7}   share by label")
    for name, t in truncation.items():
        by_label = "  ".join(f"{k} {v:.1%}" for k, v in t.get("fraction_by_label", {}).items())
        share = "n/a" if t["fraction"] is None else f"{t['fraction']:.1%}"
        print(f"{name:<10}{t['n']:>6}{t['truncated']:>7}{share:>8}{t['tokens_median'] or 0:>8.0f}{t['tokens_max'] or 0:>7}   {by_label}")


def print_epochs(log: list[dict]) -> None:
    print("\n=== Validation by epoch ===")
    for e in log:
        print(f"  epoch {e['epoch']:<5g} step {e['step']:>5}   val loss {e['val_loss']:.4f}   acc {e['val_accuracy']:.4f}   F1 {e['val_f1']:.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune xlm-roberta-base to detect AI-generated Khmer text.")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR, help="folder with the split files (default: %(default)s)")
    p.add_argument("--model", default=MODEL_NAME, help="Hugging Face model name or local folder (default: %(default)s)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where the final model and tokenizer go (default: %(default)s)")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="where the Trainer keeps checkpoints while training (default: <out-dir>/checkpoints); "
                        "on Colab a local folder avoids writing ~1 GB per epoch to Drive")
    p.add_argument("--metrics", "--metrics-out", dest="metrics_out", type=Path, default=METRICS_PATH,
                   help="metrics JSON (default: %(default)s)")
    p.add_argument("--predictions", "--predictions-out", dest="predictions_out", type=Path, default=PREDICTIONS_PATH,
                   help="per-row predictions JSONL (default: %(default)s)")
    p.add_argument("--max-length", type=int, default=MAX_LENGTH, help="truncate texts to this many tokens (default: %(default)s)")
    p.add_argument("--lr", type=float, default=LEARNING_RATE, help="peak learning rate (default: %(default)s)")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs (default: %(default)s)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="per-device train batch size (default: %(default)s)")
    p.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO, help="share of steps used for warmup (default: %(default)s)")
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="(default: %(default)s)")
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps, overriding --epochs (smoke tests)")
    p.add_argument("--limit", type=int, default=0, help="use only N rows per split, label-balanced and seeded (smoke tests)")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Khmer must not crash a cp1252 console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        splits = load_splits(args.splits_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.limit > 0:
        splits = {name: limit_rows(rows, args.limit, args.seed) for name, rows in splits.items()}
    print("Rows: " + ", ".join(f"{name} {len(rows)}" for name, rows in splits.items()))
    if len(set(labels_of(splits["train"]).tolist())) < 2:
        print("error: train needs both human and ai rows", file=sys.stderr)
        return 2

    import torch  # heavy imports live here so that --help and the pure functions above stay fast
    import transformers
    from transformers import AutoTokenizer

    use_cuda = torch.cuda.is_available()
    device = f"cuda ({torch.cuda.get_device_name(0)})" if use_cuda else "cpu"
    print(f"Device: {device}; fp16 {'on' if use_cuda else 'off'}; model {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        check_max_length(args.max_length, tokenizer)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    datasets, truncation = prepare_data(splits, tokenizer, args.max_length)
    print_truncation(truncation, args.max_length)

    trainer, train_seconds, train_metrics = train_model(args, tokenizer, datasets, use_cuda)
    log = epoch_log(trainer.state.log_history)
    print(f"\nTrained in {train_seconds:.0f}s ({trainer.state.global_step} steps, train loss {train_metrics.get('train_loss', float('nan')):.4f})")
    print_epochs(log)

    metrics, parts, predictions = evaluate_splits(trainer, datasets, splits)
    print_metrics(metrics)
    for name in ("test", "test_ood"):
        print_breakdowns(name, parts[name])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))
    remove_checkpoints(checkpoint_dir(args))  # the final model in out_dir is the best checkpoint
    write_jsonl(args.predictions_out, predictions)
    best_ckpt = trainer.state.best_model_checkpoint
    report = {
        "config": {
            "model": args.model, "seed": args.seed, "max_length": args.max_length, "learning_rate": args.lr,
            "epochs": args.epochs, "batch_size": args.batch_size, "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay, "max_steps": args.max_steps, "limit": args.limit,
            "splits_dir": str(args.splits_dir), "select_by": "val f1 (ai = positive)",
        },
        "environment": {
            "device": device, "fp16": use_cuda, "torch": torch.__version__, "transformers": transformers.__version__,
        },
        "rows": {name: len(rows) for name, rows in splits.items()},
        "rows_by_label": {name: {LABEL_NAMES[k]: len(v) for k, v in group_by(rows, lambda r: r["label"]).items()}
                          for name, rows in splits.items()},
        "truncation": truncation,
        "training": {
            "train_seconds": round(train_seconds, 1), "steps": int(trainer.state.global_step),
            "train_loss": round(float(train_metrics["train_loss"]), 4) if "train_loss" in train_metrics else None,
            "best_val_f1": round(float(trainer.state.best_metric), 4) if trainer.state.best_metric is not None else None,
            "best_checkpoint": Path(best_ckpt).name if best_ckpt else None,
            "val_by_epoch": log,
        },
        "metrics": metrics,
        "breakdowns": parts,
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved model to {args.out_dir}\nSaved metrics to {args.metrics_out}\nSaved predictions to {args.predictions_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
