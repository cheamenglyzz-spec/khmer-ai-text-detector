"""Step 5: baseline detector, TF-IDF character n-grams + logistic regression.

Khmer has few word spaces, so the features are character n-grams (1-4 by default) instead of words.
The vectorizer is fitted on train only. C is picked from a small grid by F1 on val (ai = positive
class); each grid model is trained on train, so the best one is the refit and is kept as is.

Metrics are reported on val, test and test_ood (the held-out generator). On test and test_ood the
report also breaks the result down per AI generator (recall) and per human source (accuracy).

Shortcut check: the top n-grams for each class by coefficient, shown with repr() so invisible or odd
characters are visible, plus how many train documents of each class contain them. An n-gram that is
frequent in one class and made of formatting characters, digits or Latin letters is an artifact the
classifier learned instead of style.

Input:  data/processed/splits/{train,val,test,test_ood}.jsonl   (step 4)
Output: models/baseline_tfidf_lr.joblib          sklearn Pipeline (vectorizer + model), gitignored
        reports/step5_baseline_metrics.json

Usage:
    python src/baseline.py
    python src/baseline.py --ngram-range 2 5 --c-grid 0.3 1 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `python src/baseline.py` can import its sibling module as src.clean
from src.clean import LABEL_NAMES, group_by, read_jsonl  # noqa: E402

SPLITS_DIR = ROOT / "data" / "processed" / "splits"
MODEL_PATH = ROOT / "models" / "baseline_tfidf_lr.joblib"
METRICS_PATH = ROOT / "reports" / "step5_baseline_metrics.json"

SEED = 42
NGRAM_RANGE = (1, 4)
MIN_DF = 2
MAX_FEATURES = 300_000
C_GRID = (0.1, 1.0, 10.0)
TOP_NGRAMS = 30
EVAL_SPLITS = ("val", "test", "test_ood")
SPLIT_FILES = ("train", "val", "test", "test_ood")


# ---------------------------------------------------------------------------
# Data, model
# ---------------------------------------------------------------------------
def load_splits(splits_dir: Path) -> dict[str, list[dict]]:
    missing = [f"{name}.jsonl" for name in SPLIT_FILES if not (splits_dir / f"{name}.jsonl").exists()]
    if missing:
        raise FileNotFoundError(f"missing in {splits_dir}: {', '.join(missing)} (run src/split.py first)")
    return {name: read_jsonl(splits_dir / f"{name}.jsonl") for name in SPLIT_FILES}


def build_vectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF, max_features=MAX_FEATURES) -> TfidfVectorizer:
    # lowercase=False: keep Latin case visible in the shortcut check; it makes no difference for Khmer
    return TfidfVectorizer(
        analyzer="char", ngram_range=tuple(ngram_range), sublinear_tf=True, min_df=min_df,
        max_features=max_features, lowercase=False, dtype=np.float32,
    )


def labels_of(rows: list[dict]) -> np.ndarray:
    return np.array([r["label"] for r in rows], dtype=np.int64)


def texts_of(rows: list[dict]) -> list[str]:
    return [r["text"] for r in rows]


def fit_best_model(X_train, y_train, X_val, y_val, c_grid, seed: int = SEED):
    """Train one model per C on train; return (best model, best C, per-C val results). Ties go to the smaller C."""
    results = []
    best = None
    for c in sorted(c_grid):
        model = LogisticRegression(C=c, solver="liblinear", max_iter=1000, random_state=seed)
        model.fit(X_train, y_train)
        f1 = float(f1_score(y_val, model.predict(X_val), zero_division=0))
        results.append({"C": c, "val_f1": round(f1, 4)})
        if best is None or f1 > best[0]:
            best = (f1, c, model)
    return best[2], best[1], results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, pred, score) -> dict:
    """Accuracy, precision, recall, F1 (ai = 1 is positive), ROC-AUC and the confusion matrix."""
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, pred, labels=[0, 1]).ravel())
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, score)), 4) if len(set(y_true.tolist())) == 2 else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def breakdowns(rows: list[dict], pred: np.ndarray) -> dict:
    """Recall per AI generator and accuracy on human rows per source."""
    ai, human = {}, {}
    for r, p in zip(rows, pred):
        side, key = (ai, r["generator"]) if r["label"] == 1 else (human, r["source"])
        side.setdefault(key, []).append(int(p == r["label"]))
    return {
        "ai_recall_by_generator": {k: {"n": len(v), "recall": round(sum(v) / len(v), 4)} for k, v in sorted(ai.items())},
        "human_accuracy_by_source": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in sorted(human.items())},
    }


# ---------------------------------------------------------------------------
# Shortcut check
# ---------------------------------------------------------------------------
def odd_codepoints(ngram: str) -> list[str]:
    """Characters that are neither Khmer, printable ASCII nor a plain space (invisible, symbols, other scripts)."""
    odd = {c for c in ngram if not (0x1780 <= ord(c) <= 0x17FF or 0x19E0 <= ord(c) <= 0x19FF or 0x20 <= ord(c) < 0x7F)}
    return [f"U+{ord(c):04X}" for c in sorted(odd)]


def top_ngrams(vectorizer: TfidfVectorizer, model: LogisticRegression, X_train, y_train, k: int = TOP_NGRAMS) -> dict:
    """The k n-grams with the largest positive (-> ai) and most negative (-> human) coefficients."""
    names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    order = np.argsort(coef)
    X_ai, X_human = X_train[np.flatnonzero(y_train == 1)], X_train[np.flatnonzero(y_train == 0)]
    out = {}
    for label, idx in (("ai", order[::-1][:k]), ("human", order[:k])):
        df_ai = X_ai[:, idx].getnnz(axis=0)
        df_human = X_human[:, idx].getnnz(axis=0)
        out[label] = [
            {
                "rank": rank + 1, "ngram": repr(str(names[i])), "coef": round(float(coef[i]), 4),
                "docs_ai": int(a), "docs_human": int(h),
                "share_ai": round(float(a) / max(X_ai.shape[0], 1), 4),
                "share_human": round(float(h) / max(X_human.shape[0], 1), 4),
                "odd_chars": odd_codepoints(str(names[i])),
            }
            for rank, (i, a, h) in enumerate(zip(idx, df_ai, df_human))
        ]
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_metrics(metrics: dict) -> None:
    print("\n=== Metrics (ai = positive class) ===")
    print(f"{'split':<10}{'n':>6}{'acc':>8}{'prec':>8}{'recall':>8}{'f1':>8}{'auc':>8}   confusion (tn fp fn tp)")
    for name, m in metrics.items():
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
        cm = m["confusion_matrix"]
        print(
            f"{name:<10}{m['n']:>6}{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}"
            f"{auc:>8}   {cm['tn']} {cm['fp']} {cm['fn']} {cm['tp']}"
        )


def print_breakdowns(name: str, b: dict) -> None:
    print(f"\n--- {name}: recall per AI generator, accuracy on human rows per source ---")
    for gen, v in b["ai_recall_by_generator"].items():
        print(f"  ai    {gen:<28}n {v['n']:>5}   recall {v['recall']:.4f}")
    for src, v in b["human_accuracy_by_source"].items():
        print(f"  human {src:<28}n {v['n']:>5}   accuracy {v['accuracy']:.4f}")


def print_top_ngrams(top: dict, n_ai: int, n_human: int) -> None:
    for label, items in top.items():
        print(f"\n=== Top {len(items)} n-grams towards '{label}' (train docs containing it: ai {n_ai}, human {n_human}) ===")
        print(f"{'#':>3}  {'coef':>8}  {'ai docs':>13}  {'human docs':>13}  n-gram (repr)   [odd chars]")
        for t in items:
            odd = f"   [{' '.join(t['odd_chars'])}]" if t["odd_chars"] else ""
            print(
                f"{t['rank']:>3}  {t['coef']:>+8.3f}  {t['docs_ai']:>5} {t['share_ai']:>6.1%}  "
                f"{t['docs_human']:>5} {t['share_human']:>6.1%}  {t['ngram']}{odd}"
            )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF-IDF character n-gram + logistic regression baseline.")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR, help="folder with the split files (default: %(default)s)")
    p.add_argument("--model-out", type=Path, default=MODEL_PATH, help="joblib file (default: %(default)s)")
    p.add_argument("--metrics-out", type=Path, default=METRICS_PATH, help="metrics JSON (default: %(default)s)")
    p.add_argument("--ngram-range", type=int, nargs=2, default=list(NGRAM_RANGE), metavar=("MIN", "MAX"))
    p.add_argument("--min-df", type=int, default=MIN_DF)
    p.add_argument("--max-features", type=int, default=MAX_FEATURES)
    p.add_argument("--c-grid", type=float, nargs="+", default=list(C_GRID), help="values of C to try (default: %(default)s)")
    p.add_argument("--top", type=int, default=TOP_NGRAMS, help="n-grams to list per class (default: %(default)s)")
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
    print("Rows: " + ", ".join(f"{name} {len(rows)}" for name, rows in splits.items()))
    y = {name: labels_of(rows) for name, rows in splits.items()}
    if len(set(y["train"].tolist())) < 2:
        print("error: train needs both human and ai rows", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    vectorizer = build_vectorizer(args.ngram_range, args.min_df, args.max_features)
    X = {"train": vectorizer.fit_transform(texts_of(splits["train"]))}
    for name in EVAL_SPLITS:
        X[name] = vectorizer.transform(texts_of(splits[name]))
    print(f"Vectorized in {time.monotonic() - t0:.0f}s: {X['train'].shape[1]} features "
          f"(char {args.ngram_range[0]}-{args.ngram_range[1]}-grams, min_df {args.min_df}, max {args.max_features})")

    t0 = time.monotonic()
    model, best_c, grid = fit_best_model(X["train"], y["train"], X["val"], y["val"], args.c_grid, args.seed)
    print(f"Fitted {len(grid)} models in {time.monotonic() - t0:.0f}s; val F1 by C: "
          + ", ".join(f"C={g['C']:g} {g['val_f1']:.4f}" for g in grid) + f"  -> best C = {best_c:g}")

    metrics, parts = {}, {}
    for name in EVAL_SPLITS:
        pred, score = model.predict(X[name]), model.decision_function(X[name])
        metrics[name] = compute_metrics(y[name], pred, score)
        parts[name] = breakdowns(splits[name], pred)
    print_metrics(metrics)
    for name in ("test", "test_ood"):
        print_breakdowns(name, parts[name])

    top = top_ngrams(vectorizer, model, X["train"], y["train"], args.top)
    print_top_ngrams(top, int((y["train"] == 1).sum()), int((y["train"] == 0).sum()))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(Pipeline([("tfidf", vectorizer), ("lr", model)]), args.model_out)
    report = {
        "config": {
            "seed": args.seed, "ngram_range": list(args.ngram_range), "min_df": args.min_df,
            "max_features": args.max_features, "c_grid": list(args.c_grid), "best_c": best_c,
            "n_features": int(X["train"].shape[1]), "splits_dir": str(args.splits_dir),
        },
        "rows": {name: len(rows) for name, rows in splits.items()},
        "rows_by_label": {name: {LABEL_NAMES[k]: len(v) for k, v in group_by(rows, lambda r: r["label"]).items()}
                          for name, rows in splits.items()},
        "c_search_val_f1": grid,
        "metrics": metrics,
        "breakdowns": parts,
        "top_ngrams": top,
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved model to {args.model_out}\nSaved metrics to {args.metrics_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
