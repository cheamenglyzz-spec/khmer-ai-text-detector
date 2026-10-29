"""Step 7: evaluate a detector on val / test / test_ood and probe how robust it is.

Three ways to get predictions:
    --model baseline   the step-5 joblib pipeline (runs anywhere, no GPU)
    --model xlmr       a saved step-6 model folder (Hugging Face format; scores on a GPU if there is one, else CPU)
    --predictions F    no model: analyse an existing predictions file (e.g. reports/step6_xlmr_predictions.jsonl,
                       made on Colab). Texts, lengths and prompt types are joined from the split files by id.
                       Robustness tests need a model, so they are skipped in this mode.

Per split the report has: overall metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ai = positive
class, plus accuracy on human rows), breakdowns by generator, source, prompt type and length bin (and by "cut off at
max_length tokens" when a tokenizer is available), and the misclassified rows, highest-confidence errors first.

Robustness (test + test_ood, needs a model):
  * char truncation: every text is cut to its first N chars (back to the last "។"/"៕" when that keeps at least half of
    N), both classes the same way, and re-scored. Reported next to the uncut result.
  * paraphrase: data/eval/paraphrased.jsonl (src/paraphrase.py) holds test rows an LLM rewrote. AI rows: how many are
    still caught (evasion test). Human rows, optional: how many become "AI" (false-positive check; --no-human-paraphrase
    leaves it out).

Input:  data/processed/splits/{val,test,test_ood}.jsonl   (step 4)
        models/baseline_tfidf_lr.joblib | models/xlmr/ | a predictions JSONL
        data/eval/paraphrased.jsonl                        (optional, src/paraphrase.py)
Output: reports/step7_eval_<name>.json

Usage:
    python src/evaluate.py --model baseline
    python src/evaluate.py --model xlmr --model-path models/xlmr                   # on Colab
    python src/evaluate.py --predictions reports/step6_xlmr_predictions.jsonl --tokenizer xlm-roberta-base
    python src/evaluate.py --model baseline --truncate-chars 200 500 --no-paraphrase --out some/where.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `python src/evaluate.py` can import its sibling modules as src.baseline / src.clean
from src.baseline import EVAL_SPLITS, compute_metrics, print_metrics  # noqa: E402
from src.clean import LABEL_NAMES, read_jsonl  # noqa: E402
from src.finetune import MAX_LENGTH, check_max_length, limit_rows, softmax_p_ai, token_lengths, truncation_stats  # noqa: E402

SPLITS_DIR = ROOT / "data" / "processed" / "splits"
BASELINE_PATH = ROOT / "models" / "baseline_tfidf_lr.joblib"
XLMR_DIR = ROOT / "models" / "xlmr"
PARAPHRASED_PATH = ROOT / "data" / "eval" / "paraphrased.jsonl"
REPORT_TEMPLATE = ROOT / "reports" / "step7_eval_{name}.json"

SEED = 42
THRESHOLD = 0.5  # predict "ai" when p_ai is above it (the same as argmax of the two logits)
ROBUSTNESS_SPLITS = ("test", "test_ood")
COMBINED = "+".join(ROBUSTNESS_SPLITS)
TRUNCATE_CHARS = (300, 600, 1000)
CUT_MIN_KEEP = 0.5  # a cut back to a sentence end must keep at least this share of N, else the text is cut hard at N
KHMER_ENDS = re.compile("[។៕]+[\"'”’»)\\]]*")  # Khmer khan / bariyoosan plus closing quotes or brackets
LENGTH_BIN_EDGES = (300, 700, 1100, 1500, 1900, 2501)  # 300-699, 700-1099, ..., 1900-2500
HEAD_CHARS = 150  # how much of an error's text is kept in the report
MAX_ERRORS = 100  # misclassified rows listed per split in the JSON (the counts are always complete)
PRINT_ERRORS = 8  # errors shown per split in the console summary
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------
def length_bin_label(length: int | None) -> str:
    """'300-699', '700-1099', ..., '1900-2500'; '<300' and '>2500' outside the dataset range."""
    if length is None:
        return "unknown"
    if length < LENGTH_BIN_EDGES[0]:
        return f"<{LENGTH_BIN_EDGES[0]}"
    if length >= LENGTH_BIN_EDGES[-1]:
        return f">{LENGTH_BIN_EDGES[-1] - 1}"
    i = bisect.bisect_right(LENGTH_BIN_EDGES, length) - 1
    return f"{LENGTH_BIN_EDGES[i]}-{LENGTH_BIN_EDGES[i + 1] - 1}"


LENGTH_BIN_ORDER = (
    [f"<{LENGTH_BIN_EDGES[0]}"]
    + [f"{lo}-{hi - 1}" for lo, hi in zip(LENGTH_BIN_EDGES, LENGTH_BIN_EDGES[1:])]
    + [f">{LENGTH_BIN_EDGES[-1] - 1}", "unknown"]
)


def cut_text(text: str, n: int, min_keep: float = CUT_MIN_KEEP) -> tuple[str, str]:
    """The first n chars of `text`, cut back to the last Khmer sentence end ("។" or "៕") when that keeps >= min_keep * n.

    Returns (cut text, kind): "unchanged" (text has at most n chars), "sentence" (ends at a Khmer sentence end) or
    "hard" (no sentence end in the window, or only an early one: cut at n and the trailing space removed).
    """
    if len(text) <= n:
        return text, "unchanged"
    window = text[:n]
    last = None
    for last in KHMER_ENDS.finditer(window):
        pass
    if last is not None and last.end() >= min_keep * n:
        return window[: last.end()], "sentence"
    return window.rstrip(), "hard"


def head_of(text: str, chars: int = HEAD_CHARS) -> str:
    return " ".join(text.split())[:chars]


def fmt(v, digits: int = 4) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def rate(hits: list[bool]) -> float | None:
    return round(sum(hits) / len(hits), 4) if hits else None


def mean_of(values: list[float], digits: int = 4) -> float | None:
    return round(float(np.mean(values)), digits) if values else None


# ---------------------------------------------------------------------------
# Scorers (one text list in, p_ai out)
# ---------------------------------------------------------------------------
class BaselineScorer:
    """The step-5 TF-IDF + logistic-regression pipeline."""

    label = "baseline"
    tokenizer = None
    max_length = None
    device = "cpu"

    def __init__(self, path: Path):
        import joblib

        self.pipeline = joblib.load(path)
        classes = list(self.pipeline.classes_)
        if classes != [0, 1]:
            raise ValueError(f"{path}: expected classes [0, 1] (human, ai), got {classes}")

    def score(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        return self.pipeline.predict_proba(texts)[:, 1]


class XlmrScorer:
    """A saved step-6 model folder; texts are cut to max_length tokens exactly as in training."""

    label = "xlmr"

    def __init__(self, model_dir: Path, max_length: int = MAX_LENGTH, batch_size: int = BATCH_SIZE):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        check_max_length(max_length, self.tokenizer)
        self.max_length, self.batch_size = max_length, batch_size
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        if id2label != LABEL_NAMES:
            print(f"warning: the model's labels are {id2label}, expected {LABEL_NAMES}; label 1 is read as 'ai'", file=sys.stderr)
        self.use_cuda = torch.cuda.is_available()
        self.device = f"cuda ({torch.cuda.get_device_name(0)})" if self.use_cuda else "cpu"
        self.model.to("cuda" if self.use_cuda else "cpu").eval()

    def score(self, texts: list[str]) -> np.ndarray:
        out = np.zeros(len(texts))
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch: little padding
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_cuda):
                    logits = self.model(**enc).logits
                out[idx] = softmax_p_ai(logits.float().cpu().numpy())
        return out


# ---------------------------------------------------------------------------
# Records: one dict per scored text
# ---------------------------------------------------------------------------
def load_eval_splits(splits_dir: Path) -> dict[str, list[dict]]:
    missing = [f"{name}.jsonl" for name in EVAL_SPLITS if not (splits_dir / f"{name}.jsonl").exists()]
    if missing:
        raise FileNotFoundError(f"missing in {splits_dir}: {', '.join(missing)} (run src/split.py first)")
    return {name: read_jsonl(splits_dir / f"{name}.jsonl") for name in EVAL_SPLITS}


def make_record(row: dict, split: str, p_ai: float) -> dict:
    return {
        "id": row["id"], "split": split, "label": int(row["label"]), "p_ai": float(p_ai), "pred": int(p_ai > THRESHOLD),
        "generator": row["generator"], "source": row["source"], "prompt_type": row.get("prompt_type", "") or "",
        "length": row.get("length") or len(row["text"]), "text": row["text"],
    }


def score_splits(scorer, splits: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {
        name: [make_record(r, name, p) for r, p in zip(rows, scorer.score([r["text"] for r in rows]))]
        for name, rows in splits.items()
    }


def records_from_predictions(preds: list[dict], splits: dict[str, list[dict]] | None) -> tuple[dict[str, list[dict]], dict]:
    """Records from a predictions file (id, split, label, pred, p_ai, generator, source); the split files add text and length."""
    need = {"id", "split", "label", "pred", "p_ai", "generator", "source"}
    for i, p in enumerate(preds):
        if need - set(p):
            raise ValueError(f"predictions row {i + 1} is missing {sorted(need - set(p))}")
    lookup = {(name, r["id"]): r for name, rows in (splits or {}).items() for r in rows}
    records: dict[str, list[dict]] = {name: [] for name in EVAL_SPLITS}
    info = {"rows": len(preds), "unmatched": 0, "label_mismatch": 0, "other_splits": 0}
    for p in preds:
        if p["split"] not in records:
            info["other_splits"] += 1
            continue
        row = lookup.get((p["split"], p["id"]))
        if row is None:
            info["unmatched"] += 1
        elif int(row["label"]) != int(p["label"]):
            info["label_mismatch"] += 1
        text = row["text"] if row else ""
        records[p["split"]].append({
            "id": p["id"], "split": p["split"], "label": int(p["label"]), "p_ai": float(p["p_ai"]), "pred": int(p["pred"]),
            "generator": p["generator"], "source": p["source"], "prompt_type": (row or {}).get("prompt_type", "") or "",
            "length": (row.get("length") or len(text)) if row else None, "text": text,
        })
    return {k: v for k, v in records.items() if v}, info


def add_token_info(records_by_split: dict[str, list[dict]], tokenizer, max_length: int) -> dict:
    """Set n_tokens / truncated on every record that has a text; return truncation stats per split."""
    stats = {}
    for name, recs in records_by_split.items():
        have = [r for r in recs if r["text"]]
        lengths = token_lengths([r["text"] for r in have], tokenizer)
        for r, n in zip(have, lengths):
            r["n_tokens"], r["truncated"] = n, n > max_length
        if have:
            stats[name] = truncation_stats(lengths, max_length, [r["label"] for r in have])
    return stats


# ---------------------------------------------------------------------------
# Metrics and breakdowns
# ---------------------------------------------------------------------------
def metrics_of(records: list[dict]) -> dict:
    """Step-5 metrics (compute_metrics) plus human_accuracy, the share of human texts called human."""
    y = np.array([r["label"] for r in records], dtype=np.int64)
    pred = np.array([r["pred"] for r in records], dtype=np.int64)
    score = np.array([r["p_ai"] for r in records], dtype=np.float64)
    m = compute_metrics(y, pred, score)
    cm = m["confusion_matrix"]
    m["human_accuracy"] = round(cm["tn"] / (cm["tn"] + cm["fp"]), 4) if cm["tn"] + cm["fp"] else None
    return m


def group_metrics(records: list[dict], key_fn, order: list[str] | None = None) -> dict:
    """Per group: n, accuracy, recall on its AI rows, accuracy on its human rows, mean p_ai, ROC-AUC (if both classes)."""
    from sklearn.metrics import roc_auc_score

    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(str(key_fn(r)), []).append(r)
    rank = {k: i for i, k in enumerate(order or [])}
    out = {}
    for key in sorted(groups, key=lambda k: (rank.get(k, len(rank)), k)):
        g = groups[key]
        ai, human = [r for r in g if r["label"] == 1], [r for r in g if r["label"] == 0]
        out[key] = {
            "n": len(g), "n_ai": len(ai), "n_human": len(human),
            "accuracy": rate([r["pred"] == r["label"] for r in g]),
            "ai_recall": rate([r["pred"] == 1 for r in ai]),
            "human_accuracy": rate([r["pred"] == 0 for r in human]),
            "mean_p_ai": mean_of([r["p_ai"] for r in g]),
            "roc_auc": round(float(roc_auc_score([r["label"] for r in g], [r["p_ai"] for r in g])), 4) if ai and human else None,
        }
    return out


def breakdowns_of(records: list[dict], max_length: int | None = None) -> dict:
    out = {
        "by_generator": group_metrics(records, lambda r: r["generator"]),
        "by_source": group_metrics(records, lambda r: r["source"]),
        "by_prompt_type": group_metrics(records, lambda r: r["prompt_type"] or "(human)"),
        "by_length_bin": group_metrics(records, lambda r: length_bin_label(r["length"]), LENGTH_BIN_ORDER),
    }
    if any("truncated" in r for r in records):
        out["by_token_truncation"] = group_metrics(
            records, lambda r: "unknown" if "truncated" not in r else (f"over {max_length} tokens" if r["truncated"] else f"within {max_length} tokens"),
        )
    return out


def error_rows(records: list[dict], limit: int = MAX_ERRORS) -> dict:
    """Misclassified rows, the ones the model was most sure about first."""
    def confidence(r):  # probability the model gave to its (wrong) answer
        return r["p_ai"] if r["pred"] == 1 else 1.0 - r["p_ai"]

    wrong = sorted((r for r in records if r["pred"] != r["label"]), key=lambda r: (-confidence(r), r["id"]))
    listed = [
        {
            "id": r["id"], "split": r["split"], "label": r["label"], "pred": r["pred"], "p_ai": round(r["p_ai"], 6),
            "confidence": round(confidence(r), 4), "generator": r["generator"], "source": r["source"],
            "prompt_type": r["prompt_type"], "length": r["length"], "text_head": head_of(r["text"]),
        }
        for r in wrong[:limit]
    ]
    return {
        "n_errors": len(wrong),
        "false_positives": sum(r["label"] == 0 for r in wrong),  # human called AI
        "false_negatives": sum(r["label"] == 1 for r in wrong),  # AI called human
        "listed": listed, "omitted": max(0, len(wrong) - limit),
    }


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def scored_copy(records: list[dict], p_ai: np.ndarray) -> list[dict]:
    return [{**r, "p_ai": float(p), "pred": int(p > THRESHOLD)} for r, p in zip(records, p_ai)]


def char_truncation(scorer, records_by_split: dict[str, list[dict]], chars, min_keep: float = CUT_MIN_KEEP) -> dict:
    """Re-score test and test_ood with every text cut to its first N chars, for each N."""
    pools = {s: records_by_split[s] for s in ROBUSTNESS_SPLITS if records_by_split.get(s)}

    def table(per_split: dict[str, list[dict]]) -> dict:
        out = {name: metrics_of(recs) for name, recs in per_split.items()}
        if len(per_split) > 1:
            out[COMBINED] = metrics_of([r for recs in per_split.values() for r in recs])
        return out

    results = {"full": table(pools)}
    cut_stats = {}
    for n in chars:
        scored, stats = {}, {}
        for split, recs in pools.items():
            cuts = [cut_text(r["text"], n, min_keep) for r in recs]
            scored[split] = scored_copy(recs, scorer.score([c for c, _ in cuts]))
            kinds = [k for _, k in cuts]
            stats[split] = {
                "n": len(recs), **{k: kinds.count(k) for k in ("unchanged", "sentence", "hard")},
                "mean_chars": {
                    LABEL_NAMES[label]: mean_of([len(c) for (c, _), r in zip(cuts, recs) if r["label"] == label], 1) for label in (0, 1)
                },
            }
        results[str(n)], cut_stats[str(n)] = table(scored), stats
    return {"chars": list(chars), "min_keep": min_keep, "results": results, "cut_stats": cut_stats}


def load_paraphrases(path: Path) -> list[dict]:
    """Paraphrase records (id of the original, text); the first record per id wins."""
    seen, out = set(), []
    for r in read_jsonl(path):
        if "id" in r and r.get("text") and r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def paired_summary(pairs: list[tuple[dict, dict]], rate_name: str) -> dict:
    """pairs = (original record, rescored paraphrase record). rate_name: 'recall' for AI rows, 'false_positive_rate' for human."""
    orig_ai = [o["pred"] == 1 for o, _ in pairs]
    para_ai = [p["pred"] == 1 for _, p in pairs]
    return {
        "n": len(pairs),
        f"{rate_name}_original": rate(orig_ai), f"{rate_name}_paraphrased": rate(para_ai),
        "changed_to_human": sum(o and not p for o, p in zip(orig_ai, para_ai)),
        "changed_to_ai": sum(p and not o for o, p in zip(orig_ai, para_ai)),
        "mean_p_ai_original": mean_of([o["p_ai"] for o, _ in pairs]), "mean_p_ai_paraphrased": mean_of([p["p_ai"] for _, p in pairs]),
        "mean_ngram_overlap": mean_of([p["ngram_overlap"] for _, p in pairs if p.get("ngram_overlap") is not None]),
    }


def paraphrase_robustness(scorer, paraphrases: list[dict], records_by_split: dict[str, list[dict]], include_human: bool = True) -> dict:
    """Score the LLM paraphrases and compare them with the model's answer on the original text of the same row."""
    originals = {r["id"]: r for recs in records_by_split.values() for r in recs}
    usable = [p for p in paraphrases if p["id"] in originals]
    p_ai = scorer.score([p["text"] for p in usable])
    pairs = []
    for p, s in zip(usable, p_ai):
        o = originals[p["id"]]
        pairs.append((o, {**o, "p_ai": float(s), "pred": int(s > THRESHOLD), "ngram_overlap": p.get("ngram_overlap"), "text": p["text"]}))
    block = {
        "models": sorted({p.get("model", "?") for p in usable}),
        "rows_in_file": len(paraphrases), "rows_scored": len(usable), "rows_unknown_id": len(paraphrases) - len(usable),
    }
    ai_pairs = [pr for pr in pairs if pr[0]["label"] == 1]
    block["ai"] = paired_summary(ai_pairs, "recall") if ai_pairs else None
    if block["ai"]:
        by_gen: dict[str, list] = {}
        for pr in ai_pairs:
            by_gen.setdefault(pr[0]["generator"], []).append(pr)
        block["ai"]["by_generator"] = {g: paired_summary(v, "recall") for g, v in sorted(by_gen.items())}
    human_pairs = [pr for pr in pairs if pr[0]["label"] == 0]
    if include_human and human_pairs:
        block["human"] = paired_summary(human_pairs, "false_positive_rate")
        by_src: dict[str, list] = {}
        for pr in human_pairs:
            by_src.setdefault(pr[0]["source"], []).append(pr)
        block["human"]["by_source"] = {s: paired_summary(v, "false_positive_rate") for s, v in sorted(by_src.items())}
        block["human"]["note"] = "false-positive check: LLM-rewritten human text is not AI-written text from scratch"
    else:
        block["human"] = None
    return block


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def print_groups(title: str, groups: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  {'group':<26}{'n':>6}{'ai':>6}{'human':>7}{'acc':>8}{'ai recall':>11}{'human acc':>11}{'mean p_ai':>11}{'auc':>8}")
    for key, g in groups.items():
        print(
            f"  {key:<26}{g['n']:>6}{g['n_ai']:>6}{g['n_human']:>7}{fmt(g['accuracy']):>8}{fmt(g['ai_recall']):>11}"
            f"{fmt(g['human_accuracy']):>11}{fmt(g['mean_p_ai']):>11}{fmt(g['roc_auc']):>8}"
        )


def print_errors(split: str, errors: dict, top: int = PRINT_ERRORS) -> None:
    print(f"\n--- {split}: {errors['n_errors']} errors ({errors['false_positives']} human called AI, "
          f"{errors['false_negatives']} AI called human); most confident first ---")
    for e in errors["listed"][:top]:
        kind = "FP human->AI" if e["label"] == 0 else "FN AI->human"
        print(f"  {kind}  conf {e['confidence']:.3f}  p_ai {e['p_ai']:.3f}  {e['generator']}/{e['source']}  {e['length']} chars  {e['id']}")
        print(f"      {e['text_head'][:100]}")


def print_token_truncation(stats: dict, max_length: int) -> None:
    print(f"\n=== Texts longer than {max_length} tokens (cut off by the model) ===")
    for name, t in stats.items():
        by_label = "  ".join(f"{k} {v:.1%}" for k, v in t.get("fraction_by_label", {}).items())
        print(f"  {name:<10}{t['n']:>6} texts   cut {t['truncated']:>5} ({fmt(t['fraction'], 3)})   {by_label}")


def print_char_truncation(block: dict) -> None:
    print(f"\n=== Texts cut to the first N chars (back to the last sentence end if >= {block['min_keep']:.0%} of N is kept) ===")
    print(f"  {'N':>5}  {'split':<14}{'n':>5}{'acc':>8}{'f1':>8}{'auc':>8}{'ai recall':>11}{'human acc':>11}   cut: unchanged/sentence/hard   mean chars human/ai")
    for key, per_split in block["results"].items():
        for split, m in per_split.items():
            extra = ""
            stats = block["cut_stats"].get(key, {}).get(split)
            if stats:
                mc = stats["mean_chars"]
                extra = f"   {stats['unchanged']:>4}/{stats['sentence']:>4}/{stats['hard']:>4}                {fmt(mc['human'], 0)}/{fmt(mc['ai'], 0)}"
            print(
                f"  {key:>5}  {split:<14}{m['n']:>5}{fmt(m['accuracy']):>8}{fmt(m['f1']):>8}{fmt(m['roc_auc']):>8}"
                f"{fmt(m['recall']):>11}{fmt(m['human_accuracy']):>11}{extra}"
            )


def print_paraphrase(block: dict) -> None:
    print(f"\n=== Paraphrased texts (by {', '.join(block['models']) or '?'}): {block['rows_scored']} scored of {block['rows_in_file']} in the file ===")
    for label, key, rate_name, what in (("AI", "ai", "recall", "share still called AI (evasion test)"),
                                         ("human", "human", "false_positive_rate", "share called AI (false-positive check)")):
        s = block.get(key)
        if not s:
            continue
        print(f"  {label} texts, n {s['n']}: {what}: {fmt(s[rate_name + '_original'])} original -> {fmt(s[rate_name + '_paraphrased'])} paraphrased; "
              f"{s['changed_to_human']} flipped to human, {s['changed_to_ai']} to AI; mean p_ai {fmt(s['mean_p_ai_original'])} -> {fmt(s['mean_p_ai_paraphrased'])}; "
              f"overlap with original {fmt(s['mean_ngram_overlap'], 2)}")
        for sub, d in s.get("by_generator", s.get("by_source", {})).items():
            print(f"      {sub:<28}n {d['n']:>4}   {fmt(d[rate_name + '_original'])} -> {fmt(d[rate_name + '_paraphrased'])}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def default_name(args: argparse.Namespace) -> str:
    if args.name:
        return args.name
    if args.model:
        return args.model
    return re.sub(r"^step\d+_|_predictions$", "", args.predictions.stem) or "predictions"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 7: evaluate a detector and test its robustness.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", choices=("baseline", "xlmr"), help="score with this model")
    src.add_argument("--predictions", type=Path, help="analyse an existing predictions JSONL instead (no model; no robustness tests)")
    p.add_argument("--model-path", type=Path, help=f"joblib file or model folder (default: {BASELINE_PATH.name} / models/xlmr)")
    p.add_argument("--tokenizer", help="with --predictions: tokenizer name or folder, to flag texts cut off at --max-length tokens (e.g. xlm-roberta-base)")
    p.add_argument("--name", help="report name (default: the model, or the predictions file name without step6_ / _predictions)")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR, help="folder with the split files (default: %(default)s)")
    p.add_argument("--out", type=Path, help="report JSON (default: reports/step7_eval_<name>.json)")
    p.add_argument("--max-length", type=int, default=MAX_LENGTH, help="xlmr: tokens per text (default: %(default)s)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="xlmr: texts per batch (default: %(default)s)")
    p.add_argument("--truncate-chars", type=int, nargs="*", default=list(TRUNCATE_CHARS), metavar="N",
                   help="cut texts to the first N chars and re-score (default: %(default)s; none = skip)")
    p.add_argument("--cut-min-keep", type=float, default=CUT_MIN_KEEP,
                   help="a cut back to a sentence end must keep this share of N (default: %(default)s)")
    p.add_argument("--paraphrased", type=Path, default=PARAPHRASED_PATH, help="paraphrases from src/paraphrase.py, used if the file exists (default: %(default)s)")
    p.add_argument("--no-paraphrase", action="store_true", help="skip the paraphrase test")
    p.add_argument("--no-human-paraphrase", action="store_true", help="paraphrase test: leave out the paraphrased human texts")
    p.add_argument("--max-errors", type=int, default=MAX_ERRORS, help="misclassified rows listed per split in the JSON (default: %(default)s)")
    p.add_argument("--limit", type=int, default=0, help="model modes: score only N rows per split, label-balanced and seeded (smoke tests)")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def build_scorer(args: argparse.Namespace):
    if args.model == "baseline":
        path = args.model_path or BASELINE_PATH
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (run src/baseline.py first)")
        return BaselineScorer(path)
    path = args.model_path or XLMR_DIR
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (train with src/finetune.py, or pass --model-path)")
    return XlmrScorer(path, args.max_length, args.batch_size)


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Khmer must not crash a cp1252 console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    name = default_name(args)
    out_path = args.out or Path(str(REPORT_TEMPLATE).format(name=name))

    try:
        splits = load_eval_splits(args.splits_dir)
    except FileNotFoundError as exc:
        if args.model:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"warning: {exc}; texts, lengths and prompt types will be missing", file=sys.stderr)
        splits = None

    scorer, tokenizer, info = None, None, None
    if args.model:
        if args.limit > 0:
            splits = {k: limit_rows(v, args.limit, args.seed) for k, v in splits.items()}
        try:
            scorer = build_scorer(args)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        tokenizer = scorer.tokenizer
        print(f"Model: {name} ({args.model_path or (BASELINE_PATH if args.model == 'baseline' else XLMR_DIR)}), device {scorer.device}")
        records = score_splits(scorer, splits)
    else:
        try:
            preds = read_jsonl(args.predictions)
            records, info = records_from_predictions(preds, splits)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not records:
            print(f"error: no val/test/test_ood rows in {args.predictions}", file=sys.stderr)
            return 2
        print(f"Predictions: {info['rows']} rows from {args.predictions}")
        if info["unmatched"] or info["label_mismatch"]:
            print(f"warning: {info['unmatched']} predictions have no row in the split files, {info['label_mismatch']} have a different label",
                  file=sys.stderr)
        if args.tokenizer:
            try:
                tokenizer = load_tokenizer(args.tokenizer)
            except Exception as exc:  # no network and not cached, bad name, ...
                print(f"warning: tokenizer {args.tokenizer!r} not available ({type(exc).__name__}); not flagging cut-off texts", file=sys.stderr)

    max_length = args.max_length
    token_stats = None
    if tokenizer is not None:
        try:
            check_max_length(max_length, tokenizer)
            token_stats = add_token_info(records, tokenizer, max_length)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    print("Rows: " + ", ".join(f"{k} {len(v)}" for k, v in records.items()))

    metrics = {k: metrics_of(v) for k, v in records.items()}
    parts = {k: breakdowns_of(v, max_length) for k, v in records.items()}
    errors = {k: error_rows(v, args.max_errors) for k, v in records.items()}
    print_metrics(metrics)
    if token_stats:
        print_token_truncation(token_stats, max_length)
    for split in ROBUSTNESS_SPLITS:
        for title, groups in parts.get(split, {}).items():
            print_groups(f"{split}: {title.replace('_', ' ')}", groups)
    for split, e in errors.items():
        print_errors(split, e)

    robustness: dict = {}
    if scorer is None:
        robustness["skipped"] = "predictions-file mode: re-scoring truncated or paraphrased texts needs a model"
        print("\n(robustness tests skipped: they need a model, not just predictions)")
    else:
        if args.truncate_chars:
            robustness["char_truncation"] = char_truncation(scorer, records, args.truncate_chars, args.cut_min_keep)
            print_char_truncation(robustness["char_truncation"])
        if not args.no_paraphrase and args.paraphrased.exists():
            paraphrases = load_paraphrases(args.paraphrased)
            if paraphrases:
                robustness["paraphrase"] = paraphrase_robustness(scorer, paraphrases, records, not args.no_human_paraphrase)
                robustness["paraphrase"]["file"] = str(args.paraphrased)
                print_paraphrase(robustness["paraphrase"])
        elif not args.no_paraphrase:
            print(f"\n(no paraphrase test: {args.paraphrased} not found; make it with src/paraphrase.py)")

    report = {
        "name": name,
        "mode": "model" if args.model else "predictions",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "model": args.model, "model_path": str(args.model_path) if args.model_path else None,
            "predictions": str(args.predictions) if args.predictions else None, "splits_dir": str(args.splits_dir),
            "device": scorer.device if scorer else None, "max_length": max_length if tokenizer is not None else None,
            "tokenizer": args.tokenizer, "limit": args.limit, "seed": args.seed,
            "truncate_chars": list(args.truncate_chars), "cut_min_keep": args.cut_min_keep, "threshold": THRESHOLD,
            "predictions_join": info,
        },
        "rows": {k: len(v) for k, v in records.items()},
        "rows_by_label": {k: {LABEL_NAMES[lab]: sum(r["label"] == lab for r in v) for lab in (0, 1)} for k, v in records.items()},
        "metrics": metrics,
        "breakdowns": parts,
        "token_truncation": token_stats,
        "errors": errors,
        "robustness": robustness,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved report to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
