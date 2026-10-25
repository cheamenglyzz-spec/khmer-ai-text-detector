"""Tests for src/baseline.py on tiny synthetic data (run: python -m pytest tests)."""
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baseline, clean  # noqa: E402

HUMAN_LETTERS = "កខគឃងចឆជ"
AI_LETTERS = "តថទធនបផព"


def make_rows(split, n_per_class, seed):
    """Synthetic Khmer-looking text: human rows use one set of letters, AI rows another, so the classes separate."""
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


@pytest.fixture
def splits_dir(tmp_path):
    folder = tmp_path / "splits"
    for i, (name, n) in enumerate({"train": 40, "val": 12, "test": 12, "test_ood": 8}.items()):
        clean.write_jsonl(folder / f"{name}.jsonl", make_rows(name, n, seed=i))
    return folder


def test_smoke_end_to_end(splits_dir, tmp_path, capsys):
    model_path, metrics_path = tmp_path / "m" / "model.joblib", tmp_path / "out" / "metrics.json"
    code = baseline.main([
        "--splits-dir", str(splits_dir), "--model-out", str(model_path), "--metrics-out", str(metrics_path),
        "--ngram-range", "1", "2", "--c-grid", "1", "10", "--top", "5",
    ])
    assert code == 0
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(report["metrics"]) == {"val", "test", "test_ood"}
    for m in report["metrics"].values():
        assert {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "n"} <= set(m)
        assert m["accuracy"] > 0.9  # the synthetic classes are trivially separable
    assert report["config"]["best_c"] in (1, 10)
    assert set(report["breakdowns"]["test"]["ai_recall_by_generator"]) == {"g/a"}
    assert set(report["breakdowns"]["test"]["human_accuracy_by_source"]) == {"wikipedia", "mc4_web"}
    top = report["top_ngrams"]
    assert len(top["ai"]) == len(top["human"]) == 5
    assert all(set(t["ngram"]) & set(AI_LETTERS) for t in top["ai"])  # repr() of an n-gram of AI-only letters
    assert all(set(t["ngram"]) & set(HUMAN_LETTERS) for t in top["human"])
    pipe = joblib.load(model_path)  # vectorizer + model in one object
    texts = [r["text"] for r in make_rows("x", 5, seed=99)]
    assert list(pipe.predict(texts)) == [0] * 5 + [1] * 5
    out = capsys.readouterr().out
    assert "Metrics (ai = positive class)" in out and "Top 5 n-grams" in out


def test_main_fails_cleanly_without_splits(tmp_path, capsys):
    assert baseline.main(["--splits-dir", str(tmp_path / "nothing")]) == 2
    assert "run src/split.py first" in capsys.readouterr().err


def test_compute_metrics_values():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    pred = np.array([1, 1, 0, 0, 0, 0, 1, 0])
    score = np.array([2.0, 1.0, -0.5, -1, -2, -3, 0.5, -1])
    m = baseline.compute_metrics(y, pred, score)
    assert m["confusion_matrix"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4) and m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["accuracy"] == 0.75 and m["n"] == 8
    assert 0.5 < m["roc_auc"] < 1.0
    one_class = baseline.compute_metrics(np.array([0, 0]), np.array([0, 1]), np.array([-1.0, 1.0]))
    assert one_class["roc_auc"] is None  # undefined with a single class


def test_breakdowns_by_generator_and_source():
    rows = [
        {"label": 1, "generator": "g/a", "source": "wikipedia"}, {"label": 1, "generator": "g/a", "source": "wikipedia"},
        {"label": 1, "generator": "g/b", "source": "mc4_web"},
        {"label": 0, "generator": "human", "source": "wikipedia"}, {"label": 0, "generator": "human", "source": "mc4_web"},
    ]
    b = baseline.breakdowns(rows, np.array([1, 0, 1, 0, 1]))
    assert b["ai_recall_by_generator"] == {"g/a": {"n": 2, "recall": 0.5}, "g/b": {"n": 1, "recall": 1.0}}
    assert b["human_accuracy_by_source"] == {"mc4_web": {"n": 1, "accuracy": 0.0}, "wikipedia": {"n": 1, "accuracy": 1.0}}


def test_fit_best_model_prefers_the_smaller_c_on_a_tie(splits_dir):
    rows = clean.read_jsonl(splits_dir / "train.jsonl")
    vec = baseline.build_vectorizer((1, 2), 2, 1000)
    X = vec.fit_transform(baseline.texts_of(rows))
    y = baseline.labels_of(rows)
    model, best_c, grid = baseline.fit_best_model(X, y, X, y, [10, 1, 0.1])
    assert [g["C"] for g in grid] == [0.1, 1, 10]
    assert best_c == min(g["C"] for g in grid if g["val_f1"] == max(x["val_f1"] for x in grid))


def test_odd_codepoints_flag_invisible_and_foreign_characters():
    assert baseline.odd_codepoints("កខ គ.1-") == []  # Khmer, space, ASCII
    assert baseline.odd_codepoints("ក" + chr(0x200B) + "ខ") == ["U+200B"]
    assert baseline.odd_codepoints("ក\nខ") == ["U+000A"]
    assert baseline.odd_codepoints("ក" + chr(0x4E2D) + chr(0x00A0)) == ["U+00A0", "U+4E2D"]


def test_top_ngrams_lists_class_markers_with_document_counts():
    texts_ai = ["កក" + chr(0x200B) + "ខខ", "ករ" + chr(0x200B) + "ខខ", "កក" + chr(0x200B) + "ងង"] * 3
    texts_human = ["កកខខ", "កងខខ", "កកងង"] * 3
    vec = baseline.build_vectorizer((1, 1), 2, 100)
    X = vec.fit_transform(texts_ai + texts_human)
    y = np.array([1] * 9 + [0] * 9)
    model, _, _ = baseline.fit_best_model(X, y, X, y, [10])
    top = baseline.top_ngrams(vec, model, X, y, k=3)
    first = top["ai"][0]
    assert first["ngram"] == repr(chr(0x200B)) and first["odd_chars"] == ["U+200B"]
    assert first["docs_ai"] == 9 and first["docs_human"] == 0 and first["share_ai"] == 1.0
    assert first["coef"] > 0 > top["human"][0]["coef"]
