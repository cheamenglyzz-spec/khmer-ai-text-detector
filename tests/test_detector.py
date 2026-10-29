"""Tests for app/detector.py and a UI smoke test of app/app.py (run: python -m pytest tests)."""
import importlib.util
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import detector  # noqa: E402
from src import baseline, clean  # noqa: E402

REAL_MODEL = ROOT / "models" / "baseline_tfidf_lr.joblib"
HUMAN_LETTERS = "កខគឃងចឆជ"
AI_LETTERS = "តថទធនបផព"
SENTENCE = "ភ្នំពេញជារដ្ឋធានីនៃព្រះរាជាណាចក្រកម្ពុជា ហើយជាទីក្រុងធំជាងគេបំផុតនៅក្នុងប្រទេស។"


def khmer_text(min_chars):
    """Whole Khmer sentences, at least `min_chars` characters (the sentence repeats, which cleaning does not mind)."""
    n = -(-min_chars // (len(SENTENCE) + 1))
    return " ".join([SENTENCE] * n)


def synthetic_doc(rng, letters, words=40):
    return " ".join("".join(rng.choice(letters) for _ in range(rng.randint(3, 7))) for _ in range(words)) + "។"


@pytest.fixture(scope="module")
def tiny_pipeline():
    """A fitted TF-IDF + LR pipeline on made-up Khmer-letter text (AI docs use other letters than human docs)."""
    rng = random.Random(0)
    texts = [synthetic_doc(rng, HUMAN_LETTERS) for _ in range(30)] + [synthetic_doc(rng, AI_LETTERS) for _ in range(30)]
    labels = [0] * 30 + [1] * 30
    vectorizer = baseline.build_vectorizer((1, 3), min_df=1)
    model = LogisticRegression(C=10, solver="liblinear", random_state=0)
    return Pipeline([("tfidf", vectorizer), ("lr", model)]).fit(texts, labels)


@pytest.fixture
def tiny_model_path(tiny_pipeline, tmp_path):
    path = tmp_path / "tiny.joblib"
    joblib.dump(tiny_pipeline, path)
    return path


@pytest.fixture
def tiny_backend(tiny_model_path):
    return detector.BaselineBackend(tiny_model_path)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def test_empty_input_is_empty():
    for raw in ("", "   ", "\n\n\t ", None):
        prepared = detector.prepare(raw)
        assert prepared.empty and prepared.n_chars == 0 and prepared.notices == ()


def test_normal_text_matches_the_training_cleaning_and_has_no_warnings():
    raw = khmer_text(600)
    prepared = detector.prepare(raw)
    assert prepared.text == clean.process_text(raw, is_ai=False).filtered.text
    assert prepared.notices == () and prepared.removed_chars == 0
    assert prepared.khmer_share > 0.7 and prepared.n_chars >= 600


def test_pasted_text_is_cleaned_like_training_text():
    body = khmer_text(400)
    raw = "ចំណងជើងខ្លី\n" + body.replace("។", " ។") + "​\nមើលផងដែរ\nទំព័រផ្សេង\n" + SENTENCE  # headline, ZWSP, spaces, footer
    prepared = detector.prepare(raw)
    assert prepared.text == body  # headline and footer gone, space before ។ and the ZWSP removed
    assert prepared.removed_chars > 0


def test_trailing_fragment_is_trimmed_to_the_last_sentence_end():
    body = khmer_text(400)
    prepared = detector.prepare(body + " ឃ្លាមិនទាន់ចប់")
    assert prepared.text == body and prepared.removed_chars > 0


def test_short_text_warns_but_is_scored():
    prepared = detector.prepare(SENTENCE)
    assert not prepared.empty and prepared.codes == ("short",)
    assert "less reliable" in prepared.notices[0].message
    assert detector.prepare(khmer_text(300)).codes == ()  # exactly the lower limit of the training range


def test_long_text_warns_but_is_scored():
    prepared = detector.prepare(khmer_text(clean.MAX_CHARS + 100))
    assert prepared.codes == ("long",) and prepared.n_chars > clean.MAX_CHARS


def test_low_khmer_share_warns_but_is_scored():
    prepared = detector.prepare("This is an English sentence! " * 20)
    assert "low_khmer" in prepared.codes and not prepared.empty
    assert prepared.khmer_share < clean.MIN_KHMER_RATIO


def test_text_without_sentence_end_is_scored_as_typed():
    prepared = detector.prepare("សួស្តី  អ្នកទាំងអស់គ្នា")
    assert not prepared.empty and "no_sentence_end" in prepared.codes
    assert prepared.text == "សួស្តី អ្នកទាំងអស់គ្នា"  # normalized only


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
class FixedBackend(detector.Backend):
    key, title = "fixed", "fixed score"

    def __init__(self, p):
        self.p = p

    def score(self, text):
        return self.p


@pytest.mark.parametrize("p, is_ai, strength", [
    (0.95, True, "strong"), (0.80, True, "moderate"), (0.55, True, "weak"),
    (0.05, False, "strong"), (0.25, False, "moderate"), (0.45, False, "weak"),
    (0.5, False, "weak"),  # the threshold is strict, like src/evaluate.py: exactly 0.5 is human
])
def test_verdict_and_strength(p, is_ai, strength):
    prediction = detector.predict(FixedBackend(p), detector.prepare(khmer_text(400)))
    assert prediction.is_ai is is_ai and prediction.strength == strength
    assert prediction.verdict == (detector.VERDICT_AI if is_ai else detector.VERDICT_HUMAN)
    assert prediction.p_ai == p and "not proof" in prediction.note


def test_short_text_note_is_more_careful_about_human_verdicts():
    short = detector.prepare(SENTENCE)
    assert "less reliable" in detector.confidence_note(0.9, short)[1]
    assert "scored as human" in detector.confidence_note(0.1, short)[1]
    assert "short" not in detector.confidence_note(0.1, detector.prepare(khmer_text(400)))[1]


# ---------------------------------------------------------------------------
# the baseline backend and explain
# ---------------------------------------------------------------------------
def test_baseline_backend_scores_and_separates_the_classes(tiny_backend):
    rng = random.Random(5)
    human = detector.prepare(synthetic_doc(rng, HUMAN_LETTERS))
    ai = detector.prepare(synthetic_doc(rng, AI_LETTERS))
    p_human, p_ai = detector.predict(tiny_backend, human), detector.predict(tiny_backend, ai)
    assert 0.0 <= p_human.p_ai < 0.5 < p_ai.p_ai <= 1.0
    assert not p_human.is_ai and p_ai.is_ai


def test_explain_shape_and_signs(tiny_backend, tiny_pipeline):
    rng = random.Random(6)
    prepared = detector.prepare(synthetic_doc(rng, HUMAN_LETTERS, 15) + " " + synthetic_doc(rng, AI_LETTERS, 15))
    explanation = detector.explain(tiny_backend, prepared, k=5)
    assert 0 < len(explanation.toward_ai) <= 5 and 0 < len(explanation.toward_human) <= 5
    assert all(c.weight > 0 for c in explanation.toward_ai) and all(c.weight < 0 for c in explanation.toward_human)
    for items in (explanation.toward_ai, explanation.toward_human):
        weights = [abs(c.weight) for c in items]
        assert weights == sorted(weights, reverse=True)  # largest push first
        assert all(len(c.ngram) >= detector.MIN_NGRAM_CHARS and c.count >= 1 for c in items)
        assert all(c.ngram in prepared.text for c in items)
        for a in items:  # no listed n-gram is a variant (substring either way) of another
            assert not any(a is not b and (a.ngram in b.ngram or b.ngram in a.ngram) for b in items)
    assert set(c.ngram[0] for c in explanation.toward_ai) <= set(AI_LETTERS + " ")  # AI-only letters push towards AI
    assert set(c.ngram[0] for c in explanation.toward_human) <= set(HUMAN_LETTERS + " ")
    assert explanation.logit == pytest.approx(float(tiny_pipeline.decision_function([prepared.text])[0]), abs=1e-4)


def test_explain_text_with_no_known_ngrams(tiny_backend):
    explanation = detector.explain(tiny_backend, detector.prepare("xyzxyzxyz!"))  # none of these characters (or a space) were in training
    assert explanation.toward_ai == () and explanation.toward_human == ()
    assert explanation.logit == pytest.approx(float(tiny_backend._lr.intercept_[0]))


def test_show_ngram_makes_invisible_parts_visible():
    assert detector.show_ngram("ក ខ") == "ក␣ខ"
    assert detector.show_ngram("ក\nខ") == "ក↵ខ"
    assert detector.show_ngram("ាក") == "◌ាក"  # lone vowel sign gets a dotted circle
    assert detector.show_ngram("កា") == "កា"


def test_pipeline_without_the_expected_steps_still_predicts_but_cannot_explain(tmp_path):
    rng = random.Random(1)
    texts = [synthetic_doc(rng, HUMAN_LETTERS) for _ in range(10)] + [synthetic_doc(rng, AI_LETTERS) for _ in range(10)]
    pipe = Pipeline([("vec", baseline.build_vectorizer((1, 2), min_df=1)), ("clf", LogisticRegression())]).fit(texts, [0] * 10 + [1] * 10)
    joblib.dump(pipe, tmp_path / "other.joblib")
    backend = detector.BaselineBackend(tmp_path / "other.joblib")
    prepared = detector.prepare(texts[0])
    assert 0.0 <= detector.predict(backend, prepared).p_ai <= 1.0
    assert detector.explain(backend, prepared) is None


def test_model_with_other_classes_is_refused(tmp_path):
    pipe = Pipeline([("tfidf", baseline.build_vectorizer((1, 2), min_df=1)), ("lr", LogisticRegression())])
    pipe.fit(["កខគ។", "ឃងច។", "កខគ។", "ឃងច។"], ["human", "ai", "human", "ai"])
    joblib.dump(pipe, tmp_path / "wrong.joblib")
    with pytest.raises(detector.ModelUnavailable, match="classes"):
        detector.BaselineBackend(tmp_path / "wrong.joblib")


# ---------------------------------------------------------------------------
# configuration and loading
# ---------------------------------------------------------------------------
def test_config_from_env():
    assert detector.config_from_env({}) == detector.ModelConfig("baseline", "")
    env = {detector.MODEL_ENV: " XLMR ", detector.MODEL_PATH_ENV: " models/x "}
    assert detector.config_from_env(env) == detector.ModelConfig("xlmr", "models/x")
    assert detector.config_from_env({detector.MODEL_ENV: ""}).kind == "baseline"


def test_load_baseline_from_absolute_and_relative_path(tiny_model_path, monkeypatch):
    result = detector.try_load(detector.ModelConfig("baseline", str(tiny_model_path)))
    assert result.error == "" and isinstance(result.backend, detector.BaselineBackend)
    monkeypatch.chdir(tiny_model_path.parent)
    assert detector.try_load(detector.ModelConfig("baseline", tiny_model_path.name)).backend is not None


def test_missing_corrupt_and_unknown_models_give_messages_not_exceptions(tmp_path):
    missing = detector.try_load(detector.ModelConfig("baseline", str(tmp_path / "nope.joblib")))
    assert missing.backend is None and "not found" in missing.error and detector.MODEL_PATH_ENV in missing.error
    (tmp_path / "bad.joblib").write_bytes(b"not a pickle")
    corrupt = detector.try_load(detector.ModelConfig("baseline", str(tmp_path / "bad.joblib")))
    assert corrupt.backend is None and "Could not load" in corrupt.error
    folder = detector.try_load(detector.ModelConfig("baseline", str(tmp_path)))
    assert folder.backend is None and "not found" in folder.error
    unknown = detector.try_load(detector.ModelConfig("gpt", ""))
    assert unknown.backend is None and "baseline" in unknown.error and "xlmr" in unknown.error
    with pytest.raises(detector.ModelUnavailable):
        detector.load_backend(detector.ModelConfig("gpt", ""))


def test_xlmr_is_reported_as_unavailable(tmp_path, monkeypatch):
    result = detector.try_load(detector.ModelConfig("xlmr", ""))
    assert result.backend is None and "XLM-R" in result.error and detector.MODEL_ENV in result.error
    assert "no model folder" in result.error  # models/xlmr does not exist in the repo
    monkeypatch.setattr(detector.importlib.util, "find_spec", lambda name: None)  # transformers and torch "missing"
    assert "torch and transformers are not installed" in detector.try_load(detector.ModelConfig("xlmr", "")).error
    hub = detector.try_load(detector.ModelConfig("xlmr", "someone/xlmr-khmer-detector"))
    assert hub.backend is None and "no model folder" not in hub.error  # a Hub id is not a missing folder
    folder = detector.try_load(detector.ModelConfig("xlmr", str(tmp_path)))
    assert folder.backend is None and "no model folder" not in folder.error and "not part of this version" in folder.error


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------
def test_examples_file():
    examples = detector.load_examples()
    human, ai = examples["human"], examples["ai"]
    assert human["label"] == 0 and ai["label"] == 1
    assert human["source"] == "wikipedia" and human["generator"] == "human" and ai["generator"] != "human"
    assert "CC BY-SA" in human["attribution"] and human["url"] in human["attribution"] and human["title"] in human["attribution"]
    assert ai["generator"] in ai["attribution"]
    for example in examples.values():
        assert 600 <= len(example["text"]) <= 1200 and example["length"] == len(example["text"])
        prepared = detector.prepare(example["text"])  # already clean: nothing is cut and nothing is flagged
        assert prepared.text == example["text"] and prepared.notices == ()


def test_load_examples_needs_both_texts(tmp_path):
    path = tmp_path / "examples.json"
    path.write_text(json.dumps({"human": {"text": "x"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="ai"):
        detector.load_examples(path)


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="models/baseline_tfidf_lr.joblib not found (run src/baseline.py)")
def test_real_baseline_model_on_the_examples():
    result = detector.try_load(detector.ModelConfig("baseline", str(REAL_MODEL)))
    assert result.backend is not None, result.error
    examples = detector.load_examples()
    human = detector.predict(result.backend, detector.prepare(examples["human"]["text"]))
    ai = detector.predict(result.backend, detector.prepare(examples["ai"]["text"]))
    assert not human.is_ai and ai.is_ai  # the examples must show what the demo is for
    explanation = detector.explain(result.backend, detector.prepare(examples["ai"]["text"]))
    assert explanation.toward_ai and explanation.toward_human
    assert explanation.logit == pytest.approx(float(np.log(ai.p_ai / (1 - ai.p_ai))), abs=1e-4)


# ---------------------------------------------------------------------------
# the Streamlit page (headless, no server)
# ---------------------------------------------------------------------------
APP_FILE = ROOT / "app" / "app.py"
needs_streamlit = pytest.mark.skipif(importlib.util.find_spec("streamlit") is None, reason="streamlit not installed")


def run_app(monkeypatch, model="baseline", path=""):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(detector.MODEL_ENV, model)
    monkeypatch.setenv(detector.MODEL_PATH_ENV, str(path))
    return AppTest.from_file(str(APP_FILE), default_timeout=60).run()


def texts_of(elements):
    return [e.value for e in elements]


@needs_streamlit
def test_page_starts_empty_and_examples_fill_the_box(tiny_model_path, monkeypatch):
    at = run_app(monkeypatch, path=tiny_model_path)
    assert not at.exception
    assert any("Nothing to check" in t for t in texts_of(at.info)) and len(at.metric) == 0
    examples = detector.load_examples()

    at.button[0].click().run()  # "Human-written example"
    assert not at.exception and at.text_area[0].value == examples["human"]["text"]
    assert len(at.metric) == 1 and any("CC BY-SA" in t for t in texts_of(at.caption))  # attribution under the human example

    at.button[1].click().run()  # "AI-generated example"
    assert at.text_area[0].value == examples["ai"]["text"]
    assert not any("CC BY-SA" in t for t in texts_of(at.caption)) and any(examples["ai"]["generator"] in t for t in texts_of(at.caption))


@needs_streamlit
def test_page_scores_typed_text_and_warns_about_short_text(tiny_model_path, monkeypatch):
    at = run_app(monkeypatch, path=tiny_model_path)
    at.text_area[0].set_value(SENTENCE)
    at.button[2].click().run()  # "Check text" form button
    assert not at.exception and len(at.metric) == 1
    assert any("Short text" in t for t in texts_of(at.warning))
    assert any("Likely" in t for t in texts_of(at.markdown))


@needs_streamlit
def test_page_shows_the_error_when_the_model_is_unavailable(monkeypatch):
    at = run_app(monkeypatch, model="xlmr")
    assert not at.exception and len(at.error) == 1 and "XLM-R" in at.error[0].value
    assert len(at.metric) == 0 and len(at.text_area) == 0
