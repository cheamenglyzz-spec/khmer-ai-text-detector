"""Step 8: the logic behind the Streamlit demo. No Streamlit in here, so it can be tested and reused.

    prepare(text)              clean pasted text like the training text (src/clean.py) and collect warnings
    try_load(config)           load the model the config asks for (never raises; returns a LoadResult)
    predict(backend, prepared) verdict, AI probability and a plain-language confidence note
    explain(backend, prepared) the character n-grams that pushed the score (baseline only)

Models are Backend classes. Only the baseline (step 5 joblib pipeline) exists so far. A later XLM-R model plugs in by
adding a Backend subclass and registering it in LOADERS: app.py does not change.

Configuration (environment variables, read by config_from_env):
    KHMER_DETECTOR_MODEL        baseline (default) | xlmr
    KHMER_DETECTOR_MODEL_PATH   baseline: the joblib file; xlmr: a model folder or a Hugging Face Hub id.
                                Default: models/baseline_tfidf_lr.joblib (baseline) / models/xlmr (xlmr).
                                A relative path is looked up from the working directory, the project root, then app/.

src/clean.py is found next to the project root (repo layout: app/detector.py -> ../src/clean.py) or next to this file
(Hugging Face Space layout: detector.py, src/clean.py and models/ in one folder).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

HERE = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    candidates = (HERE.parent, HERE)  # repo layout, then Space layout
    for base in candidates:
        if (base / "src" / "clean.py").is_file():
            return base
    raise ImportError(
        "src/clean.py not found. Looked next to "
        + " and ".join(str(b) for b in candidates)
        + ". Copy src/clean.py (and src/__init__.py) next to the app for deployment."
    )


ROOT = _find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import clean  # noqa: E402  (the training-text cleaning, reused as is)

# ---------------------------------------------------------------------------
# Settings (edit here)
# ---------------------------------------------------------------------------
MODEL_ENV = "KHMER_DETECTOR_MODEL"
MODEL_PATH_ENV = "KHMER_DETECTOR_MODEL_PATH"
DEFAULT_MODEL = "baseline"
DEFAULT_PATHS = {"baseline": "models/baseline_tfidf_lr.joblib", "xlmr": "models/xlmr"}
EXAMPLES_PATH = HERE / "examples.json"

THRESHOLD = 0.5  # "AI" when p_ai is above it (the same rule as src/evaluate.py)
STRONG_SIGNAL = 0.90  # probability of the predicted class from which the signal is called strong ...
MODERATE_SIGNAL = 0.70  # ... and moderate; below that it is weak
TOP_K = 10  # n-grams listed per direction
MIN_NGRAM_CHARS = 2  # single characters are in every text and say little, so explain() leaves them out
MAX_INPUT_CHARS = 20_000  # the app's input limit; the model was trained on 300-2,500 chars

VERDICT_AI = "Likely AI-generated"
VERDICT_HUMAN = "Likely human-written"
EXAMPLE_KEYS = ("human", "ai")


class ModelUnavailable(Exception):
    """The requested model cannot be used; the message says why and is meant for the user."""


# ---------------------------------------------------------------------------
# Preparing the text
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Notice:
    code: str  # no_sentence_end | low_khmer | short | long
    message: str


@dataclass(frozen=True)
class Prepared:
    raw: str
    text: str  # what the model scores: '' for an empty input
    khmer_share: float
    removed_chars: int  # characters the structure cleaning and the trim to the last sentence end took off
    notices: tuple[Notice, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.text

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(n.code for n in self.notices)


def prepare(raw: str) -> Prepared:
    """Clean pasted text exactly as the training text was (src/clean.process_text) and collect warnings.

    The dataset filters (length, Khmer share, sentence end) are warnings here, never rejections. If the cleaning
    leaves nothing (no sentence-end mark anywhere), the text is scored as normalized.
    """
    proc = clean.process_text(raw or "", is_ai=False)
    normalized, cleaned = proc.normalized, proc.filtered.text
    if not normalized:
        return Prepared(raw or "", "", 0.0, 0)

    notices: list[Notice] = []
    text, removed = cleaned, len(normalized) - len(cleaned)
    if proc.filtered.reason == "truncated_no_sentence_end" or not cleaned.strip():  # nothing to cut at: score what was typed
        text, removed = normalized, 0
        notices.append(Notice(
            "no_sentence_end",
            "No sentence ending (។ ៕ ? !) was found. Every training text ended on one, so this text could not be "
            "cleaned the way the training texts were. It is scored as typed and the result is less reliable.",
        ))
    share = clean.khmer_ratio(text)
    if share < clean.MIN_KHMER_RATIO:
        notices.append(Notice(
            "low_khmer",
            f"Only {share:.0%} of the characters are Khmer (the training texts were at least "
            f"{clean.MIN_KHMER_RATIO:.0%}). The result may be meaningless for text in another language.",
        ))
    if len(text) < clean.MIN_CHARS:
        notices.append(Notice(
            "short",
            f"Short text ({len(text)} characters). The model was trained on {clean.MIN_CHARS:,} to {clean.MAX_CHARS:,} "
            "characters and is less reliable below that: in our tests it missed far more AI-written texts at this length.",
        ))
    elif len(text) > clean.MAX_CHARS:
        notices.append(Notice(
            "long",
            f"Long text ({len(text):,} characters). The model was trained on {clean.MIN_CHARS:,} to "
            f"{clean.MAX_CHARS:,} characters; the text is still scored, but results for very long text were not tested.",
        ))
    return Prepared(raw, text, share, removed, tuple(notices))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Contribution:
    ngram: str
    shown: str  # the n-gram with spaces, line breaks and lone combining marks made visible
    weight: float  # TF-IDF value x model coefficient; positive pushes towards AI, negative towards human
    count: int  # how often the n-gram occurs in the text


@dataclass(frozen=True)
class Explanation:
    """Largest pushes first. Only n-grams of MIN_NGRAM_CHARS or more characters are listed, and a variant of an
    already listed n-gram (one that contains it or is part of it) is skipped, so the lists are not one word repeated."""

    toward_ai: tuple[Contribution, ...]
    toward_human: tuple[Contribution, ...]
    logit: float  # ALL contributions (listed or not) plus the intercept: the model's log-odds for AI


class Backend:
    """A detector. `score` takes a prepared text and returns P(AI)."""

    key = ""
    title = ""  # one line for the UI

    def score(self, text: str) -> float:
        raise NotImplementedError

    def explain(self, text: str, k: int = TOP_K) -> Explanation | None:
        """The n-grams behind the score, or None if this backend cannot say."""
        return None


def show_ngram(ngram: str) -> str:
    """Make an n-gram readable: visible space and line break, dotted circle in front of a lone combining mark."""
    shown = ngram.replace(" ", "␣").replace("\n", "↵")
    if shown and unicodedata.category(shown[0]) in ("Mn", "Mc"):
        shown = "◌" + shown
    return shown


class BaselineBackend(Backend):
    """The step 5 pipeline: TF-IDF character n-grams + logistic regression (classes 0 = human, 1 = ai)."""

    key = "baseline"
    title = "TF-IDF character n-grams + logistic regression (baseline)"

    def __init__(self, path: Path):
        import joblib

        try:
            self.pipeline = joblib.load(path)
        except Exception as exc:  # corrupt file, version mismatch, not a pickle, ...
            raise ModelUnavailable(f"Could not load {path.name}: {type(exc).__name__}: {exc}") from exc
        if not hasattr(self.pipeline, "predict_proba") or list(getattr(self.pipeline, "classes_", [])) != [0, 1]:
            raise ModelUnavailable(f"{path.name} is not a classifier with classes [0, 1] (human, ai).")
        steps = getattr(self.pipeline, "named_steps", {})
        self._vectorizer, self._lr = steps.get("tfidf"), steps.get("lr")
        if self._vectorizer is not None and self._lr is not None:  # needed by explain
            self._names = self._vectorizer.get_feature_names_out()
            self._coef = np.asarray(self._lr.coef_[0], dtype=np.float64)
            self._intercept = float(self._lr.intercept_[0])

    @property
    def can_explain(self) -> bool:
        return self._vectorizer is not None and self._lr is not None

    def score(self, text: str) -> float:
        return float(self.pipeline.predict_proba([text])[0, 1])

    def explain(self, text: str, k: int = TOP_K) -> Explanation | None:
        if not self.can_explain:
            return None
        row = self._vectorizer.transform([text]).tocsr()
        idx = row.indices
        weights = row.data.astype(np.float64) * self._coef[idx]
        names = [str(n) for n in self._names[idx]]
        # biggest push first; the n-gram itself breaks ties so the order never depends on the hash order
        order = sorted(range(len(idx)), key=lambda j: (-abs(weights[j]), names[j]))

        def pick(sign: int) -> tuple[Contribution, ...]:
            chosen: list[int] = []
            for j in order:
                if weights[j] * sign <= 0 or len(names[j]) < MIN_NGRAM_CHARS:
                    continue
                if any(names[j] in names[i] or names[i] in names[j] for i in chosen):  # a variant of one already listed
                    continue
                chosen.append(j)
                if len(chosen) == k:
                    break
            return tuple(Contribution(names[j], show_ngram(names[j]), float(weights[j]), text.count(names[j])) for j in chosen)

        return Explanation(pick(+1), pick(-1), float(weights.sum() + self._intercept))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    kind: str = DEFAULT_MODEL
    path: str = ""  # '' = the default place for this kind


def config_from_env(environ: Mapping[str, str] | None = None) -> ModelConfig:
    env = os.environ if environ is None else environ
    return ModelConfig(
        kind=(env.get(MODEL_ENV) or DEFAULT_MODEL).strip().lower(),
        path=(env.get(MODEL_PATH_ENV) or "").strip(),
    )


def find_path(raw: str) -> Path | None:
    """An existing file or folder for `raw`: as given (absolute) or relative to the working directory, ROOT, HERE."""
    path = Path(raw).expanduser()
    for candidate in ([path] if path.is_absolute() else [Path.cwd() / path, ROOT / path, HERE / path]):
        if candidate.exists():
            return candidate
    return None


def _load_baseline(path: str) -> Backend:
    target = path or DEFAULT_PATHS["baseline"]
    found = find_path(target)
    if found is None or not found.is_file():
        raise ModelUnavailable(
            f"Baseline model file not found: {target}. Create it with `python src/baseline.py` or point "
            f"{MODEL_PATH_ENV} at the joblib file."
        )
    return BaselineBackend(found)


def _load_xlmr(path: str) -> Backend:
    # Not built yet. When it is: an XlmrBackend that loads AutoTokenizer / AutoModelForSequenceClassification from the
    # folder or Hub id and scores like src/evaluate.py's XlmrScorer (truncation to 512 tokens, softmax_p_ai on the logits).
    # Until then this only reports why the model cannot be used.
    target = path or DEFAULT_PATHS["xlmr"]
    problems = []
    missing = [pkg for pkg in ("torch", "transformers") if importlib.util.find_spec(pkg) is None]
    if missing:
        problems.append(f"the packages {' and '.join(missing)} are not installed")
    hub_id = bool(path) and re.fullmatch(r"[\w.-]+/[\w.-]+", path)  # a path someone set that looks like "org/name" (the default does not count)
    if find_path(target) is None and not hub_id:
        problems.append(f"no model folder at {target}")
    problems.append("the XLM-R backend is not part of this version of the demo yet")
    raise ModelUnavailable(
        f"The XLM-R model is not available: {'; '.join(problems)}. Set {MODEL_ENV}=baseline to use the baseline model."
    )


LOADERS = {"baseline": _load_baseline, "xlmr": _load_xlmr}


def load_backend(config: ModelConfig) -> Backend:
    loader = LOADERS.get(config.kind)
    if loader is None:
        raise ModelUnavailable(f"Unknown model {config.kind!r} in {MODEL_ENV}. Use one of: {', '.join(LOADERS)}.")
    return loader(config.path)


@dataclass(frozen=True)
class LoadResult:
    backend: Backend | None
    error: str = ""


def try_load(config: ModelConfig) -> LoadResult:
    """load_backend that returns the problem instead of raising, so the app can cache it and show a message."""
    try:
        return LoadResult(load_backend(config))
    except ModelUnavailable as exc:
        return LoadResult(None, str(exc))


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Prediction:
    p_ai: float
    is_ai: bool
    verdict: str
    strength: str  # strong | moderate | weak
    note: str  # plain-language confidence note


def confidence_note(p_ai: float, prepared: Prepared) -> tuple[str, str]:
    """(strength, note) for a score. Short texts get an extra caution: step 7 found AI texts scored as human there."""
    is_ai = p_ai > THRESHOLD
    p_pred = p_ai if is_ai else 1.0 - p_ai
    kind = "AI-generated" if is_ai else "human-written"
    if p_pred >= STRONG_SIGNAL:
        strength, note = "strong", f"Strong signal: the text looks much more like {kind} text than like the other kind."
    elif p_pred >= MODERATE_SIGNAL:
        strength, note = "moderate", f"Moderate signal: the text looks more like {kind} text, but not by a wide margin."
    else:
        strength, note = "weak", "Weak signal: the score is close to 50%, so this is little better than a guess."
    if "short" in prepared.codes:
        note += (
            " The text is short, so the score is less reliable."
            if is_ai
            else " The text is short, and short AI-written texts are often scored as human, so “human” is weak evidence here."
        )
    return strength, note + " Treat it as a hint, not proof."


def predict(backend: Backend, prepared: Prepared) -> Prediction:
    p_ai = float(backend.score(prepared.text))
    is_ai = p_ai > THRESHOLD
    strength, note = confidence_note(p_ai, prepared)
    return Prediction(p_ai, is_ai, VERDICT_AI if is_ai else VERDICT_HUMAN, strength, note)


def explain(backend: Backend, prepared: Prepared, k: int = TOP_K) -> Explanation | None:
    return backend.explain(prepared.text, k)


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
def load_examples(path: Path = EXAMPLES_PATH) -> dict[str, dict]:
    """{'human': {...}, 'ai': {...}} from examples.json; each has text, attribution, generator, source, url."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in EXAMPLE_KEYS if not data.get(k, {}).get("text")]
    if missing:
        raise ValueError(f"{path} needs a 'text' for: {', '.join(missing)}")
    return {k: data[k] for k in EXAMPLE_KEYS}
