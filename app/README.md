---
title: Khmer AI-Text Detector
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: streamlit
python_version: "3.12"
app_file: app.py
pinned: false
license: mit # placeholder: the repo has no LICENSE file yet; confirm before publishing
---

# Khmer AI-Text Detector (demo)

A small web app for the Khmer AI-text detector. Paste Khmer text and it says whether the text looks human-written or AI-generated, gives the probability, a plain-language confidence note and, for the baseline model, the character n-grams that pushed the score.

This is a research demo, not proof of who wrote a text. It was trained on Khmer Wikipedia and mC4 Khmer web news (human) against text written by four language models (gemini-3.1-flash-lite, gemma-4-31b-it, openai/gpt-oss-120b, qwen/qwen3.8-27b), currently on partial data. Do not use it to accuse anyone. The page repeats these limits under "About this demo".

## Run it locally

From the repository root, with the project's virtual environment:

```powershell
.venv\Scripts\python.exe -m streamlit run app/app.py
```

It needs `models/baseline_tfidf_lr.joblib` (made by `python src/baseline.py`). Pasted text goes through the same cleaning as the training text (`src/clean.py`), so the cleaned text is what is scored.

## Choosing the model

| Environment variable | Meaning |
|---|---|
| `KHMER_DETECTOR_MODEL` | `baseline` (default) or `xlmr` |
| `KHMER_DETECTOR_MODEL_PATH` | baseline: the joblib file. xlmr: a model folder or a Hugging Face Hub id. A relative path is looked up from the working directory, the project root, then this folder. Default: `models/baseline_tfidf_lr.joblib` / `models/xlmr` |

Only the baseline is implemented. `xlmr` shows a clear "not available" message on the page until an XLM-R backend is added in `detector.py` (a `Backend` subclass registered in `LOADERS`; `app.py` does not change).

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit page only |
| `detector.py` | Logic with no Streamlit import: model loading, text preparation and warnings, prediction, explanation |
| `examples.json` | The two example buttons: a Khmer Wikipedia passage (CC BY-SA 4.0, with attribution) and an AI text on the same topic, both from `data/processed/splits/test.jsonl` |
| `requirements.txt` | What a Space needs (scikit-learn is pinned to the version that wrote the joblib) |

Tests: `tests/test_detector.py` (including a headless run of the page).

## Deploying to a Hugging Face Space (not done yet)

Make a Streamlit Space and put this in its root:

```
README.md              this file (the YAML block above configures the Space)
app.py  detector.py  examples.json  requirements.txt
src/__init__.py  src/clean.py        copied from the repository: detector.py reuses the training-text cleaning
models/baseline_tfidf_lr.joblib      copied from the repository (gitignored there; about 6 MB, so add it with Git LFS)
```

`detector.py` finds `src/clean.py` next to itself (Space layout) or one level up (repository layout). After the baseline is retrained on the full data, copy the new joblib, keep `scikit-learn` in `requirements.txt` equal to the version that trained it, and set `TRAINED_ON_PARTIAL_DATA = False` in `app.py`.
