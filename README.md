# Khmer AI-Text Detector

Detect whether a Khmer text is human-written or AI-generated.

## Pipeline
1. Collect human Khmer text (Khmer Wikipedia + VOA Khmer, before 30 Nov 2022) — `notebooks/01_collect_human_text.ipynb`
2. Generate AI Khmer text with 3–4 LLMs (Gemini, Groq)
3. Clean + length-match
4. Train/val/test split + held-out generator
5. Baseline: TF-IDF char n-grams + Logistic Regression
6. Fine-tune `xlm-roberta-base`
7. Evaluate: F1, ROC-AUC, per-generator, robustness
8. Streamlit demo on Hugging Face Spaces (`app/`)
9. Publish dataset + model + report

## Layout
```
notebooks/   Colab notebooks (01_, 02_, ...)
src/         shared helper code
app/         Streamlit demo
data/        raw / generated / processed (not in git; lives on Google Drive)
models/      checkpoints (not in git)
reports/     figures + final report
```

Notebooks run on Colab and save data to `MyDrive/khmer-ai-detector/` on Google Drive.
