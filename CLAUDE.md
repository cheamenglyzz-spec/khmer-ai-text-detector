# Khmer AI-Text Detector

Portfolio project: a binary classifier that detects whether Khmer text is human-written or AI-generated. The dataset, model and report will be published on Hugging Face and GitHub.

## Environment
- Local machine (via Claude Code): data collection, generation, cleaning, splits, baseline, Streamlit app, publishing.
- Google Colab (free GPU): fine-tuning xlm-roberta-base and heavy evaluation. Training notebooks live in this repo and are opened in Colab from GitHub.
- Free APIs for AI text: Gemini (Google AI Studio), Groq. Keys go in `.env` (never commit it).

## Pipeline
1. Collect human Khmer text: Khmer Wikipedia + VOA Khmer articles dated before 30 Nov 2022 (pre-ChatGPT).
2. Generate AI Khmer text with 3–4 LLMs, topic-matched to the human texts.
3. Clean, deduplicate, and length-match the human and AI texts.
4. Train/val/test split, holding out one generator for the out-of-distribution test.
5. Baseline: TF-IDF char n-grams + Logistic Regression.
6. Fine-tune xlm-roberta-base (Colab GPU).
7. Evaluate: F1, ROC-AUC, per-generator results, robustness (paraphrase, truncation).
8. Streamlit demo on Hugging Face Spaces.
9. Publish the dataset, model and report.

## Status
- Step 1 DONE: `01_collect_human_text.ipynb` was run locally (Windows), output in `data/human.csv`.
  - 7,964 passages: Wikipedia 3,969 + mC4 Khmer web news 3,995 (crawled 2013–2020).
  - Length 300–2,500 chars, mean ~1,484 chars.
  - VOA Khmer skipped (SSL error on voakhmer.com); mC4 already includes ~147 khmer.voanews.com passages.
  - Known nit: some Wikipedia passages start with leftover "()" from stripped content; clean in step 3.
- Step 2 IN PROGRESS: `src/generate_ai.py` is tested (100- and 40-row runs); the full run is not started.
  - Generators: gemini-3.1-flash-lite, gemma-4-31b-it (Gemini API), openai/gpt-oss-120b, qwen/qwen3.8-27b (Groq).
  - Prompts are built from the step-3-cleaned human text; 7,230 passages qualify (~1,807 per generator).
  - Free-tier limits: Groq 1,000 requests/day per model, qwen ~1,000 output tokens/min (100 s interval); gemma often returns 500s (retried on rerun).
  - `data/generated/ai_raw.jsonl` still holds 15 rows from the first smoke test; clear it before the full run.
- Step 3 script READY: `src/clean.py` (tests in `tests/`). Same rules for both classes: strip invisible chars (gpt-oss adds U+200B), Wikipedia footers, title/URL/wikitext and fragment lines, trim to last sentence end, filter, dedup, length-match per 200-char bin.
  - Known leftover: inline site-nav text in some mc4 passages (e.g. "- Kampuchea Thmey Home ...").

## Layout
```
data/raw/        scraped human text, raw API outputs (gitignored)
data/processed/  cleaned, length-matched, split files
notebooks/       Colab notebooks (training, eval)
src/             reusable Python scripts (collect, generate, clean, split, baseline)
app/             Streamlit demo
reports/         figures, metrics, final report
```

## Conventions
- Python 3.10+, dependencies in `requirements.txt`.
- Each record: `text, label, source, generator, url, date, length`.
- Labels are integers: `0` = human, `1` = ai.
- Save data as JSONL/CSV in UTF-8; Khmer text must round-trip without mojibake.
- Human-text quota is 4,000 per source.
- Fixed random seed (42) for all splits and sampling.
- Respect API rate limits: retry with backoff, resume from checkpoints.
- After each step, report the counts, the length distribution and a few random samples.
