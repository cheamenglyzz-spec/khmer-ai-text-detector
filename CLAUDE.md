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
- Step 2 IN PROGRESS: full run started 2026-10-25, output `data/generated/ai_raw.jsonl`, log `data/generated/full_run.log`.
  - Run it with `src/run_generation.ps1` in its own window: it reruns `generate_ai.py` (which resumes) every 30 min until done, so daily quotas and session restarts don't end the run.
  - Generators: gemini-3.1-flash-lite, gemma-4-31b-it (Gemini API), openai/gpt-oss-120b, qwen/qwen3.8-27b (Groq).
  - Prompts are built from the step-3-cleaned human text; 7,230 passages qualify (~1,807 per generator).
  - The plan is frozen in `data/generated/plan.jsonl` (built with clean.py as of commit c4e86d7). Generation reads it, so later clean.py changes don't reshuffle generators. Never use `--rebuild-plan` mid-run.
  - Free-tier limits: gpt-oss-120b 200k tokens/day (~150 texts/day, the bottleneck); Groq 1,000 requests/day per model; qwen ~1,000 output tokens/min (100 s interval); gemma is slow (~50 s/request) and often returns 500s (retried on rerun).
  - First smoke-test output was moved to `data/generated/old_smoke_test/`.
- Step 3 script READY: `src/clean.py` (tests in `tests/`). Same rules for both classes: strip invisible chars (gpt-oss adds U+200B), Wikipedia footers, title/URL/wikitext and fragment lines, trim to last sentence end, filter, dedup, length-match per 200-char bin.
  - Known leftover: inline site-nav text in some mc4 passages (e.g. "- Kampuchea Thmey Home ...").
- Steps 4–5 scripts READY: `src/split.py`, `src/baseline.py`. Their reports in `reports/` are from partial data; rerun steps 3–5 once generation is done.
- Step 6 READY and run once on a Colab T4 (2026-10-26, partial data: 1,404 train rows; 150 s): `src/finetune.py` + `notebooks/02_finetune_xlmr.ipynb`.
  - Results in `reports/step6_xlmr_*`: F1 val 0.951 / test 0.978 / test_ood 0.949, a bit under the TF-IDF baseline; all errors but one are humans called AI. Rerun on the full data.
  - Colab route: `python src/make_colab_bundle.py` makes `colab_bundle.zip` (code + splits, gitignored); drag it into the root of My Drive, open `Colab Notebooks/02_finetune_xlmr.ipynb` in Drive, T4, Run all. Outputs go to Drive `khmer-ai-text-detector/`.
  - The GitHub repo is still private (making it public did not take effect), so the notebook's `CODE_SOURCE = "github"` path needs `USE_TOKEN = True` + a `GITHUB_TOKEN` secret.
  - At 512 tokens about half the texts are truncated, humans more often than AI (~60% vs ~50% in train); check this in step 7.

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
