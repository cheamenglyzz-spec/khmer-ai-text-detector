"""Step 7 helper: have an LLM paraphrase a sample of test rows, for the evasion test in src/evaluate.py.

A detector that works on raw LLM output may still be fooled when the LLM text is rewritten. This script samples rows
from the test and test_ood splits (default 150 AI + 150 human, seed 42) and asks Gemini to rewrite each one in its own
words, keeping meaning and length. evaluate.py then scores the paraphrases: how many AI texts are still caught, and how
many human texts become "AI" (a false-positive check: a human text polished by an LLM is no longer purely human).

Sampling is stratified with equal shares: AI rows per generator, human rows per source (a stratum with too few rows gives
its unused share to the others). The sample comes out interleaved (AI, human, AI, ...) and round-robin over the strata,
so --limit N is a balanced prefix and a rerun with a larger --limit extends the same sample.

Every paraphrase goes through the same text rules as the dataset (src/clean.py, process_text; rewritten human text is
treated like any LLM output, so one over 2,500 chars is trimmed to a sentence end instead of dropped). A paraphrase is
rejected, and tried once more, if it is empty, too short, not Khmer, below 0.65x or above 1.4x the original length, or a
near copy of the original (over 90% of its character 5-grams occur in the original).

Output (append-only JSONL, flushed after every record, doubles as the checkpoint):
    data/eval/paraphrased.jsonl         one record per success: id (of the ORIGINAL row), split, label, generator,
                                        source, prompt_type, orig_length, length, text (the paraphrase), model, date,
                                        finish_reason, ngram_overlap
    data/eval/paraphrased_errors.jsonl  one record per failed row (the run goes on)
Re-running skips every id that is already in the output file. The key comes from `.env` (GEMINI_API_KEY) and is never printed.

Usage:
    python src/paraphrase.py --dry-run --limit 4       # sample table + prompts, no API calls, no key needed
    python src/paraphrase.py --limit 5                 # small real test
    python src/paraphrase.py                           # the full 300-row run (resumable, Ctrl+C is safe)
    python src/paraphrase.py --n-ai 100 --n-human 0    # AI texts only
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `python src/paraphrase.py` can import its sibling modules as src.clean / src.generate_ai
from src.clean import process_text, read_jsonl as read_split  # noqa: E402  (strict reader for the split files)
from src.generate_ai import (  # noqa: E402  (client, error classification and checkpoint writer shared with step 2)
    MAX_OUTPUT_TOKENS, BadOutput, GeneratorSpec, JsonlWriter, MissingKey, QuotaExhausted, RateLimiter, RetryPolicy,
    StopRequested, classify, log, make_provider, read_jsonl, row_max_tokens, strip_thinking,
)

SPLITS_DIR = ROOT / "data" / "processed" / "splits"
OUT_PATH = ROOT / "data" / "eval" / "paraphrased.jsonl"

SEED = 42
SAMPLE_SPLITS = ("test", "test_ood")
N_AI = 150
N_HUMAN = 150
MODEL = "gemini-3.1-flash-lite"
TEMPERATURE = 0.7
INTERVAL_S = 8.0  # minimum seconds between requests; step 2 may be using the same key at the same time
LENGTH_SLACK = 0.2  # the prompt asks for the original length +/- this share
MIN_LENGTH_RATIO, MAX_LENGTH_RATIO = 0.65, 1.4  # accepted paraphrase length / original length (the prompt asks for 0.8-1.2; models overshoot)
NGRAM = 5
MAX_OVERLAP = 0.9  # a paraphrase with more of its 5-grams in the original than this is a copy
BAD_OUTPUT_RETRIES = 1  # extra attempts for a row whose output was unusable

SYSTEM_PROMPT = (
    "You are a native Khmer writer and editor. You rewrite Khmer text fluently and naturally in Khmer script "
    "(ភាសាខ្មែរ). Reply with the rewritten text only: plain prose, no markdown, no headings, no title line, no "
    "bullet points, no English, no preamble such as \"Here is...\", and no closing remarks."
)

PARAPHRASE_PROMPT = """\
Paraphrase the Khmer text below: rewrite it in your own words. Keep the same meaning, facts, names, numbers and order of ideas, but change the wording and the sentence structure.

Text:
<<<
{text}
>>>

Requirements:
- Write in Khmer (Khmer script) only. No English or other languages, except names or numbers that appear in the text.
- Length: about {target_chars} characters, between {min_chars} and {max_chars} characters. Keep the paragraph breaks (a single line break between paragraphs).
- Plain prose only: no markdown, no headings, no bullet points, no quotation marks around the whole text.
- Do not add facts that are not in the text and do not leave facts out.
- Start directly with the rewritten text. No preamble such as "Here is..." and no closing remarks or notes.
"""


# ---------------------------------------------------------------------------
# Prompt, output checks (pure functions)
# ---------------------------------------------------------------------------
def build_prompt(text: str) -> tuple[str, int]:
    """Return (user prompt, target_chars) for one text; the target is its own length."""
    target = len(text)
    prompt = PARAPHRASE_PROMPT.format(
        text=text, target_chars=target, min_chars=int(target * (1 - LENGTH_SLACK)), max_chars=int(target * (1 + LENGTH_SLACK)),
    )
    return prompt, target


def unwrap(raw: str) -> str:
    """Strip <think> blocks, code fences and the <<< >>> markers a model may echo around its answer."""
    text = strip_thinking(raw or "").strip()
    text = re.sub(r"^```[^\n]*\n", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    if text.startswith("<<<"):
        text = text[3:]
    if text.endswith(">>>"):
        text = text[:-3]
    return text.strip()


def ngrams(text: str, n: int = NGRAM) -> set[str]:
    flat = " ".join(text.split())
    return {flat[i : i + n] for i in range(len(flat) - n + 1)}


def ngram_overlap(paraphrase: str, original: str, n: int = NGRAM) -> float:
    """Share of the paraphrase's character n-grams that also occur in the original (1.0 = a copy)."""
    mine = ngrams(paraphrase, n)
    return len(mine & ngrams(original, n)) / len(mine) if mine else 0.0


def clean_paraphrase(raw: str, original: str) -> tuple[str | None, str | None, float | None]:
    """(cleaned paraphrase, None, overlap) if usable, else (None, reason, overlap or None)."""
    text = unwrap(raw)
    if not text:
        return None, "empty output", None
    done = process_text(text, is_ai=True).filtered  # the dataset's own text rules
    if done.reason:
        return None, done.reason, None
    ratio = len(done.text) / len(original)
    if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
        return None, f"length {ratio:.2f}x the original", None
    overlap = ngram_overlap(done.text, original)
    if overlap > MAX_OVERLAP:
        return None, f"near copy of the original (overlap {overlap:.2f})", overlap
    return done.text, None, overlap


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def load_pool(splits_dir: Path, splits=SAMPLE_SPLITS) -> list[dict]:
    pool = []
    for name in splits:
        path = splits_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (run src/split.py first)")
        pool += [{**r, "split": name} for r in read_split(path)]
    return pool


def allocate(sizes: dict[str, int], n: int, rng: random.Random) -> dict[str, int]:
    """Split n over the strata as equally as their sizes allow; leftovers of a small stratum go to the others."""
    alloc = {k: 0 for k in sizes}
    remaining = min(n, sum(sizes.values()))
    while remaining > 0:
        open_keys = sorted(k for k in sizes if alloc[k] < sizes[k])
        rng.shuffle(open_keys)  # who gets the odd extra ones is random but seeded
        share, extra = divmod(remaining, len(open_keys))
        for i, k in enumerate(open_keys):
            give = min(share + (1 if i < extra else 0), sizes[k] - alloc[k])
            alloc[k] += give
            remaining -= give
    return alloc


def round_robin(lists: list[list[dict]]) -> list[dict]:
    out, i = [], 0
    while any(i < len(x) for x in lists):
        out += [x[i] for x in lists if i < len(x)]
        i += 1
    return out


def stratified_pick(rows: list[dict], key, n: int, rng: random.Random) -> list[dict]:
    """n rows with equal shares per key(row); returned round-robin over the strata."""
    strata: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["id"]):  # sorted: the sample does not depend on the file order
        strata.setdefault(str(key(r)), []).append(r)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n, rng)
    return round_robin([rng.sample(strata[k], alloc[k]) for k in sorted(strata)])


def sample_rows(pool: list[dict], n_ai: int = N_AI, n_human: int = N_HUMAN, seed: int = SEED) -> list[dict]:
    """The sample, interleaved AI / human / AI / ...: AI rows stratified by generator, human rows by source."""
    rng = random.Random(seed)
    ai = stratified_pick([r for r in pool if r["label"] == 1], lambda r: r["generator"], n_ai, rng)
    human = stratified_pick([r for r in pool if r["label"] == 0], lambda r: r["source"], n_human, rng)
    out = []
    for i in range(max(len(ai), len(human))):
        out += ai[i : i + 1] + human[i : i + 1]
    return out


def done_ids(path: Path) -> set[str]:
    return {str(r["id"]) for r in read_jsonl(path) if "id" in r}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def _short(msg: str, n: int = 240) -> str:
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= n else msg[: n - 3] + "..."


@dataclass
class RunStats:
    done: int = 0
    failed: int = 0
    stopped: str = ""


class Paraphraser:
    """Runs the rows one after another: rate limit, retry with backoff, checkpoint after every row."""

    def __init__(self, provider, model: str, out: JsonlWriter, errors: JsonlWriter, stop: threading.Event,
                 policy: RetryPolicy, interval: float = INTERVAL_S, temperature: float = TEMPERATURE, progress=None):
        self.provider, self.model, self.out, self.errors = provider, model, out, errors
        self.stop, self.policy, self.temperature, self.progress = stop, policy, temperature, progress
        self.limiter = RateLimiter(interval, stop)
        self.stats = RunStats()

    def _sleep(self, seconds: float) -> None:
        if self.stop.wait(seconds):
            raise StopRequested()

    def _wait(self, retry_state) -> float:
        backoff = wait_exponential_jitter(initial=self.policy.initial_wait, max=self.policy.max_wait, jitter=self.policy.jitter)(retry_state)
        return max(backoff, classify(retry_state.outcome.exception()).retry_after or 0.0)  # honour Retry-After

    def _before_sleep(self, retry_state) -> None:
        info = classify(retry_state.outcome.exception())
        log(f"[{self.model}] {info.kind} (attempt {retry_state.attempt_number}/{self.policy.max_attempts}), "
            f"retrying in {retry_state.next_action.sleep:.0f}s: {_short(info.message, 160)}")

    def _attempt(self, prompt: str, target_chars: int):
        self.limiter.wait()
        try:
            return self.provider.generate(
                SYSTEM_PROMPT, prompt, max_tokens=row_max_tokens(target_chars, MAX_OUTPUT_TOKENS, self.model), temperature=self.temperature,
            )
        except StopRequested:
            raise
        except Exception as exc:
            info = classify(exc)
            if info.kind == "rate_limit" and (info.daily or (info.retry_after or 0.0) > self.policy.max_retry_after):
                raise QuotaExhausted(info.message) from exc
            raise

    def _call(self, prompt: str, target_chars: int):
        retryer = Retrying(
            stop=stop_after_attempt(self.policy.max_attempts), wait=self._wait, retry=retry_if_exception(lambda e: classify(e).retryable),
            before_sleep=self._before_sleep, sleep=self._sleep, reraise=True,
        )
        return retryer(self._attempt, prompt, target_chars)

    def _process(self, row: dict) -> dict:
        prompt, target = build_prompt(row["text"])
        problem = ""
        for _ in range(1 + BAD_OUTPUT_RETRIES):
            comp = self._call(prompt, target)
            text, reason, overlap = clean_paraphrase(comp.text, row["text"])
            if text is not None:
                return {
                    "id": row["id"], "split": row["split"], "label": int(row["label"]), "generator": row["generator"],
                    "source": row["source"], "prompt_type": row.get("prompt_type", ""), "orig_length": len(row["text"]),
                    "length": len(text), "text": text, "model": self.model,
                    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "finish_reason": comp.finish_reason, "ngram_overlap": round(overlap, 4),
                }
            problem = f"{reason}, finish_reason={comp.finish_reason}"
            log(f"[{self.model}] bad output for {row['id']}: {problem}")
        raise BadOutput(problem)

    def _record_error(self, row: dict, info) -> None:
        self.errors.write({
            "id": row["id"], "label": int(row["label"]), "model": self.model, "error_type": info.kind, "status": info.status,
            "message": _short(info.message, 400), "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    def _halt(self, reason: str) -> None:
        self.stats.stopped = reason
        log(f"[{self.model}] STOPPED: {reason}. Re-run later to resume the remaining rows.")

    def run(self, rows: list[dict]) -> RunStats:
        consecutive_failures = quota_strikes = 0
        for row in rows:
            if self.stop.is_set():
                break
            try:
                record = self._process(row)
            except StopRequested:
                break
            except QuotaExhausted as exc:
                self.stats.failed += 1
                self._record_error(row, classify(exc.__cause__ or exc))
                self._halt(f"daily quota exhausted ({_short(str(exc), 120)})")
                break
            except Exception as exc:
                info = classify(exc)
                self.stats.failed += 1
                self._record_error(row, info)
                if self.progress:
                    self.progress.update(1)
                log(f"[{self.model}] failed {row['id']}: {_short(info.message, 200)}")
                if info.stop_generator:
                    self._halt(f"{info.kind} error, check the key / model id")
                    break
                consecutive_failures += 1
                quota_strikes = quota_strikes + 1 if info.kind == "rate_limit" else 0
                if quota_strikes >= self.policy.quota_strikes:
                    self._halt(f"still rate-limited after retries on {quota_strikes} rows in a row")
                    break
                if consecutive_failures >= self.policy.max_consecutive_failures:
                    self._halt(f"{consecutive_failures} failures in a row")
                    break
                continue
            if self.out.write(record):
                self.stats.done += 1
            consecutive_failures = quota_strikes = 0
            if self.progress:
                self.progress.update(1)
        return self.stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_plan(sample: list[dict], done: set[str], limited: bool) -> None:
    print(f"Sample: {len(sample)} rows{' (after --limit)' if limited else ''}, {sum(r['id'] in done for r in sample)} already paraphrased")
    print(f"  {'class':<7}{'stratum':<28}{'in sample':>10}{'done':>7}{'pending':>9}")
    for label, name, key in ((1, "ai", "generator"), (0, "human", "source")):
        for k in sorted({r[key] for r in sample if r["label"] == label}):
            mine = [r for r in sample if r["label"] == label and r[key] == k]
            n_done = sum(r["id"] in done for r in mine)
            print(f"  {name:<7}{k:<28}{len(mine):>10}{n_done:>7}{len(mine) - n_done:>9}")


def example_prompts(rows: list[dict]) -> str:
    chunks = [f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n"]
    for label, name in ((1, "ai"), (0, "human")):
        row = next((r for r in rows if r["label"] == label), None)
        if row:
            prompt, target = build_prompt(row["text"])
            chunks.append(f"=== EXAMPLE USER PROMPT: {name} row {row['id']} ({row['split']}, {row['generator']}, target {target} chars) ===\n{prompt}")
    return "\n".join(chunks)


def print_summary(stats: RunStats, total_in_file: int, interrupted: bool) -> None:
    print(f"\n=== Summary (this run) ===\ndone {stats.done}, failed {stats.failed}" + (f"  STOPPED: {stats.stopped}" if stats.stopped else ""))
    print(f"Records now in output file: {total_in_file}")
    if interrupted:
        print("Run was interrupted; re-run the same command to resume.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paraphrase a sample of test rows with Gemini, for the step 7 evasion test.")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR, help="folder with the split files (default: %(default)s)")
    p.add_argument("--splits", nargs="+", default=list(SAMPLE_SPLITS), help="splits to sample from (default: %(default)s)")
    p.add_argument("--out", type=Path, default=OUT_PATH, help="output JSONL (default: %(default)s)")
    p.add_argument("--errors", type=Path, help="error JSONL (default: <out name>_errors.jsonl next to --out)")
    p.add_argument("--n-ai", type=int, default=N_AI, help="AI rows in the sample (default: %(default)s)")
    p.add_argument("--n-human", type=int, default=N_HUMAN, help="human rows in the sample, 0 = none (default: %(default)s)")
    p.add_argument("--limit", type=int, help="only the first N rows of the sample (interleaved AI / human)")
    p.add_argument("--model", default=MODEL, help="Gemini model (default: %(default)s)")
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--interval", type=float, default=INTERVAL_S, help="minimum seconds between requests (default: %(default)s)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dry-run", action="store_true", help="show the sample and example prompts; no API calls, no key needed")
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Khmer must not crash a cp1252 console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    load_dotenv(ROOT / ".env")
    try:
        pool = load_pool(args.splits_dir, args.splits)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sample = sample_rows(pool, args.n_ai, args.n_human, args.seed)
    if args.limit is not None:
        sample = sample[: args.limit]
    errors_path = args.errors or args.out.with_name(args.out.stem + "_errors.jsonl")
    done = done_ids(args.out)
    print(f"Pool: {len(pool)} rows from {', '.join(args.splits)}; model {args.model}")
    print_plan(sample, done, args.limit is not None)
    pending = [r for r in sample if r["id"] not in done]
    if args.dry_run:
        print("\n" + example_prompts(pending or sample))
        return 0
    if not pending:
        print("Nothing to do: every sampled row is already paraphrased.")
        return 0

    try:
        provider = make_provider(GeneratorSpec("gemini", args.model, args.interval))
    except MissingKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out, errors = JsonlWriter(args.out, unique_key="id"), JsonlWriter(errors_path)
    stop = threading.Event()
    runner = Paraphraser(provider, args.model, out, errors, stop, RetryPolicy(), args.interval, args.temperature)
    interrupted = False
    try:
        with tqdm(total=len(pending), desc="paraphrasing", unit="text", dynamic_ncols=True) as bar:
            runner.progress = bar
            try:
                runner.run(pending)
            except KeyboardInterrupt:
                interrupted = True
                stop.set()
                log("Ctrl+C: stopped; finished rows are saved.")
    finally:
        out.close()
        errors.close()
    print_summary(runner.stats, len(done_ids(args.out)), interrupted)
    return 130 if interrupted else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
