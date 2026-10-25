"""Step 2: generate AI-written Khmer text, topic-matched 1:1 to data/human.csv.

Every human passage gets exactly one AI passage, written by one generator.
Generators are assigned round-robin after a seed-42 shuffle, so the mapping is
deterministic and a `--limit N` run is a stable, balanced prefix of the full run.

Human passages are first cleaned with the exact per-text rules of step 3 (src/clean.py,
clean_human_text). Passages step 3 would drop are left out of the assignment, and the
prompts (target length, paragraph count, web_hint opening) are built from the cleaned text.

The assignment is frozen in data/generated/plan.jsonl (one task per line, prompts included).
The first real run writes it; every later run, including each rerun of run_generation.ps1, loads
it and ignores human.csv and src/clean.py. Otherwise any change to the cleaning rules would
reshuffle which generator writes which passage while a run is in progress. `--rebuild-plan`
replaces it on purpose; a plan built for other generators than --generators is an error.

Output (append-only JSONL, flushed after every record, doubles as the checkpoint):
    data/generated/ai_raw.jsonl      one record per success
    data/generated/ai_errors.jsonl   one record per failed row (the run continues)
Re-running skips every human_id that is already in ai_raw.jsonl.

Keys come from `.env` (GEMINI_API_KEY, GROQ_API_KEY); they are never printed.

Usage:
    python src/generate_ai.py --dry-run --limit 20     # plan + example prompts, no API calls
    python src/generate_ai.py --check-models           # confirm the model ids with your keys
    python src/generate_ai.py --limit 20               # small real test (5 per generator)
    python src/generate_ai.py                          # full run (resumable, Ctrl+C is safe)
    python src/generate_ai.py --rebuild-plan           # build a new frozen plan from human.csv and clean.py
    python src/generate_ai.py --generators gemini:gemini-3.1-flash-lite,groq:openai/gpt-oss-120b@4
        (optional "@seconds" after a model overrides its minimum request interval)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import groq
import httpx
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `python src/generate_ai.py` can import its sibling module as src.clean

HUMAN_CSV = ROOT / "data" / "human.csv"
OUT_PATH = ROOT / "data" / "generated" / "ai_raw.jsonl"
PLAN_PATH = ROOT / "data" / "generated" / "plan.jsonl"  # the frozen assignment, see the module docstring
ERRORS_NAME = "ai_errors.jsonl"  # lives next to the output file unless --errors is given

SEED = 42
TEMPERATURE = 0.8
MAX_OUTPUT_TOKENS = 4096  # Khmer is token-heavy; 2,500 chars can take well over 2k tokens
REQUEST_TIMEOUT_S = 120

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
# One model per family, so any one can be held out as the out-of-distribution generator.
# qwen gets a long interval: Groq caps it at ~1,000 output tokens/min on the free tier.
DEFAULT_GENERATORS = (
    "gemini:gemini-3.1-flash-lite,"
    "gemini:gemma-4-31b-it,"
    "groq:openai/gpt-oss-120b,"
    "groq:qwen/qwen3.8-27b@100"
)
# Minimum seconds between request starts, per generator. Conservative for free tiers
# (Gemini flash is ~10 requests/min; Groq is ~30/min but also has tokens/min limits).
DEFAULT_INTERVAL_S = {"gemini": 7.0, "groq": 3.0}
API_KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}

# Optional per-model request params. If the API rejects them (HTTP 400 naming one of
# them) the provider drops them for the rest of the run and retries once.
GROQ_EXTRA_PARAMS = {
    "qwen/qwen3.8-27b": {"reasoning_effort": "none", "reasoning_format": "hidden"},
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
}
# How to switch thinking off, by model-name substring (first match wins). Flash models take
# thinking_budget=0; gemma rejects that and takes thinking_level instead (unset, it thinks
# away the whole output budget and returns empty text).
GEMINI_NO_THINKING = (("gemma", {"thinking_level": "minimal"}), ("flash", {"thinking_budget": 0}))

# ---------------------------------------------------------------------------
# Prompt templates (edit here)
# ---------------------------------------------------------------------------
PROMPT_WIKI = "wiki_title"
PROMPT_WEB = "web_hint"
HINT_CHARS = 200  # how much of the human text a web_hint prompt shows as the topic hint
CHARS_PER_PARAGRAPH = 400  # paragraph-count target = human length / this
LENGTH_SLACK = 0.2  # prompt asks for target +/- this fraction
# Per-row output cap: tokens = max_chars * this + headroom (reasoning tokens, Khmer is ~0.6-1.1 tok/char).
# A row that hits it ends with finish_reason=length and is trimmed or dropped in step 3.
TOKENS_PER_MAX_CHAR = 1.1
# Models that tokenize Khmer heavily or overshoot the asked length get more room.
TOKENS_PER_MAX_CHAR_BY_MODEL = {"qwen/qwen3.8-27b": 1.6}
TOKEN_HEADROOM = 300

SYSTEM_PROMPT = (
    "You are a native Khmer writer. You write fluent, natural prose in Khmer script "
    "(ភាសាខ្មែរ). Reply with the requested text only: plain prose, no markdown, no "
    "headings, no title line, no bullet points, no English, no preamble such as "
    "\"Here is...\", and no closing remarks."
)

WIKI_PROMPT_TEMPLATE = """\
Write an encyclopedia-style article in Khmer about the following topic.

Topic: {title}

Requirements:
- Write in Khmer (Khmer script) only. No English or other languages, except unavoidable names or numbers.
- Length: about {target_chars} characters, between {min_chars} and {max_chars} characters. Do not write more than {max_chars} characters. Use about {n_paragraphs} paragraphs separated by a single line break.
- Plain prose only: no markdown, no headings, no title line, no bullet points, no lists.
- Neutral, factual, encyclopedic tone, like a Wikipedia article.
- Start directly with the article text. No preamble such as "Here is..." and no closing remarks.
"""

WEB_PROMPT_TEMPLATE = """\
Write a new Khmer news or web article on the same topic as the opening text below.

Opening text (only a topic hint; do NOT copy it, quote it or paraphrase it sentence by sentence):
<<<
{hint}
>>>

Requirements:
- Write in Khmer (Khmer script) only. No English or other languages, except unavoidable names or numbers.
- Length: about {target_chars} characters, between {min_chars} and {max_chars} characters. Do not write more than {max_chars} characters. Use about {n_paragraphs} paragraphs separated by a single line break.
- Plain prose only: no markdown, no headings, no title line, no bullet points, no lists.
- Write it as a fresh news or web article in a natural journalistic style, with your own wording and structure.
- Start directly with the article text. No preamble such as "Here is..." and no closing remarks.
"""

# ---------------------------------------------------------------------------
# Robustness settings
# ---------------------------------------------------------------------------
MIN_TEXT_CHARS = 100
MIN_KHMER_RATIO = 0.5
BAD_OUTPUT_RETRIES = 1  # extra attempts for a row whose output was empty/short/not Khmer
INTERRUPT_GRACE_S = 20  # on Ctrl+C, how long to wait for in-flight requests


@dataclass
class RetryPolicy:
    max_attempts: int = 6  # per row, for rate-limit/5xx/timeout/connection errors
    initial_wait: float = 5.0  # exponential backoff: 5, 10, 20, 40, 80 (+ jitter), capped
    max_wait: float = 90.0
    jitter: float = 5.0
    max_retry_after: float = 180.0  # server asks for a longer wait than this => quota exhausted
    quota_strikes: int = 3  # consecutive rows that still end in a rate-limit error => stop
    max_consecutive_failures: int = 10  # consecutive failed rows of any kind => stop


# ---------------------------------------------------------------------------
# Generators, tasks, assignment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GeneratorSpec:
    provider: str
    model: str
    interval: float

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


def parse_generators(arg: str) -> list[GeneratorSpec]:
    """'gemini:gemini-3.1-flash-lite,groq:openai/gpt-oss-120b@4' -> specs ('@secs' = min interval)."""
    specs: list[GeneratorSpec] = []
    for item in (p.strip() for p in arg.split(",")):
        if not item:
            continue
        interval = None
        if "@" in item:
            item, _, secs = item.rpartition("@")
            interval = float(secs)
        provider, sep, model = item.partition(":")
        provider = provider.strip().lower()
        if not sep or not model.strip() or provider not in API_KEY_ENV:
            raise ValueError(
                f"bad generator {item!r}: use provider:model with provider in {sorted(API_KEY_ENV)}"
            )
        specs.append(GeneratorSpec(provider, model.strip(), DEFAULT_INTERVAL_S[provider] if interval is None else interval))
    keys = [s.key for s in specs]
    if not specs or len(set(keys)) != len(keys):
        raise ValueError("--generators needs at least one generator and no duplicates")
    return specs


@dataclass(frozen=True)
class Task:
    position: int  # index in the deterministic assignment order
    human_id: str
    source: str
    title: str
    url: str
    prompt_type: str
    prompt: str
    target_chars: int
    generator: str  # GeneratorSpec.key


def make_hint(text: str, max_chars: int = HINT_CHARS) -> str:
    flat = " ".join(text.split())
    hint = flat[:max_chars]
    if len(flat) > max_chars:
        cut = hint.rfind(" ")  # Khmer phrases are space-separated; avoid cutting mid-phrase
        if cut >= max_chars * 0.6:
            hint = hint[:cut]
    return hint


def build_prompt(source: str, title: str, text: str) -> tuple[str, str, int]:
    """Return (prompt_type, prompt, target_chars) for one human passage (its cleaned text, see clean_human)."""
    target_chars = min(2500, max(300, int(round(len(text) / 100.0)) * 100))
    n_paragraphs = max(1, int(round(len(text) / CHARS_PER_PARAGRAPH)))
    length = dict(
        target_chars=target_chars,
        min_chars=int(target_chars * (1 - LENGTH_SLACK)),
        max_chars=int(target_chars * (1 + LENGTH_SLACK)),
        n_paragraphs=n_paragraphs,
    )
    if source == "wikipedia" and title.strip():
        prompt = WIKI_PROMPT_TEMPLATE.format(title=title.strip(), **length)
        return PROMPT_WIKI, prompt, target_chars
    prompt = WEB_PROMPT_TEMPLATE.format(hint=make_hint(text), **length)
    return PROMPT_WEB, prompt, target_chars


def row_max_tokens(target_chars: int, cap: int, model: str = "") -> int:
    """Output-token budget for one row: room for the prompt's upper length bound, never above `cap`."""
    max_chars = target_chars * (1 + LENGTH_SLACK)
    per_char = TOKENS_PER_MAX_CHAR_BY_MODEL.get(model, TOKENS_PER_MAX_CHAR)
    return min(cap, int(max_chars * per_char) + TOKEN_HEADROOM)


def load_human(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = {"id", "text", "title", "source", "url"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    if not df["id"].is_unique:
        raise ValueError(f"{path} has duplicate ids")
    return df


def clean_human(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply step 3's per-text rules to every human passage.

    Returns the surviving rows, with `text` replaced by the cleaned text, and the number of
    excluded rows per source (step 3 would drop them, so generating for them wastes quota).
    """
    from src.clean import clean_human_text  # only needed when a plan is built; a frozen plan never imports it

    cleaned = [clean_human_text(text) for text in df["text"]]
    keep = pd.Series([c is not None for c in cleaned], index=df.index)
    excluded = {str(src): int(n) for src, n in df.loc[~keep, "source"].value_counts().sort_index().items()}
    out = df.loc[keep].copy()
    out["text"] = [c for c in cleaned if c is not None]
    return out.reset_index(drop=True), excluded


def build_assignment(df: pd.DataFrame, specs: list[GeneratorSpec], seed: int = SEED) -> list[Task]:
    """Deterministic 1:1 assignment of human rows to generators.

    Rows are shuffled per source with `seed`, then laid out in one sequence where
    position i goes to generator i % G. The source alternates across rounds, shifted
    by the generator index, so every generator sees a near 50/50 mix of sources and
    any prefix of the sequence (what `--limit` takes) stays balanced.
    """
    rng = random.Random(seed)
    sources = sorted(df["source"].unique())
    queues = {s: list(df.index[df["source"] == s]) for s in sources}
    for s in sources:
        rng.shuffle(queues[s])
    n_gen = len(specs)
    tasks: list[Task] = []
    i = 0
    while any(queues.values()):
        want = sources[((i // n_gen) + (i % n_gen)) % len(sources)]
        src = want if queues[want] else next(s for s in sources if queues[s])
        row = df.loc[queues[src].pop(0)]
        prompt_type, prompt, target = build_prompt(row["source"], row["title"], row["text"])
        tasks.append(
            Task(
                position=i,
                human_id=row["id"],
                source=row["source"],
                title=row["title"].strip(),
                url=row["url"],
                prompt_type=prompt_type,
                prompt=prompt,
                target_chars=target,
                generator=specs[i % n_gen].key,
            )
        )
        i += 1
    return tasks


def write_plan(path: Path, tasks: list[Task]) -> None:
    """Write the frozen plan atomically (temp file, then replace), so a crash never leaves a half plan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for task in tasks:
            f.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_plan(path: Path) -> list[Task]:
    """Read a frozen plan, ordered by position. Strict: a bad line, a missing field or a repeated human_id is an error."""
    names = [f.name for f in fields(Task)]
    tasks: list[Task] = []
    with open(path, "rb") as f:
        for n, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                data = json.loads(raw.decode("utf-8"))
                tasks.append(Task(**{name: data[name] for name in names}))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path.name} line {n} is not a valid plan row ({type(exc).__name__}: {exc})") from exc
    if not tasks:
        raise ValueError(f"{path.name} is empty")
    if len({t.human_id for t in tasks}) != len(tasks):
        raise ValueError(f"{path.name} has a repeated human_id")
    return sorted(tasks, key=lambda t: t.position)


def plan_generator_problem(tasks: list[Task], specs: list[GeneratorSpec]) -> str | None:
    """A message if the plan was built for other generators than the ones asked for, else None."""
    in_plan, wanted = {t.generator for t in tasks}, {s.key for s in specs}
    if in_plan == wanted:
        return None
    return (
        f"the frozen plan is for generators {sorted(in_plan)} but --generators is {sorted(wanted)}; "
        "pass --rebuild-plan to replace the plan (the saved texts are kept)"
    )


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (qwen3 safety net); an unclosed block is dropped too."""
    text = _THINK_BLOCK.sub("", text)
    low = text.lower()
    if "</think>" in low:  # closing tag without an opening one: keep what follows it
        text = text[low.rfind("</think>") + len("</think>"):]
        low = text.lower()
    if "<think>" in low:
        text = text[: low.find("<think>")]
    return text.strip()


def khmer_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    khmer = sum(1 for c in chars if "ក" <= c <= "៿" or "᧠" <= c <= "᧿")
    return khmer / len(chars)


def check_output(text: str) -> str | None:
    """Return a reason string if the text is unusable, else None."""
    if not text:
        return "empty output"
    if len(text) < MIN_TEXT_CHARS:
        return f"too short ({len(text)} chars)"
    ratio = khmer_ratio(text)
    if ratio < MIN_KHMER_RATIO:
        return f"not Khmer (Khmer script ratio {ratio:.2f})"
    return None


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
class StopRequested(Exception):
    """Ctrl+C / stop event seen while waiting."""


class QuotaExhausted(Exception):
    """Daily (or otherwise not-worth-waiting-for) quota is used up."""


class BadOutput(Exception):
    """The API call worked but the text is empty/short/not Khmer."""


@dataclass
class ErrInfo:
    kind: str  # rate_limit | server | timeout | connection | auth | not_found | client | bad_output | other
    retryable: bool
    daily: bool
    retry_after: float | None
    status: int | None
    message: str

    @property
    def stop_generator(self) -> bool:  # no point continuing with this model/key
        return self.kind in ("auth", "not_found")


_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)")
_RETRY_IN_TEXT = re.compile(r"(?:try again|retry) in\s+((?:\d+(?:\.\d+)?\s*(?:ms|h|m|s)\s*)+)", re.IGNORECASE)
_DAILY_MARKERS = (
    "per day", "perday", "(tpd)", "(rpd)", "daily", "limit: 0",
)
_AUTH_MARKERS = ("api key not valid", "api_key_invalid", "invalid api key", "invalid_api_key")


def parse_duration(value: str) -> float | None:
    """'34s', '7.66s', '14m23.5s', '1h2m3s', '250ms' -> seconds."""
    parts = _DURATION.findall(str(value))
    if not parts:
        return None
    unit = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(n) * unit[u] for n, u in parts)


def _retry_after_seconds(exc: BaseException, text: str) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after")
        except Exception:
            raw = None
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    details = getattr(exc, "details", None)  # google-genai keeps the parsed error JSON here
    if isinstance(details, dict):
        err = details.get("error", details)
        for item in err.get("details", []) if isinstance(err, dict) else []:
            if isinstance(item, dict) and item.get("retryDelay"):
                secs = parse_duration(item["retryDelay"])
                if secs is not None:
                    return secs
    m = _RETRY_IN_TEXT.search(text)
    return parse_duration(m.group(1)) if m else None


def classify(exc: BaseException) -> ErrInfo:
    text = " ".join(str(exc).split())
    low = text.lower()
    message = f"{type(exc).__name__}: {text}"[:500]

    def info(kind, retryable=False, daily=False, retry_after=None, status=None):
        return ErrInfo(kind, retryable, daily, retry_after, status, message)

    if isinstance(exc, BadOutput):
        return info("bad_output")
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, groq.APITimeoutError)):
        return info("timeout", retryable=True)
    if isinstance(exc, (httpx.TransportError, ConnectionError, groq.APIConnectionError)):
        return info("connection", retryable=True)

    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "code", None)  # google.genai.errors.APIError.code
    if not isinstance(status, int):
        return info("other")

    if status == 429:
        retry_after = _retry_after_seconds(exc, text)
        daily = any(marker in low for marker in _DAILY_MARKERS)
        return info("rate_limit", retryable=not daily, daily=daily, retry_after=retry_after, status=status)
    if status in (408, 409, 425) or status >= 500:
        return info("server", retryable=True, retry_after=_retry_after_seconds(exc, text), status=status)
    if status in (401, 403) or any(marker in low for marker in _AUTH_MARKERS):
        return info("auth", status=status)
    if status == 404:
        return info("not_found", status=status)
    return info("client", status=status)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
@dataclass
class Completion:
    text: str
    finish_reason: str


def normalize_finish_reason(reason) -> str:
    name = getattr(reason, "name", reason)
    name = str(name or "unknown").lower()
    return "length" if name == "max_tokens" else name


class MissingKey(Exception):
    pass


class Provider:
    """One model on one API. `generate` is the only thing the worker calls."""

    name = ""

    def __init__(self, model: str):
        self.model = model
        self._extras_ok = True

    def extras(self) -> dict:
        return {}

    def _request(self, system: str, prompt: str, max_tokens: int, temperature: float, use_extras: bool) -> Completion:
        raise NotImplementedError

    def list_models(self) -> set[str]:
        raise NotImplementedError

    def generate(self, system: str, prompt: str, *, max_tokens: int, temperature: float) -> Completion:
        extras = self.extras()
        if not (extras and self._extras_ok):
            return self._request(system, prompt, max_tokens, temperature, False)
        try:
            return self._request(system, prompt, max_tokens, temperature, True)
        except Exception as exc:
            low = str(exc).lower()
            names = [*extras, "reasoning", "thinking"]
            if classify(exc).status == 400 and any(n in low for n in names):
                log(f"[{self.name}:{self.model}] API rejected optional params {sorted(extras)}; retrying without them")
                self._extras_ok = False
                return self._request(system, prompt, max_tokens, temperature, False)
            raise


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        self.client = genai.Client(
            api_key=api_key, http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000)
        )

    def extras(self) -> dict:
        return next((dict(cfg) for hint, cfg in GEMINI_NO_THINKING if hint in self.model), {})

    def _request(self, system, prompt, max_tokens, temperature, use_extras) -> Completion:
        config = genai_types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=genai_types.ThinkingConfig(**self.extras()) if use_extras else None,
        )
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        candidate = resp.candidates[0] if resp.candidates else None
        finish = normalize_finish_reason(candidate.finish_reason if candidate else None)
        if candidate is None:  # prompt blocked before generation
            block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
            finish = f"blocked:{normalize_finish_reason(block)}"
        return Completion(text=resp.text or "", finish_reason=finish)

    def list_models(self) -> set[str]:
        return {m.name.removeprefix("models/") for m in self.client.models.list()}


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        # max_retries=0: tenacity in the worker owns all retrying
        self.client = groq.Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_S, max_retries=0)

    def extras(self) -> dict:
        return dict(GROQ_EXTRA_PARAMS.get(self.model, {}))

    def _request(self, system, prompt, max_tokens, temperature, use_extras) -> Completion:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        kwargs = dict(
            model=self.model, messages=messages, temperature=temperature, max_completion_tokens=max_tokens
        )
        if use_extras:
            kwargs.update(self.extras())
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return Completion(
            text=choice.message.content or "", finish_reason=normalize_finish_reason(choice.finish_reason)
        )

    def list_models(self) -> set[str]:
        return {m.id for m in self.client.models.list().data}


PROVIDER_CLASSES = {"gemini": GeminiProvider, "groq": GroqProvider}


def make_provider(spec: GeneratorSpec) -> Provider:
    env = API_KEY_ENV[spec.provider]
    api_key = os.environ.get(env, "").strip()
    if not api_key:
        raise MissingKey(f"{env} is not set (add it to .env)")
    return PROVIDER_CLASSES[spec.provider](spec.model, api_key)


# ---------------------------------------------------------------------------
# JSONL checkpoint
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    tqdm.write(f"{datetime.now().strftime('%H:%M:%S')} {msg}")


def repair_jsonl_tail(path: Path) -> None:
    """A crash can leave a half-written last line; drop it (it was never a finished record)."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "r+b") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return
        start = data.rfind(b"\n") + 1
        try:
            json.loads(data[start:].decode("utf-8"))
            f.write(b"\n")  # complete record that only lacks its newline
        except ValueError:
            f.truncate(start)
            log(f"WARNING: dropped a truncated last line ({len(data) - start} bytes) from {path.name}")


def read_jsonl(path: Path) -> list[dict]:
    """Read records, skipping blank or unparseable lines."""
    records: list[dict] = []
    if not path.exists():
        return records
    bad = 0
    with open(path, "rb") as f:  # decode per line: one bad line (e.g. cut mid-character) must not hide the rest
        for raw in f:
            if not raw.strip():
                continue
            try:
                records.append(json.loads(raw.decode("utf-8")))
            except ValueError:  # includes JSONDecodeError and UnicodeDecodeError
                bad += 1
    if bad:
        log(f"WARNING: skipped {bad} unparseable line(s) in {path.name}")
    return records


def load_done_ids(path: Path) -> set[str]:
    return {str(r["human_id"]) for r in read_jsonl(path) if "human_id" in r}


class JsonlWriter:
    """Thread-safe, append-only JSONL writer. Flushes every record; refuses duplicate keys."""

    def __init__(self, path: Path, unique_key: str | None = None):
        self.path = path
        self.unique_key = unique_key
        path.parent.mkdir(parents=True, exist_ok=True)
        repair_jsonl_tail(path)
        self._seen = {str(r[unique_key]) for r in read_jsonl(path) if unique_key in r} if unique_key else set()
        self._lock = threading.Lock()
        self._closed = False
        self._file = open(path, "a", encoding="utf-8", newline="\n")

    def write(self, record: dict) -> bool:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            if self._closed:
                return False
            if self.unique_key:
                key = str(record[self.unique_key])
                if key in self._seen:
                    return False
                self._seen.add(key)
            self._file.write(line)
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except OSError:
                pass
            return True

    def close(self) -> None:
        with self._lock:  # waits for an in-flight write, so the file never ends mid-line
            if not self._closed:
                self._closed = True
                self._file.close()


# ---------------------------------------------------------------------------
# Worker (one thread per generator)
# ---------------------------------------------------------------------------
@dataclass
class GenStats:
    done: int = 0
    failed: int = 0
    lengths: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    stopped: str = ""  # why this generator stopped early, if it did


class RateLimiter:
    """Keeps at least `interval` seconds between request starts."""

    def __init__(self, interval: float, stop: threading.Event):
        self.interval = interval
        self.stop = stop
        self._last = float("-inf")

    def wait(self) -> None:
        delay = self._last + self.interval - time.monotonic()
        if delay > 0 and self.stop.wait(delay):
            raise StopRequested()
        if self.stop.is_set():
            raise StopRequested()
        self._last = time.monotonic()


def _short(msg: str, n: int = 240) -> str:
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= n else msg[: n - 3] + "..."


class GeneratorWorker:
    def __init__(
        self,
        spec: GeneratorSpec,
        provider: Provider,
        tasks: list[Task],
        out: JsonlWriter,
        errors: JsonlWriter,
        stop: threading.Event,
        policy: RetryPolicy,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float = TEMPERATURE,
        progress=None,
    ):
        self.spec, self.provider, self.tasks = spec, provider, tasks
        self.out, self.errors, self.stop, self.policy = out, errors, stop, policy
        self.max_tokens, self.temperature, self.progress = max_tokens, temperature, progress
        self.stats = GenStats()
        self.limiter = RateLimiter(spec.interval, stop)
        self._tag = f"[{spec.key}]"

    # -- retrying API call -------------------------------------------------
    def _sleep(self, seconds: float) -> None:
        if self.stop.wait(seconds):
            raise StopRequested()

    def _wait(self, retry_state) -> float:
        backoff = wait_exponential_jitter(
            initial=self.policy.initial_wait, max=self.policy.max_wait, jitter=self.policy.jitter
        )(retry_state)
        info = classify(retry_state.outcome.exception())
        return max(backoff, info.retry_after or 0.0)  # honour Retry-After

    def _before_sleep(self, retry_state) -> None:
        info = classify(retry_state.outcome.exception())
        wait = retry_state.next_action.sleep
        log(
            f"{self._tag} {info.kind} (attempt {retry_state.attempt_number}/{self.policy.max_attempts}), "
            f"retrying in {wait:.0f}s: {_short(info.message, 160)}"
        )

    def _attempt(self, task: Task) -> Completion:
        self.limiter.wait()
        try:
            return self.provider.generate(
                SYSTEM_PROMPT,
                task.prompt,
                max_tokens=row_max_tokens(task.target_chars, self.max_tokens, self.spec.model),
                temperature=self.temperature,
            )
        except StopRequested:
            raise
        except Exception as exc:
            info = classify(exc)
            if info.kind == "rate_limit" and (
                info.daily or (info.retry_after or 0.0) > self.policy.max_retry_after
            ):
                raise QuotaExhausted(info.message) from exc
            raise

    def _call_with_retry(self, task: Task) -> Completion:
        retryer = Retrying(
            stop=stop_after_attempt(self.policy.max_attempts),
            wait=self._wait,
            retry=retry_if_exception(lambda e: classify(e).retryable),
            before_sleep=self._before_sleep,
            sleep=self._sleep,
            reraise=True,
        )
        return retryer(self._attempt, task)

    # -- one row -----------------------------------------------------------
    def _process(self, task: Task) -> dict:
        problem = ""
        for _ in range(1 + BAD_OUTPUT_RETRIES):
            comp = self._call_with_retry(task)
            text = strip_thinking(comp.text)
            reason = check_output(text)
            if reason is None:
                return {
                    "id": f"ai_{task.human_id}",
                    "human_id": task.human_id,
                    "text": text,
                    "title": task.title,
                    "source": task.source,
                    "url": task.url,
                    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "generator": self.spec.model,
                    "provider": self.spec.provider,
                    "label": 1,
                    "length": len(text),
                    "prompt_type": task.prompt_type,
                    "finish_reason": comp.finish_reason,
                }
            problem = f"{reason}, finish_reason={comp.finish_reason}"
            log(f"{self._tag} bad output for {task.human_id}: {problem}")
        raise BadOutput(problem)

    def _record_error(self, task: Task, exc: BaseException, info: ErrInfo) -> None:
        self.errors.write(
            {
                "human_id": task.human_id,
                "generator": self.spec.model,
                "provider": self.spec.provider,
                "error_type": info.kind,
                "status": info.status,
                "message": _short(info.message, 400),
                "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )

    def _halt(self, reason: str) -> None:
        self.stats.stopped = reason
        log(f"{self._tag} STOPPED: {reason}. Re-run later to resume the remaining rows.")

    def run(self) -> None:
        consecutive_failures = quota_strikes = 0
        try:
            for task in self.tasks:
                if self.stop.is_set():
                    break
                try:
                    record = self._process(task)
                except StopRequested:
                    break
                except QuotaExhausted as exc:
                    self.stats.failed += 1
                    self._record_error(task, exc, classify(exc.__cause__ or exc))
                    self._halt(f"daily quota exhausted ({_short(str(exc), 120)})")
                    break
                except Exception as exc:
                    info = classify(exc)
                    self.stats.failed += 1
                    self._record_error(task, exc, info)
                    if self.progress:
                        self.progress.update(1)
                    log(f"{self._tag} failed {task.human_id}: {_short(info.message, 200)}")
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
                    self.stats.lengths.append(record["length"])
                    self.stats.targets.append(task.target_chars)
                consecutive_failures = quota_strikes = 0
                if self.progress:
                    self.progress.update(1)
        except Exception as exc:  # a bug must not take the other generators down
            self._halt(f"crashed: {type(exc).__name__}: {exc}")
            log(traceback.format_exc())


def run_generators(
    plan: dict[str, list[Task]],
    specs: list[GeneratorSpec],
    providers: dict[str, Provider],
    out: JsonlWriter,
    errors: JsonlWriter,
    stop: threading.Event,
    policy: RetryPolicy,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = TEMPERATURE,
    progress=None,
) -> tuple[dict[str, GenStats], bool]:
    """Run one thread per generator. Returns (stats by generator key, interrupted)."""
    workers = [
        GeneratorWorker(s, providers[s.key], plan[s.key], out, errors, stop, policy, max_tokens, temperature, progress)
        for s in specs
        if s.key in providers
    ]
    threads = [threading.Thread(target=w.run, name=w.spec.key, daemon=True) for w in workers]
    interrupted = False
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        log(f"Ctrl+C: stopping, waiting up to {INTERRUPT_GRACE_S}s for in-flight requests...")
        try:
            deadline = time.monotonic() + INTERRUPT_GRACE_S
            for t in threads:
                t.join(max(0.0, deadline - time.monotonic()))
        except KeyboardInterrupt:
            log("Second Ctrl+C: leaving now.")
    return {w.spec.key: w.stats for w in workers}, interrupted


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_stats(values: list[int]) -> str:
    if not values:
        return "n/a"
    return (
        f"min {min(values)}  mean {statistics.mean(values):.0f}  "
        f"median {statistics.median(values):.0f}  max {max(values)}"
    )


def print_plan(tasks: list[Task], specs: list[GeneratorSpec], done: set[str], limited: bool) -> None:
    print(f"Assignment: {len(tasks)} human rows{' (after --limit)' if limited else ''}, {len(specs)} generators")
    sources = sorted({t.source for t in tasks})
    header = f"{'generator':<38}{'assigned':>9}" + "".join(f"{s:>11}" for s in sources) + f"{'done':>7}{'pending':>9}"
    print(header)
    print("-" * len(header))
    for spec in specs:
        mine = [t for t in tasks if t.generator == spec.key]
        n_done = sum(t.human_id in done for t in mine)
        row = f"{spec.key:<38}{len(mine):>9}"
        row += "".join(f"{sum(t.source == s for t in mine):>11}" for s in sources)
        print(row + f"{n_done:>7}{len(mine) - n_done:>9}")
    print("Intervals (s between requests): " + ", ".join(f"{s.key}={s.interval:g}" for s in specs))


def example_prompts(tasks: list[Task]) -> str:
    chunks = [f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}\n"]
    for prompt_type in (PROMPT_WIKI, PROMPT_WEB):
        task = next((t for t in tasks if t.prompt_type == prompt_type), None)
        if task:
            chunks.append(
                f"=== EXAMPLE USER PROMPT: {prompt_type} (human_id={task.human_id}, "
                f"generator={task.generator}, target={task.target_chars} chars) ===\n{task.prompt}"
            )
    return "\n".join(chunks)


def print_summary(specs, stats: dict[str, GenStats], total_in_file: int, interrupted: bool) -> None:
    print("\n=== Summary (this run) ===")
    all_lengths: list[int] = []
    for spec in specs:
        st = stats.get(spec.key)
        if st is None:
            print(f"{spec.key}: skipped (no API key)")
            continue
        all_lengths += st.lengths
        extra = f"  STOPPED: {st.stopped}" if st.stopped else ""
        print(f"{spec.key}: done {st.done}, failed {st.failed}{extra}")
        print(f"    length  {fmt_stats(st.lengths)}")
        if st.lengths:
            print(f"    target  mean {statistics.mean(st.targets):.0f} (human length of the same topics)")
    print(f"All generators: done {sum(s.done for s in stats.values())}, failed {sum(s.failed for s in stats.values())}")
    print(f"    length  {fmt_stats(all_lengths)}")
    print(f"Records now in output file: {total_in_file}")
    if interrupted:
        print("Run was interrupted; re-run the same command to resume.")


# ---------------------------------------------------------------------------
# --check-models
# ---------------------------------------------------------------------------
def check_models(specs: list[GeneratorSpec]) -> int:
    listings: dict[str, set[str] | None] = {}
    problems = 0
    for spec in specs:
        try:
            provider = make_provider(spec)
        except MissingKey as exc:
            print(f"[SKIP ] {spec.key}: {exc}")
            problems += 1
            continue
        if spec.provider not in listings:
            try:
                listings[spec.provider] = provider.list_models()
            except Exception as exc:
                listings[spec.provider] = None
                print(f"        ({spec.provider}: list-models failed: {_short(classify(exc).message, 160)})")
        listing = listings[spec.provider]
        listed = "unknown" if listing is None else ("yes" if spec.model in listing else "NO")
        t0 = time.monotonic()
        try:
            comp = provider.generate(
                "You are a helpful assistant.", "Reply with the single word: OK", max_tokens=256, temperature=0.0
            )
            print(
                f"[OK   ] {spec.key}: listed={listed}, {time.monotonic() - t0:.1f}s, "
                f"reply={comp.text.strip()[:30]!r}, finish={comp.finish_reason}"
            )
        except Exception as exc:
            info = classify(exc)
            label = {
                "rate_limit": "BUSY ", "auth": "AUTH ", "not_found": "404  ",
            }.get(info.kind, "FAIL ")
            note = " (reachable but rate-limited/quota)" if info.kind == "rate_limit" else ""
            print(f"[{label}] {spec.key}: listed={listed}{note}: {_short(info.message, 220)}")
            problems += 0 if info.kind == "rate_limit" else 1
        if listing is not None and spec.model not in listing:
            shown = sorted(listing)
            print(f"        {spec.provider} models available: {', '.join(shown[:60])}{' ...' if len(shown) > 60 else ''}")
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AI Khmer text topic-matched to data/human.csv.")
    p.add_argument("--limit", type=int, help="only the first N rows of the assignment (balanced across generators)")
    p.add_argument("--generators", default=DEFAULT_GENERATORS, help="provider:model[@interval_s],... (default: %(default)s)")
    p.add_argument("--out", type=Path, default=OUT_PATH, help="output JSONL (default: %(default)s)")
    p.add_argument("--errors", type=Path, help=f"error JSONL (default: {ERRORS_NAME} next to --out)")
    p.add_argument("--human", type=Path, default=HUMAN_CSV, help="human passages CSV, read only when the plan is built (default: %(default)s)")
    p.add_argument("--plan", type=Path, default=PLAN_PATH, help="frozen plan JSONL, loaded if it exists (default: %(default)s)")
    p.add_argument("--rebuild-plan", action="store_true", help="build the plan again from human.csv and overwrite the plan file")
    p.add_argument("--seed", type=int, default=SEED, help="seed for building a plan (a loaded plan keeps its own assignment)")
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS, help="max output tokens per request")
    p.add_argument("--dry-run", action="store_true", help="show the plan and example prompts; no API calls, no keys needed")
    p.add_argument("--prompts-out", type=Path, help="with --dry-run: also write the example prompts to this UTF-8 file")
    p.add_argument("--check-models", action="store_true", help="test that each configured model is reachable with your keys")
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Khmer must not crash a cp1252 console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    load_dotenv(ROOT / ".env")
    try:
        specs = parse_generators(args.generators)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.check_models:
        return check_models(specs)

    if args.plan.exists() and not args.rebuild_plan:
        try:
            tasks = load_plan(args.plan)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        problem = plan_generator_problem(tasks, specs)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 2
        print(f"Using frozen plan {args.plan}: {len(tasks)} rows")
    else:
        df = load_human(args.human)
        n_loaded = len(df)
        df, excluded = clean_human(df)
        print(
            f"Human passages: {n_loaded} loaded, {len(df)} kept after step-3 cleaning, "
            f"{n_loaded - len(df)} excluded ({', '.join(f'{s} {n}' for s, n in excluded.items()) or 'none'})"
        )
        tasks = build_assignment(df, specs, args.seed)
        if args.dry_run:
            print(f"(dry run: the plan is not written to {args.plan})")
        else:
            write_plan(args.plan, tasks)
            print(f"Wrote frozen plan {args.plan}: {len(tasks)} rows")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    out_path = args.out
    errors_path = args.errors or out_path.with_name(ERRORS_NAME)
    done = load_done_ids(out_path)

    print_plan(tasks, specs, done, args.limit is not None)
    if args.dry_run:
        examples = example_prompts(tasks)
        print("\n" + examples)
        if args.prompts_out:
            args.prompts_out.parent.mkdir(parents=True, exist_ok=True)
            args.prompts_out.write_text(examples, encoding="utf-8")
            print(f"\n(example prompts also written to {args.prompts_out})")
        return 0

    providers: dict[str, Provider] = {}
    for spec in specs:
        try:
            providers[spec.key] = make_provider(spec)
        except MissingKey as exc:
            print(f"warning: skipping {spec.key}: {exc}", file=sys.stderr)
    if not providers:
        print("error: no generator has an API key; see .env.example", file=sys.stderr)
        return 2

    plan = {s.key: [t for t in tasks if t.generator == s.key and t.human_id not in done] for s in specs}
    pending = sum(len(plan[k]) for k in providers)
    if pending == 0:
        print("Nothing to do: every assigned row is already generated (or its generator has no key).")
        return 0

    out = JsonlWriter(out_path, unique_key="human_id")
    errors = JsonlWriter(errors_path)
    stop = threading.Event()
    interrupted = False
    stats: dict[str, GenStats] = {}
    try:
        with tqdm(total=pending, desc="generating", unit="text", dynamic_ncols=True) as bar:
            stats, interrupted = run_generators(
                plan, specs, providers, out, errors, stop, RetryPolicy(), args.max_tokens, args.temperature, bar
            )
    finally:
        out.close()
        errors.close()
    print_summary(specs, stats, len(load_done_ids(out_path)), interrupted)
    return 130 if interrupted else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # Ctrl+C before/after the generation loop (nothing half-written to clean up)
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
