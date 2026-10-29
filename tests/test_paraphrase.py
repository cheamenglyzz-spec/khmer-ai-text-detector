"""Tests for src/paraphrase.py. No network: the provider is a fake and the real one is never built (run: python -m pytest tests)."""
import json
import random
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import clean, paraphrase  # noqa: E402
from src.generate_ai import Completion, JsonlWriter, RetryPolicy  # noqa: E402

LETTERS = "កខគឃងចឆជតថទធនបផព"


def khmer_text(rng, sentences=10):
    """~20 Khmer-letter words per sentence ending in a full stop: long enough (>300 chars) to pass the dataset rules."""
    return " ".join(
        " ".join("".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 6))) for _ in range(6)) + "។" for _ in range(sentences)
    )


def make_pool(n_per_generator=6, n_per_source=8, seed=1):
    """Test rows: AI rows from three generators, human rows from two sources, split over test and test_ood."""
    rng = random.Random(seed)
    pool = []
    for gen in ("g/a", "g/b", "g/c"):
        for i in range(n_per_generator):
            pool.append({"id": f"ai_{gen[-1]}{i}", "text": khmer_text(rng), "label": 1, "generator": gen, "source": "wikipedia" if i % 2 else "mc4_web",
                         "prompt_type": "wiki_title", "split": "test_ood" if gen == "g/c" else "test"})
    for src in ("wikipedia", "mc4_web"):
        for i in range(n_per_source):
            pool.append({"id": f"h_{src[0]}{i}", "text": khmer_text(rng), "label": 0, "generator": "human", "source": src, "prompt_type": "",
                         "split": "test_ood" if i % 4 == 0 else "test"})
    return pool


class FakeProvider:
    """Answers with a reworded copy of the prompt's text, or whatever `reply` returns; counts the calls."""

    def __init__(self, reply=None):
        self.calls, self.reply = [], reply

    def generate(self, system, prompt, *, max_tokens, temperature):
        self.calls.append({"system": system, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature})
        text = prompt.split("<<<\n", 1)[1].split("\n>>>", 1)[0]
        if self.reply is not None:
            out = self.reply(text, len(self.calls))
            if isinstance(out, Exception):
                raise out
            return out
        return Completion(text=reword(text), finish_reason="stop")


def reword(text):
    """A 'paraphrase': the sentences in reverse order with a different letter in every word (about 20% overlap)."""
    sentences = text.replace("\n", " ").split("។")
    sentences = [s.strip() for s in sentences if s.strip()][::-1]
    table = str.maketrans(LETTERS, LETTERS[3:] + LETTERS[:3])
    return " ".join(s.translate(table) + "។" for s in sentences)


@pytest.fixture
def runner_parts(tmp_path):
    out, errors = JsonlWriter(tmp_path / "out.jsonl", unique_key="id"), JsonlWriter(tmp_path / "errors.jsonl")
    yield out, errors, threading.Event(), RetryPolicy(max_attempts=2, initial_wait=0, max_wait=0, jitter=0)
    out.close()
    errors.close()


def make_runner(provider, parts, **kw):
    out, errors, stop, policy = parts
    return paraphrase.Paraphraser(provider, "test-model", out, errors, stop, policy, interval=0, **kw)


# ---------------------------------------------------------------------------
# Prompt and output parsing
# ---------------------------------------------------------------------------
def test_build_prompt_contains_the_text_and_the_length_range():
    text = khmer_text(random.Random(3))
    prompt, target = paraphrase.build_prompt(text)
    assert target == len(text) and f"<<<\n{text}\n>>>" in prompt
    assert f"about {len(text)} characters" in prompt and f"between {int(len(text) * 0.8)} and {int(len(text) * 1.2)} characters" in prompt
    assert "Khmer (Khmer script) only" in prompt and "No preamble" in prompt
    assert "{" not in prompt.replace(text, "")  # every placeholder was filled
    braces, _ = paraphrase.build_prompt("ក{text}ខ{0}")  # braces in the data cannot break the template
    assert "ក{text}ខ{0}" in braces


def test_unwrap_strips_thinking_fences_and_markers():
    assert paraphrase.unwrap("  ក។  ") == "ក។"
    assert paraphrase.unwrap("```\nក។\n```") == "ក។"
    assert paraphrase.unwrap("```text\nក។\nខ។\n```\n") == "ក។\nខ។"
    assert paraphrase.unwrap("<<<\nក។\n>>>") == "ក។"
    assert paraphrase.unwrap("<think>hmm</think>ក។") == "ក។"
    assert paraphrase.unwrap("") == "" and paraphrase.unwrap(None) == ""


def test_ngram_overlap():
    assert paraphrase.ngram_overlap("ក" * 30, "ក" * 30) == 1.0
    assert paraphrase.ngram_overlap("ក" * 30, "ខ" * 30) == 0.0
    assert paraphrase.ngram_overlap("ab", "ab") == 0.0  # shorter than one n-gram
    half = paraphrase.ngram_overlap("កខគឃងចឆជតថ", "កខគឃងចឆជ" + "ដដដដដ")
    assert 0 < half < 1
    assert paraphrase.ngram_overlap("ក\n\nខ  គ ឃ ង ច", "ក ខ គ ឃ ង ច") == 1.0  # whitespace is flattened


def test_clean_paraphrase_accepts_a_good_rewrite_and_applies_the_dataset_rules():
    original = khmer_text(random.Random(4))
    rewrite = reword(original)
    text, reason, overlap = paraphrase.clean_paraphrase(rewrite, original)
    assert reason is None and text == clean.process_text(rewrite, is_ai=True).filtered.text and 0 <= overlap < 0.9
    messy = "```\n" + rewrite.replace(" ", "​ ", 5).replace("។", " ។", 3) + "\n```"
    text2, reason2, _ = paraphrase.clean_paraphrase(messy, original)
    assert reason2 is None and "​" not in text2 and " ។" not in text2  # invisible characters and spaces before marks are gone
    assert text2 == text


@pytest.mark.parametrize("raw, expected", [
    ("", "empty output"),
    ("   \n  ", "empty output"),
    ("ក" * 60 + "។", "too_short"),
    ("English text only. " * 40 + "។", "low_khmer_ratio"),
    ("ក" * 500, "truncated_no_sentence_end"),
])
def test_clean_paraphrase_rejects_unusable_output(raw, expected):
    original = khmer_text(random.Random(5))
    text, reason, _ = paraphrase.clean_paraphrase(raw, original)
    assert text is None and reason == expected


def test_clean_paraphrase_rejects_copies_and_wrong_lengths():
    original = khmer_text(random.Random(6), sentences=30)  # about 1,000 chars
    text, reason, overlap = paraphrase.clean_paraphrase(original, original)
    assert text is None and reason.startswith("near copy") and overlap == 1.0
    sentences = clean.process_text(reword(original), is_ai=True).filtered.text.split("។")
    short = "។".join(sentences[:15]) + "។"  # about half the length: over 300 chars, under 0.65x
    assert 300 < len(short) and len(short) / len(original) < paraphrase.MIN_LENGTH_RATIO
    assert paraphrase.clean_paraphrase(short, original)[1].startswith("length 0.")
    double = reword(original) + " " + reword(original)  # twice the length, still under 2,500 chars
    assert len(double) < clean.MAX_CHARS
    assert paraphrase.clean_paraphrase(double, original)[1].startswith("length 2.0")


def test_clean_paraphrase_trims_an_overlong_rewrite_to_a_sentence_end_instead_of_dropping_it():
    original = khmer_text(random.Random(7), sentences=90)
    original = original[: original.rfind("។", 0, 2300) + 1]  # about 2,300 chars, like the longest dataset rows
    assert 2000 < len(original) <= 2300
    rewrite = reword(original) + " " + reword(original)[:400] + "។"
    assert len(rewrite) > clean.MAX_CHARS
    text, reason, _ = paraphrase.clean_paraphrase(rewrite, original)
    assert reason is None and len(text) <= clean.MAX_CHARS and text.endswith("។")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def test_allocate_is_equal_capped_and_seeded():
    rng = lambda: random.Random(1)  # noqa: E731
    assert paraphrase.allocate({"a": 50, "b": 50, "c": 50}, 9, rng()) == {"a": 3, "b": 3, "c": 3}
    assert paraphrase.allocate({"a": 2, "b": 50, "c": 50}, 12, rng()) == {"a": 2, "b": 5, "c": 5}  # a small stratum hands its share on
    assert paraphrase.allocate({"a": 1, "b": 1}, 10, rng()) == {"a": 1, "b": 1}  # asking for more than exists
    assert paraphrase.allocate({"a": 5}, 0, rng()) == {"a": 0}
    odd = paraphrase.allocate({"a": 50, "b": 50, "c": 50}, 10, rng())
    assert sorted(odd.values()) == [3, 3, 4] and odd == paraphrase.allocate({"a": 50, "b": 50, "c": 50}, 10, rng())
    assert paraphrase.allocate({}, 5, rng()) == {}


def test_sample_rows_is_stratified_seeded_and_interleaved():
    pool = make_pool()
    sample = paraphrase.sample_rows(pool, n_ai=9, n_human=6, seed=42)
    assert len(sample) == 15 and len({r["id"] for r in sample}) == 15
    ai, human = [r for r in sample if r["label"] == 1], [r for r in sample if r["label"] == 0]
    assert Counter(r["generator"] for r in ai) == {"g/a": 3, "g/b": 3, "g/c": 3}  # equal shares per generator, test_ood included
    assert Counter(r["source"] for r in human) == {"wikipedia": 3, "mc4_web": 3}
    assert [r["label"] for r in sample[:12]] == [1, 0] * 6  # AI / human alternate while both last
    assert {r["generator"] for r in sample[:6] if r["label"] == 1} == {"g/a", "g/b", "g/c"}  # so does a --limit 6 prefix: every stratum shows up
    assert sample == paraphrase.sample_rows(list(reversed(pool)), 9, 6, 42)  # independent of the order of the pool
    assert [r["id"] for r in sample] != [r["id"] for r in paraphrase.sample_rows(pool, 9, 6, 7)]
    assert {r["split"] for r in sample} == {"test", "test_ood"}


def test_sample_rows_smaller_pools_and_zero_humans():
    pool = make_pool(n_per_generator=2, n_per_source=3)
    everything = paraphrase.sample_rows(pool, n_ai=150, n_human=150, seed=42)
    assert len(everything) == len(pool)  # asking for more than exists gives everything
    only_ai = paraphrase.sample_rows(pool, n_ai=4, n_human=0, seed=42)
    assert len(only_ai) == 4 and all(r["label"] == 1 for r in only_ai)
    assert paraphrase.sample_rows(pool, 0, 0) == []


def test_load_pool_reads_the_requested_splits(tmp_path):
    for name, n in (("test", 3), ("test_ood", 2), ("val", 5)):
        clean.write_jsonl(tmp_path / f"{name}.jsonl", [
            {"id": f"{name}{i}", "text": "ក", "label": i % 2, "source": "wikipedia", "generator": "g", "url": "", "date": "", "length": 1,
             "human_id": f"{name}{i}", "title": "", "prompt_type": ""} for i in range(n)
        ])
    pool = paraphrase.load_pool(tmp_path)
    assert Counter(r["split"] for r in pool) == {"test": 3, "test_ood": 2}
    with pytest.raises(FileNotFoundError, match="run src/split.py first"):
        paraphrase.load_pool(tmp_path, ("test", "nothing"))


# ---------------------------------------------------------------------------
# Running: resume, errors, retries
# ---------------------------------------------------------------------------
def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_run_writes_complete_records_and_resumes_without_repeating_rows(tmp_path, runner_parts):
    sample = paraphrase.sample_rows(make_pool(), 4, 4, 42)
    provider = FakeProvider()
    stats = make_runner(provider, runner_parts).run(sample[:3])
    assert (stats.done, stats.failed) == (3, 0) and len(provider.calls) == 3
    out_path = tmp_path / "out.jsonl"
    first = read(out_path)
    assert [r["id"] for r in first] == [r["id"] for r in sample[:3]]
    r0, s0 = first[0], sample[0]
    assert set(r0) == {"id", "split", "label", "generator", "source", "prompt_type", "orig_length", "length", "text", "model", "date",
                       "finish_reason", "ngram_overlap"}
    assert (r0["id"], r0["label"], r0["generator"], r0["split"], r0["model"]) == (s0["id"], s0["label"], s0["generator"], s0["split"], "test-model")
    assert r0["orig_length"] == len(s0["text"]) and r0["length"] == len(r0["text"]) and r0["finish_reason"] == "stop"
    assert r0["text"] != s0["text"] and "ក" in out_path.read_text(encoding="utf-8")  # UTF-8 on disk, not \u escapes
    assert provider.calls[0]["system"] == paraphrase.SYSTEM_PROMPT and provider.calls[0]["temperature"] == paraphrase.TEMPERATURE
    assert provider.calls[0]["max_tokens"] <= paraphrase.MAX_OUTPUT_TOKENS

    # a second run over the whole sample: the first three are skipped
    out, errors, stop, policy = runner_parts
    assert paraphrase.done_ids(out_path) == {r["id"] for r in sample[:3]}
    pending = [r for r in sample if r["id"] not in paraphrase.done_ids(out_path)]
    assert len(pending) == 5
    provider2 = FakeProvider()
    stats2 = make_runner(provider2, runner_parts).run(pending)
    assert stats2.done == 5 and len(provider2.calls) == 5
    assert [r["id"] for r in read(out_path)] == [r["id"] for r in sample]  # every row exactly once, in sample order
    # the writer also refuses a duplicate id on its own
    assert make_runner(FakeProvider(), runner_parts).run(sample[:1]).done == 0


def test_a_failed_row_goes_to_the_errors_file_and_is_retried_next_time(tmp_path, runner_parts):
    sample = paraphrase.sample_rows(make_pool(), 2, 2, 42)
    bad_id = sample[1]["id"]
    provider = FakeProvider(reply=lambda text, n: RuntimeError("boom") if n == 2 else Completion(text=reword(text), finish_reason="stop"))
    stats = make_runner(provider, runner_parts).run(sample)
    assert (stats.done, stats.failed, stats.stopped) == (3, 1, "")  # one failure does not end the run
    assert [r["id"] for r in read(tmp_path / "out.jsonl")] == [r["id"] for r in sample if r["id"] != bad_id]
    errors = read(tmp_path / "errors.jsonl")
    assert len(errors) == 1 and errors[0]["id"] == bad_id and errors[0]["error_type"] == "other" and "boom" in errors[0]["message"]
    assert set(errors[0]) == {"id", "label", "model", "error_type", "status", "message", "date"}
    pending = [r for r in sample if r["id"] not in paraphrase.done_ids(tmp_path / "out.jsonl")]
    assert [r["id"] for r in pending] == [bad_id]  # only the failed row is left to do


def test_bad_output_is_retried_once_then_recorded(tmp_path, runner_parts):
    sample = paraphrase.sample_rows(make_pool(), 1, 1, 42)
    answers = {}

    def reply(text, n):
        answers[n] = text
        if n in (1, 3, 4):  # first row: empty, then fine; second row: a copy, twice
            return Completion(text="" if n == 1 else text, finish_reason="stop")
        return Completion(text=reword(text), finish_reason="stop")

    provider = FakeProvider(reply=reply)
    stats = make_runner(provider, runner_parts).run(sample)
    assert (stats.done, stats.failed) == (1, 1) and len(provider.calls) == 4
    out = read(tmp_path / "out.jsonl")
    errors = read(tmp_path / "errors.jsonl")
    assert [r["id"] for r in out] == [sample[0]["id"]]
    assert errors[0]["id"] == sample[1]["id"] and errors[0]["error_type"] == "bad_output" and "near copy" in errors[0]["message"]


def test_transient_errors_are_retried_with_backoff(tmp_path, runner_parts):
    class Flaky(Exception):
        status_code = 503

    sample = paraphrase.sample_rows(make_pool(), 1, 0, 42)
    provider = FakeProvider(reply=lambda text, n: Flaky("unavailable") if n == 1 else Completion(text=reword(text), finish_reason="stop"))
    stats = make_runner(provider, runner_parts).run(sample)
    assert (stats.done, stats.failed) == (1, 0) and len(provider.calls) == 2  # 503 once, then success on the second attempt


def test_a_daily_quota_stops_the_run_and_auth_errors_too(tmp_path, runner_parts):
    class Quota(Exception):
        status_code = 429

    sample = paraphrase.sample_rows(make_pool(), 3, 3, 42)
    provider = FakeProvider(reply=lambda text, n: Quota("Quota exceeded: RequestsPerDay limit: 0"))
    stats = make_runner(provider, runner_parts).run(sample)
    assert len(provider.calls) == 1 and stats.failed == 1 and "daily quota" in stats.stopped  # no point trying the other rows
    assert read(tmp_path / "errors.jsonl")[0]["error_type"] == "rate_limit"

    class Auth(Exception):
        status_code = 403

    provider2 = FakeProvider(reply=lambda text, n: Auth("API key not valid"))
    stats2 = make_runner(provider2, runner_parts).run(sample)
    assert len(provider2.calls) == 1 and "check the key" in stats2.stopped


def test_the_run_gives_up_after_too_many_failures_in_a_row(runner_parts):
    sample = paraphrase.sample_rows(make_pool(), 8, 8, 42)
    policy = RetryPolicy(max_attempts=1, initial_wait=0, max_wait=0, jitter=0, max_consecutive_failures=3)
    out, errors, stop, _ = runner_parts
    provider = FakeProvider(reply=lambda text, n: RuntimeError("down"))
    stats = paraphrase.Paraphraser(provider, "m", out, errors, stop, policy, interval=0).run(sample)
    assert len(provider.calls) == 3 and stats.failed == 3 and "3 failures in a row" in stats.stopped


def test_a_stop_request_ends_the_run_before_the_next_row(runner_parts):
    sample = paraphrase.sample_rows(make_pool(), 3, 3, 42)
    out, errors, stop, policy = runner_parts

    def reply(text, n):
        stop.set()  # Ctrl+C arrives while the first request is in flight
        return Completion(text=reword(text), finish_reason="stop")

    provider = FakeProvider(reply=reply)
    stats = paraphrase.Paraphraser(provider, "m", out, errors, stop, policy, interval=0).run(sample)
    assert len(provider.calls) == 1 and stats.done == 1  # the finished row is kept, the rest is left for the next run


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
@pytest.fixture
def splits_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paraphrase, "load_dotenv", lambda *a, **k: None)  # tests must never pick up the real API key
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    folder = tmp_path / "splits"
    pool = make_pool()
    for name in ("test", "test_ood"):
        clean.write_jsonl(folder / f"{name}.jsonl", [
            {"id": r["id"], "text": r["text"], "label": r["label"], "source": r["source"], "generator": r["generator"], "url": "", "date": "",
             "length": len(r["text"]), "human_id": r["id"], "title": "", "prompt_type": r["prompt_type"]} for r in pool if r["split"] == name
        ])
    return folder


def test_dry_run_prints_the_plan_and_prompts_and_needs_no_key(splits_dir, tmp_path, capsys):
    out = tmp_path / "eval" / "p.jsonl"
    code = paraphrase.main(["--dry-run", "--splits-dir", str(splits_dir), "--out", str(out), "--n-ai", "6", "--n-human", "4", "--limit", "5"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "Sample: 5 rows (after --limit)" in printed and "SYSTEM PROMPT" in printed
    assert "EXAMPLE USER PROMPT: ai row" in printed and "EXAMPLE USER PROMPT: human row" in printed and "<<<" in printed
    assert not out.exists() and not out.parent.exists()  # a dry run writes nothing


def test_main_runs_with_a_fake_provider_and_resumes(splits_dir, tmp_path, monkeypatch, capsys):
    provider = FakeProvider()
    monkeypatch.setattr(paraphrase, "make_provider", lambda spec: provider)
    out = tmp_path / "eval" / "p.jsonl"
    args = ["--splits-dir", str(splits_dir), "--out", str(out), "--n-ai", "4", "--n-human", "4", "--interval", "0"]
    assert paraphrase.main(args + ["--limit", "3"]) == 0
    assert len(read(out)) == 3 and len(provider.calls) == 3
    assert (tmp_path / "eval" / "p_errors.jsonl").exists()  # the sidecar sits next to the output
    capsys.readouterr()
    assert paraphrase.main(args) == 0  # the full sample: only the 5 missing rows are requested
    assert len(read(out)) == 8 and len(provider.calls) == 8
    printed = capsys.readouterr().out
    assert "3 already paraphrased" in printed and "done 5, failed 0" in printed
    assert paraphrase.main(args) == 0 and len(provider.calls) == 8
    assert "Nothing to do" in capsys.readouterr().out
    assert len({r["id"] for r in read(out)}) == 8


def test_main_without_a_key_or_splits_fails_cleanly(splits_dir, tmp_path, capsys):
    assert paraphrase.main(["--splits-dir", str(splits_dir), "--out", str(tmp_path / "p.jsonl"), "--limit", "2"]) == 2
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err
    assert not (tmp_path / "p.jsonl").exists()
    assert paraphrase.main(["--splits-dir", str(tmp_path / "nothing"), "--out", str(tmp_path / "p.jsonl")]) == 2
    assert "run src/split.py first" in capsys.readouterr().err


def test_defaults_follow_the_brief():
    args = paraphrase.parse_args([])
    assert (args.n_ai, args.n_human, args.seed, args.model, args.splits) == (150, 150, 42, "gemini-3.1-flash-lite", ["test", "test_ood"])
    assert args.out == ROOT / "data" / "eval" / "paraphrased.jsonl" and args.limit is None and not args.dry_run
