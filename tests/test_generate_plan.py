"""Tests for the frozen generation plan in src/generate_ai.py (no API calls: only --dry-run is used)."""
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import generate_ai as G  # noqa: E402

THREE = "gemini:gemini-3.1-flash-lite,gemini:gemma-4-31b-it,groq:openai/gpt-oss-120b"


def human_df(n=12, offset=0):
    rows = []
    for i in range(n):
        text = " ".join(f"ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា ទី{i + offset} ជាច្រើនដង។" for _ in range(25))
        rows.append({
            "id": f"h{i:03d}", "text": text, "title": f"ប្រធានបទ {i}" if i % 2 else "", "source": "wikipedia" if i % 2 else "mc4_web",
            "url": f"http://example.com/{i}", "date": "", "license": "", "generator": "human", "label": "0",
        })
    return pd.DataFrame(rows)


def make_tasks(n=12):
    return G.build_assignment(human_df(n), G.parse_generators(THREE), seed=42)


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr(G, "load_dotenv", lambda *a, **k: None)  # tests must never pick up real API keys


def run_dry(capsys, *args):
    code = G.main(["--dry-run", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_plan_round_trip_is_exact_and_atomic(tmp_path):
    tasks = make_tasks()
    path = tmp_path / "gen" / "plan.jsonl"
    G.write_plan(path, tasks)
    assert G.load_plan(path) == tasks  # prompts, Khmer text and positions all survive
    assert [p.name for p in path.parent.iterdir()] == ["plan.jsonl"]  # no temp file left behind
    assert "ខ្ញុំ" in path.read_text(encoding="utf-8")  # stored as UTF-8, not \u escapes
    G.write_plan(path, tasks[:3])  # writing again replaces the file
    assert len(G.load_plan(path)) == 3


def test_load_plan_is_strict(tmp_path):
    tasks = make_tasks(4)
    path = tmp_path / "plan.jsonl"
    G.write_plan(path, tasks)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1] + [lines[-1][:40]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid plan row"):
        G.load_plan(path)  # a cut-off line is an error, never silently dropped
    path.write_text("\n".join(lines[:2] + [lines[0]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repeated human_id"):
        G.load_plan(path)
    path.write_text(lines[0].replace('"prompt":', '"prompt_x":') + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="KeyError"):
        G.load_plan(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        G.load_plan(path)


def test_generator_mismatch_is_an_error(tmp_path, capsys):
    plan = tmp_path / "plan.jsonl"
    G.write_plan(plan, make_tasks())
    code, out, err = run_dry(capsys, "--plan", str(plan), "--out", str(tmp_path / "ai.jsonl"),
                             "--generators", "gemini:gemini-3.1-flash-lite,groq:openai/gpt-oss-120b")
    assert code == 2 and "frozen plan is for generators" in err and "--rebuild-plan" in err
    assert "Using frozen plan" not in out
    code, out, err = run_dry(capsys, "--plan", str(plan), "--out", str(tmp_path / "ai.jsonl"), "--generators", THREE)
    assert code == 0 and err == ""  # the same generators in another order are fine


def test_loaded_plan_ignores_human_csv(tmp_path, capsys):
    plan = tmp_path / "plan.jsonl"
    tasks = make_tasks()
    G.write_plan(plan, tasks)
    common = ["--plan", str(plan), "--out", str(tmp_path / "ai.jsonl"), "--generators", THREE]
    results = []
    for csv in (None, tmp_path / "missing.csv", tmp_path / "other.csv"):
        if csv is not None and csv.name == "other.csv":
            human_df(5, offset=100).to_csv(csv, index=False, encoding="utf-8")  # a different, shorter human.csv
        extra = [] if csv is None else ["--human", str(csv)]
        code, out, err = run_dry(capsys, *common, *extra)
        assert code == 0 and err == ""
        assert f"Using frozen plan {plan}: 12 rows" in out
        assert "Human passages:" not in out  # human.csv was never read
        results.append(out)
    assert results[0] == results[1] == results[2]
    first_web = next(t for t in tasks if t.prompt_type == "web_hint")
    assert first_web.prompt in results[0]  # the example prompt comes from the stored plan


def test_plan_is_used_with_limit_and_done_counts(tmp_path, capsys):
    plan = tmp_path / "plan.jsonl"
    tasks = make_tasks()
    G.write_plan(plan, tasks)
    ai = tmp_path / "ai.jsonl"
    ai.write_text('{"human_id": "%s"}\n' % tasks[0].human_id, encoding="utf-8")
    code, out, _ = run_dry(capsys, "--plan", str(plan), "--out", str(ai), "--generators", THREE, "--limit", "6")
    assert code == 0
    assert "Assignment: 6 human rows (after --limit)" in out
    rows = [line for line in out.splitlines() if line.startswith(("gemini:", "groq:"))]
    assert len(rows) == 3 and sum(int(r.split()[1]) for r in rows) == 6
    assert sum(int(r.split()[-2]) for r in rows) == 1  # the one saved text counts as done


def test_dry_run_builds_but_does_not_write_a_plan(tmp_path, capsys):
    csv, plan = tmp_path / "human.csv", tmp_path / "plan.jsonl"
    human_df().to_csv(csv, index=False, encoding="utf-8")
    code, out, _ = run_dry(capsys, "--human", str(csv), "--plan", str(plan), "--out", str(tmp_path / "ai.jsonl"), "--generators", THREE)
    assert code == 0 and "Human passages: 12 loaded" in out and "not written" in out
    assert not plan.exists()
    G.write_plan(plan, G.build_assignment(human_df(), G.parse_generators(THREE), 42))
    # --rebuild-plan ignores an existing plan and reads human.csv again (still no write in a dry run)
    other_csv = tmp_path / "small.csv"
    human_df(6).to_csv(other_csv, index=False, encoding="utf-8")
    before = plan.read_text(encoding="utf-8")
    code, out, _ = run_dry(capsys, "--human", str(other_csv), "--plan", str(plan), "--rebuild-plan",
                           "--out", str(tmp_path / "ai.jsonl"), "--generators", THREE)
    assert code == 0 and "Human passages: 6 loaded" in out and "Using frozen plan" not in out
    assert plan.read_text(encoding="utf-8") == before


def test_task_fields_match_the_plan_format():
    names = {f.name for f in fields(G.Task)}
    assert names == {"position", "human_id", "source", "title", "url", "prompt_type", "prompt", "target_chars", "generator"}
