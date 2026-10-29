"""Pack the code, the step-4 splits and (if it exists) the step-7 paraphrases into one zip for Colab (notebooks/02_finetune_xlmr.ipynb).

The GitHub repo cannot be cloned from Colab while it is private, and the splits are gitignored, so the
notebook can instead unzip this bundle from Google Drive. Paths inside the zip are repo-relative
(src/..., data/processed/splits/..., data/eval/paraphrased.jsonl), so the unzipped folder works like a clone and
`python src/finetune.py` and `python src/evaluate.py --model xlmr` both run in it. A MANIFEST.json records the commit and
the row counts, so a stale bundle is easy to spot.

Input:  src/*.py, data/processed/splits/{train,val,test,test_ood}.jsonl, data/eval/paraphrased.jsonl (optional, src/paraphrase.py)
Output: colab_bundle.zip in the repo root (gitignored); drag it into the root of "My Drive"

Usage:
    python src/make_colab_bundle.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
EVAL_DIR = ROOT / "data" / "eval"
OUT_PATH = ROOT / "colab_bundle.zip"
SPLIT_NAMES = ("train", "val", "test", "test_ood")
OPTIONAL_EVAL_FILES = ("paraphrased.jsonl",)  # step 7 robustness input; left out of the bundle if it has not been made yet


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+uncommitted src changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def count_rows(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> int:
    splits = [SPLITS_DIR / f"{name}.jsonl" for name in SPLIT_NAMES]
    missing = [p.name for p in splits if not p.exists()]
    if missing:
        print(f"error: missing in {SPLITS_DIR}: {', '.join(missing)} (run src/split.py first)", file=sys.stderr)
        return 2
    code = sorted((ROOT / "src").glob("*.py"))
    extras = [p for p in (EVAL_DIR / name for name in OPTIONAL_EVAL_FILES) if p.exists()]
    manifest = {
        "commit": git_commit(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {p.stem: count_rows(p) for p in splits + extras},
        "code": [p.name for p in code],
        "eval_files": [p.name for p in extras],
    }
    with zipfile.ZipFile(OUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in code + splits + extras:
            zf.write(path, path.relative_to(ROOT).as_posix())
        zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB): {len(code)} code files, rows {manifest['rows']}, commit {manifest['commit']}")
    if not extras:
        print(f"Note: no {', '.join(OPTIONAL_EVAL_FILES)} in {EVAL_DIR}; evaluate.py will skip the paraphrase test (make it with src/paraphrase.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
