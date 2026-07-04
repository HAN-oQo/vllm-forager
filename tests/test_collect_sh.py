"""Regression test for scripts/collect.sh's health-record / ntfy-notification record count.

collect.sh used to count records with a hardcoded `glob.glob("data/*.jsonl")`, which silently
reported 0 records (in data/last_run.json AND the "collect done: ok · N records" ntfy push)
whenever FORAGER_DATA_DIR points the real dataset outside ./data — exactly the shared-data-dir
setup this repo uses. The fix delegates to src.stats.summarize(), which resolves
config.DATA_DIR (FORAGER_DATA_DIR-aware). This test runs the *actual* heredoc embedded in
collect.sh (rather than re-implementing the counting logic) so a future edit to the script
can't silently reintroduce the hardcoded path without failing here.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.m0

ROOT = Path(__file__).resolve().parent.parent


def _extract_counts_snippet() -> str:
    """Pull the python heredoc collect.sh uses to build the health-record `counts` JSON."""
    text = (ROOT / "scripts" / "collect.sh").read_text()
    match = re.search(r"python - <<'PY'\n(.*?)\nPY\n", text, re.DOTALL)
    assert match, "collect.sh's counts heredoc not found — did its structure change?"
    return match.group(1)


def test_collect_sh_counts_respect_forager_data_dir(tmp_path, monkeypatch):
    """The embedded counting snippet must count FORAGER_DATA_DIR, not a hardcoded ./data."""
    (tmp_path / "o__r.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"number": 1, "type": "issue"}),
                json.dumps({"number": 2, "type": "pr"}),
            ]
        )
        + "\n"
    )
    monkeypatch.setenv("FORAGER_DATA_DIR", str(tmp_path))

    result = subprocess.run(
        [sys.executable, "-"],
        input=_extract_counts_snippet(),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    counts = json.loads(result.stdout)
    assert counts["_total"] == 2  # NOT 0 — the bug's symptom when DATA_DIR != ./data
    assert counts["o__r.jsonl"] == 2
