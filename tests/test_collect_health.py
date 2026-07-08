"""Tests for scripts/collect.sh's health record (T4.6) — offline & deterministic.

Per the DEVPLAN todo: a stubbed run writes a well-formed `data/last_run.json` (ok + error
cases). Runs the *real* `scripts/collect.sh` end to end (not a re-implementation of its
logic) so a future edit to the script can't silently break the health record without
failing here.

`collect.sh` always operates on `$(dirname "$0")/..` -- whatever directory contains the
copy of the script being run, not a path parameterized by an env var -- so this test builds
a self-contained scratch "repo" and runs the script there instead of against this actual
checkout. The scratch repo copies only what `collect.sh` actually needs beyond the collector
call itself: `src/__init__.py`/`config.py`/`stats.py` (the only modules its own post-collector
logic imports -- `from src.stats import summarize`, which only imports `config`) plus
`scripts/collect.sh`/`notify.sh`/`triage.sh`, with `src/collector.py` replaced by a
controllable stub. Deliberately *not* a full copy of `src/` (~30 unrelated modules --
`gate.py`, `engineer.py`, `pr_author.py`, etc.) that `collect.sh` never touches, so this test
can't start failing (or paying import-time cost) for reasons that have nothing to do with the
health record it's actually checking.

That sidesteps four real side effects running the genuine `scripts/collect.sh` in place would
otherwise have:
- The real `python -m src.collector` hits live GitHub -- replaced by a stub `main()` reading
  its exit code/output from env vars this test controls.
- `collect.sh` writes `data/last_run.json` and `data/logs/collect-<ts>.log` at fixed,
  repo-root-relative paths (not parameterized by `FORAGER_DATA_DIR` like the item counts
  are) -- in the scratch repo, not this real checkout, so nothing here clobbers the
  developer's own last real collector run's health record.
- No `.venv/` exists in the scratch repo, so `collect.sh`'s own
  `[ -f .venv/bin/activate ] && source .venv/bin/activate` line is a no-op there -- avoiding
  the real venv's `activate` script re-prepending the *real* `python` onto `PATH` ahead of
  anything this test could otherwise put there.
- `scripts/triage.sh` (called on a failed run) checks `command -v claude`/`command -v gh`,
  then a clean-working-tree guard, before ever calling `gh pr list` -- all three bail-out
  branches print "... — alert only" and `exit 0` before touching the network. The scratch
  repo is deliberately not a git repository, so the clean-tree guard fails there regardless of
  PATH (both `claude`/`gh` happen to be installed system-wide on this machine as well as
  user-locally, so excluding one PATH directory can't be relied on for the earlier two
  branches). As defense in depth against a future reordering of triage.sh's own checks, a
  stub `gh` is also placed on `PATH` -- one that always reports "an open triage/* PR already
  exists" -- so even if the network call were ever reached first, it would hit the stub, not
  the real GitHub API.

One more real side effect, neutralized without needing the scratch-repo trick:
- `scripts/notify.sh` reads a real `NOTIFY_URL` from this repo's own `.env` whenever the
  `NOTIFY_URL` env var resolves to empty -- pointing it at an unused local port instead
  (rather than clearing it) keeps that env var genuinely non-empty, so the `.env` fallback
  never triggers, and the failed `curl` (connection refused, instant) is swallowed by
  `notify.sh`'s own `|| true`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.m4

ROOT = Path(__file__).resolve().parent.parent

# `src/stats.py::summarize()` (what collect.sh's own heredoc calls) only imports `config` --
# these three files are the entire real dependency surface collect.sh's post-collector logic
# needs; `collector.py` is replaced by a stub, not copied.
_SRC_FILES_COLLECT_SH_NEEDS = ("__init__.py", "config.py", "stats.py")


def _build_scratch_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    for name in _SRC_FILES_COLLECT_SH_NEEDS:
        shutil.copy2(ROOT / "src" / name, repo / "src" / name)
    (repo / "scripts").mkdir()
    for name in ("collect.sh", "notify.sh", "triage.sh"):
        shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
        (repo / "scripts" / name).chmod(0o755)

    (repo / "src" / "collector.py").write_text(
        "import os, sys\n"
        "def main(argv=None):\n"
        '    sys.stdout.write(os.environ.get("COLLECT_HEALTH_TEST_STUB_OUTPUT", ""))\n'
        '    sys.exit(int(os.environ.get("COLLECT_HEALTH_TEST_STUB_EXIT_CODE", "0")))\n'
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
    return repo


def _build_gh_stub(tmp_path: Path) -> Path:
    """Defense in depth (see module docstring): even if a future `triage.sh` reordering ever
    reached `gh pr list` before its clean-tree guard, this stub -- placed ahead of the real
    `gh` on `PATH` -- reports an open triage PR so `triage.sh` skips rather than proceeding to
    `claude -p`, without ever touching the real GitHub API."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text('#!/usr/bin/env bash\necho \'[{"headRefName": "triage/stub"}]\'\n')
    gh.chmod(0o755)
    return bin_dir


def _run_collect_sh_stubbed(
    tmp_path: Path, *, collector_exit_code: int, collector_output: str
) -> tuple[subprocess.CompletedProcess, Path]:
    repo = _build_scratch_repo(tmp_path)
    gh_stub_dir = _build_gh_stub(tmp_path)

    env = os.environ.copy()
    # Prepend, don't replace: the `gh` stub needs to come first (so it's found ahead of the
    # real `gh`), and `sys.executable`'s own dir next (guarantees a real interpreter with this
    # repo's dependencies installed, since no `.venv/bin/activate` exists in the scratch repo
    # to do that automatically -- see module docstring) -- but `bash`/`git`/`curl` etc. must
    # still resolve via whatever the real ambient `PATH` already provides, which varies by
    # platform (not guaranteed to be under `/usr/bin`/`/bin` everywhere).
    env["PATH"] = f"{gh_stub_dir}:{Path(sys.executable).parent}:{env.get('PATH', '')}"
    env["COLLECT_HEALTH_TEST_STUB_EXIT_CODE"] = str(collector_exit_code)
    env["COLLECT_HEALTH_TEST_STUB_OUTPUT"] = collector_output
    env["NOTIFY_URL"] = "http://127.0.0.1:1/unused"  # non-empty but never reachable
    env.pop(
        "FORAGER_DATA_DIR", None
    )  # count the scratch repo's own (empty) data/, not the real one

    result = subprocess.run(
        ["bash", "scripts/collect.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, repo


def test_collect_sh_writes_a_well_formed_last_run_json_on_success(tmp_path: Path) -> None:
    result, repo = _run_collect_sh_stubbed(tmp_path, collector_exit_code=0, collector_output="")

    assert result.returncode == 0, result.stderr
    record = json.loads((repo / "data" / "last_run.json").read_text())
    assert record["status"] == "ok"
    assert record["exit_code"] == 0
    assert record["started_at"] and record["finished_at"]
    assert isinstance(record["total_records"], int)
    assert isinstance(record["counts"], dict)
    assert record["error_tail"] == ""
    assert (repo / record["log"]).exists()


def test_collect_sh_writes_a_well_formed_last_run_json_on_failure(tmp_path: Path) -> None:
    """Also exercises (without a separate, dedicated test) that a failed run's self-heal
    `triage.sh` call doesn't hang or reach a live service -- see the module docstring's
    network-safety notes. If it did, this test would time out rather than complete in well
    under a second."""
    result, repo = _run_collect_sh_stubbed(
        tmp_path, collector_exit_code=1, collector_output="rate limited by GitHub\n"
    )

    assert result.returncode == 1
    record = json.loads((repo / "data" / "last_run.json").read_text())
    assert record["status"] == "error"
    assert record["exit_code"] == 1
    assert "rate limited by GitHub" in record["error_tail"]
    assert (repo / record["log"]).exists()
