"""Tests for scripts/collect.sh's health record (T4.6) — offline & deterministic.

Per the DEVPLAN todo: a stubbed run writes a well-formed `data/last_run.json` (ok + error
cases). Runs the *real* `scripts/collect.sh` end to end (not a re-implementation of its
logic) so a future edit to the script can't silently break the health record without
failing here.

`collect.sh` always operates on `$(dirname "$0")/..` -- whatever directory contains the
copy of the script being run, not a path parameterized by an env var -- so this test builds
a self-contained scratch "repo" (a copy of `src/`, `scripts/collect.sh`, `scripts/notify.sh`,
`scripts/triage.sh`, with `src/collector.py` replaced by a controllable stub) and runs the
script there instead of against this actual checkout. That sidesteps three real side effects
running the genuine `scripts/collect.sh` in place would otherwise have:
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
- The scratch repo is deliberately not a git repository. `scripts/triage.sh` (called on a
  failed run) checks `command -v claude`/`command -v gh` before ever touching the network,
  but both happen to be installed system-wide on this machine (`/usr/bin/claude`,
  `/usr/bin/gh`) as well as user-locally, so excluding one directory from `PATH` can't be
  relied on to make those checks fail. Not being a git repo is what actually keeps this
  test offline: triage.sh's own `git diff --quiet` guard fails there, so it bails out via
  its "working tree dirty (another session may be active)" branch -- before ever reaching
  `gh pr list` -- regardless of what's on `PATH`.

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


def _build_scratch_repo(tmp_path: Path) -> Path:
    """A minimal copy of this repo collect.sh actually needs: `src/` (with `collector.py`
    replaced by a stub reading its behavior from env vars) and the three `scripts/*.sh`
    collect.sh itself invokes."""
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "src", repo / "src")
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


def _run_collect_sh_stubbed(
    tmp_path: Path, *, collector_exit_code: int, collector_output: str
) -> tuple[subprocess.CompletedProcess, Path]:
    repo = _build_scratch_repo(tmp_path)

    env = os.environ.copy()
    # `sys.executable`'s own dir first, so the counts/JSON-encoding snippets collect.sh runs
    # get a real interpreter with this repo's dependencies installed (no `.venv/bin/activate`
    # exists in the scratch repo to do this automatically -- see module docstring).
    env["PATH"] = f"{Path(sys.executable).parent}:/usr/bin:/bin"
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
    result, repo = _run_collect_sh_stubbed(
        tmp_path, collector_exit_code=1, collector_output="rate limited by GitHub\n"
    )

    assert result.returncode == 1
    record = json.loads((repo / "data" / "last_run.json").read_text())
    assert record["status"] == "error"
    assert record["exit_code"] == 1
    assert "rate limited by GitHub" in record["error_tail"]
    assert (repo / record["log"]).exists()


def test_collect_sh_triage_call_does_not_hang_or_touch_the_network_on_failure(
    tmp_path: Path,
) -> None:
    """The error-path test above already exercises this implicitly (it would time out if
    triage.sh's own network call weren't neutralized) -- this test names that guarantee
    explicitly rather than leaving it as an accident of the 30s subprocess timeout.

    The scratch repo built by `_build_scratch_repo` is deliberately not a git repository, so
    `triage.sh`'s own `git diff --quiet` guard fails and it bails out via its "working tree
    dirty (another session may be active)" branch -- before ever reaching `gh pr list`,
    regardless of whether `claude`/`gh` themselves happen to be on `PATH` (both are, on this
    machine, at `/usr/bin/claude`/`/usr/bin/gh` as well as their user-local installs, so
    excluding a single directory from `PATH` can't be relied on for this). Every one of
    triage.sh's three bail-out branches (`claude` missing, `gh` missing, dirty tree) ends in
    the same "— alert only" suffix, so asserting on that -- rather than on which specific
    branch fired -- is what actually matters here: no branch reaches the network."""
    result, _repo = _run_collect_sh_stubbed(
        tmp_path, collector_exit_code=1, collector_output="boom"
    )

    assert "— alert only" in result.stdout or "— alert only" in result.stderr
