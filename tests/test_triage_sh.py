"""Tests for scripts/triage.sh's skip-guards (T4.7) — offline & deterministic.

Per the DEVPLAN todo: the full self-heal flow (asking `claude -p` to diagnose and fix a real
collector failure) is integration-only -- it needs live `claude`+`gh` credentials and an
actual LLM call, so it isn't exercised here. What's shell-checkable, and what this file
tests, are the four early bail-out guards `triage.sh` runs *before* ever calling `claude -p`
or touching a real GitHub API: `claude` missing, `gh` missing, a dirty working tree, and an
already-open `triage/*` PR. Every guard exits 0 ("alert only"/"skipping") without invoking
`claude -p` -- the tests confirm this by planting a stub `claude` that records whether it was
ever actually called, and asserting it wasn't, rather than only checking the exit code/message
(which alone wouldn't catch a guard silently falling through to a real, expensive call before
still reporting a similar-looking message).

`triage.sh` always operates on `$(dirname "$0")/..`, so each test builds a scratch git
repository (a REAL one this time -- unlike `tests/test_collect_health.py`'s deliberately
non-git scratch repo, the dirty-tree and open-PR guards need actual git state to exercise) and
copies only `scripts/triage.sh` into it. `PATH` is built from scratch per test (a curated
directory with real `git`/`tail` symlinks, plus stub `claude`/`gh` executables added or
omitted per guard under test) rather than filtering the ambient `PATH`, since `claude`/`gh`
are installed system-wide on this machine as well as user-locally (see
`tests/test_collect_health.py`'s own docstring) -- excluding one directory can't reliably make
`command -v` fail for either.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.m4

ROOT = Path(__file__).resolve().parent.parent

_CLAUDE_CALLED_MARKER = "TRIAGE_TEST_CLAUDE_CALLED_MARKER"


def _build_scratch_git_repo(tmp_path: Path, *, dirty: bool) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "triage.sh", repo / "scripts" / "triage.sh")
    (repo / "scripts" / "triage.sh").chmod(0o755)

    # -b main: don't rely on the git installation's own configured default branch name --
    # triage.sh itself does `git checkout main` once its guards pass (see
    # test_triage_sh_proceeds_to_claude_once_every_guard_passes), so the scratch repo's
    # actual default branch must match that literal name for a realistic test.
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("scratch repo for triage.sh tests\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    if dirty:
        (repo / "README.md").write_text("uncommitted change\n")
    return repo


def _build_path(
    tmp_path: Path, *, with_claude: bool, with_gh: bool, gh_open_triage_count: int = 0
) -> str:
    """A curated PATH: real `git`/`tail` (triage.sh needs both), plus stub `claude`/`gh`
    added only when requested -- never the ambient PATH, since the real `claude`/`gh` are
    reachable there regardless of which directory is excluded."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for real in ("bash", "dirname", "git", "tail"):
        (bin_dir / real).symlink_to(shutil.which(real))

    if with_claude:
        claude = bin_dir / "claude"
        # Records that it was actually invoked -- the tests assert this file is absent,
        # not just that triage.sh printed a particular message (see module docstring).
        claude.write_text(
            "#!/usr/bin/env bash\n"
            # bash builtin redirection -- no external `touch` needed, keeping the curated
            # PATH minimal.
            f'> "$TRIAGE_TEST_TMP/{_CLAUDE_CALLED_MARKER}"\n'
            'echo "not a code bug"\n'
        )
        claude.chmod(0o755)

    if with_gh:
        gh = bin_dir / "gh"
        # triage.sh's real invocation is `gh pr list ... --jq '[...] | length'` -- it wants
        # the FILTERED COUNT on stdout, not the raw PR list -- so the stub prints the count
        # directly rather than trying to replicate jq's filtering.
        gh.write_text(f"#!/usr/bin/env bash\necho '{gh_open_triage_count}'\n")
        gh.chmod(0o755)

    return str(bin_dir)


def _run_triage_sh(repo: Path, path: str, tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PATH"] = path
    env["TRIAGE_TEST_TMP"] = str(tmp_path)
    return subprocess.run(
        ["bash", "scripts/triage.sh", "some.log"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_triage_sh_skips_when_claude_not_on_path(tmp_path: Path) -> None:
    repo = _build_scratch_git_repo(tmp_path, dirty=False)
    path = _build_path(tmp_path, with_claude=False, with_gh=True)

    result = _run_triage_sh(repo, path, tmp_path)

    assert result.returncode == 0
    assert "claude not on PATH" in result.stderr
    assert not (tmp_path / _CLAUDE_CALLED_MARKER).exists()


def test_triage_sh_skips_when_gh_not_on_path(tmp_path: Path) -> None:
    repo = _build_scratch_git_repo(tmp_path, dirty=False)
    path = _build_path(tmp_path, with_claude=True, with_gh=False)

    result = _run_triage_sh(repo, path, tmp_path)

    assert result.returncode == 0
    assert "gh not on PATH" in result.stderr
    assert not (tmp_path / _CLAUDE_CALLED_MARKER).exists()


def test_triage_sh_skips_when_working_tree_is_dirty(tmp_path: Path) -> None:
    repo = _build_scratch_git_repo(tmp_path, dirty=True)
    path = _build_path(tmp_path, with_claude=True, with_gh=True)

    result = _run_triage_sh(repo, path, tmp_path)

    assert result.returncode == 0
    assert "working tree dirty" in result.stderr
    assert not (tmp_path / _CLAUDE_CALLED_MARKER).exists()


def test_triage_sh_skips_when_an_open_triage_pr_already_exists(tmp_path: Path) -> None:
    repo = _build_scratch_git_repo(tmp_path, dirty=False)
    path = _build_path(tmp_path, with_claude=True, with_gh=True, gh_open_triage_count=1)

    result = _run_triage_sh(repo, path, tmp_path)

    assert result.returncode == 0
    assert "an open triage/* PR already exists" in result.stderr
    assert not (tmp_path / _CLAUDE_CALLED_MARKER).exists()


def test_triage_sh_proceeds_to_claude_once_every_guard_passes(tmp_path: Path) -> None:
    """Confirms the stubs themselves are correctly wired -- with a clean tree, no open
    triage PR, and both tools present, triage.sh actually reaches (and this stub
    intercepts) the `claude -p` call, proving the four guard tests above are testing real
    early-exit behavior and not just a stub that never gets reached at all."""
    repo = _build_scratch_git_repo(tmp_path, dirty=False)
    path = _build_path(tmp_path, with_claude=True, with_gh=True, gh_open_triage_count=0)

    result = _run_triage_sh(repo, path, tmp_path)

    assert result.returncode == 0
    assert (tmp_path / _CLAUDE_CALLED_MARKER).exists()
