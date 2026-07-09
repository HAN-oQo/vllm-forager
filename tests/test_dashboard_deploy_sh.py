"""Tests for scripts/dashboard-deploy.sh and scripts/dashboard-restart.sh (T5.15) — offline &
deterministic where possible; real-tmux/real-server tests are `@pytest.mark.integration`.

Per the DEVPLAN todo: "the restart/deploy script is shell-checkable." What's checkable
offline: `dashboard-deploy.sh`'s own guards (dirty tree, wrong branch, git-pull failure, and
the "already up to date, nothing to restart" short-circuit) using a real scratch git repo +
remote (mirrors `tests/test_triage_sh.py`'s own dirty-tree-guard pattern) and a stub
`dashboard-restart.sh` that records whether it was actually invoked, rather than only
checking exit codes/messages (which alone wouldn't catch a guard silently falling through to
a real restart before still printing a similar-looking message). The real tmux-supervision
behavior of `dashboard-restart.sh` itself (kill + relaunch + verify-alive) is exercised
against real `tmux` + a real `python -m dashboard` process, gated `@pytest.mark.integration`
since it needs both actually installed and a real port bind.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.m5

ROOT = Path(__file__).resolve().parent.parent

_RESTART_CALLED_MARKER = "DASHBOARD_RESTART_CALLED_MARKER"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, timeout=30, capture_output=True)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=path)
    _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=path)
    _git("config", "user.email", "test@example.com", cwd=path)
    _git("config", "user.name", "Test", cwd=path)
    _git("config", "commit.gpgsign", "false", cwd=path)


def _write_scripts(repo: Path, tmp_path: Path, *, restart_exit_code: int = 0) -> None:
    """Copy the real `dashboard-deploy.sh` into `repo`, plus a stub `dashboard-restart.sh`
    that records whether it was invoked (never a real tmux/dashboard process here -- that's
    covered separately, against the real thing, by the integration tests below) and a no-op
    `notify.sh` stub (the real one is a silent no-op without `NOTIFY_URL` set anyway; stubbed
    here purely to avoid a noisy "file not found" on stderr for a script this test isn't
    about). Writes files only -- doesn't commit; callers use this on the seed repo, before
    its own initial commit, so the scripts are part of `origin`'s history from the start."""
    scripts_dir = repo / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "dashboard-deploy.sh", scripts_dir / "dashboard-deploy.sh")
    (scripts_dir / "dashboard-deploy.sh").chmod(0o755)

    stub_restart = scripts_dir / "dashboard-restart.sh"
    stub_restart.write_text(
        "#!/usr/bin/env bash\n"
        f'> "{tmp_path}/{_RESTART_CALLED_MARKER}"\n'
        f"exit {restart_exit_code}\n"
    )
    stub_restart.chmod(0o755)

    stub_notify = scripts_dir / "notify.sh"
    stub_notify.write_text("#!/usr/bin/env bash\nexit 0\n")
    stub_notify.chmod(0o755)


def _run_deploy_sh(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "scripts/dashboard-deploy.sh"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _restart_was_called(tmp_path: Path) -> bool:
    return (tmp_path / _RESTART_CALLED_MARKER).exists()


def _build_origin_and_clone(tmp_path: Path, *, restart_exit_code: int = 0) -> tuple[Path, Path]:
    """A bare `origin` repo + a real clone of it on `main` -- `dashboard-deploy.sh` needs a
    real remote to `git pull --ff-only` from, not just a standalone repo. The scripts (real
    dashboard-deploy.sh + stubs) are baked into the *seed's own initial commit*, not added to
    `clone` afterward -- committing them onto `clone` post-clone would diverge its history
    from `origin` (an extra commit `origin` never gets), breaking every later `--ff-only`
    pull in these tests; a real deploy clone's scripts/ is simply already part of `main`."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    _init_repo(seed)
    (seed / "README.md").write_text("seed\n")
    # Mirrors the real repo's own `data/` gitignore entry -- dashboard-deploy.sh's lock file
    # (and dashboard-restart.sh's logs) live under `data/`; without this, creating them would
    # itself trip the dirty-tree guard these tests are exercising.
    (seed / ".gitignore").write_text("data/\n")
    _write_scripts(seed, tmp_path, restart_exit_code=restart_exit_code)
    _git("add", "-A", cwd=seed)
    _git("commit", "-q", "-m", "initial", cwd=seed)
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(origin)], check=True, timeout=30)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, timeout=30)
    _git("config", "user.email", "test@example.com", cwd=clone)
    _git("config", "user.name", "Test", cwd=clone)
    _git("config", "commit.gpgsign", "false", cwd=clone)
    return origin, clone


def _push_a_new_commit(origin: Path, tmp_path: Path) -> None:
    """A second, throwaway clone pushes one new commit to `origin` -- so the test's own
    `clone` (already checked out) has something real to `git pull`."""
    other = tmp_path / "other-clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(other)], check=True, timeout=30)
    _git("config", "user.email", "test@example.com", cwd=other)
    _git("config", "user.name", "Test", cwd=other)
    _git("config", "commit.gpgsign", "false", cwd=other)
    (other / "NEWS.md").write_text("a new commit\n")
    _git("add", "-A", cwd=other)
    _git("commit", "-q", "-m", "a new commit", cwd=other)
    _git("push", "-q", "origin", "main", cwd=other)


# --------------------------------------------------------------------- dashboard-deploy.sh


def test_deploy_sh_refuses_a_dirty_working_tree(tmp_path: Path) -> None:
    _origin, clone = _build_origin_and_clone(tmp_path)
    (clone / "README.md").write_text("uncommitted local change\n")

    result = _run_deploy_sh(clone)

    assert result.returncode == 1
    assert "working tree dirty" in result.stderr
    assert not _restart_was_called(tmp_path)


def test_deploy_sh_refuses_a_dirty_working_tree_with_only_an_untracked_file(
    tmp_path: Path,
) -> None:
    """A stray *untracked* file must count as dirty too, not just a tracked-file edit --
    `git diff --quiet` alone (the original guard) is blind to untracked files."""
    _origin, clone = _build_origin_and_clone(tmp_path)
    (clone / "UNTRACKED.md").write_text("never committed\n")

    result = _run_deploy_sh(clone)

    assert result.returncode == 1
    assert "working tree dirty" in result.stderr
    assert not _restart_was_called(tmp_path)


def test_deploy_sh_refuses_a_non_main_branch(tmp_path: Path) -> None:
    _origin, clone = _build_origin_and_clone(tmp_path)
    _git("checkout", "-q", "-b", "some-feature-branch", cwd=clone)

    result = _run_deploy_sh(clone)

    assert result.returncode == 1
    assert "not 'main'" in result.stderr
    assert not _restart_was_called(tmp_path)


def test_deploy_sh_no_op_when_already_up_to_date(tmp_path: Path) -> None:
    _origin, clone = _build_origin_and_clone(tmp_path)

    result = _run_deploy_sh(clone)

    assert result.returncode == 0
    assert "already up to date" in result.stdout
    assert not _restart_was_called(tmp_path)


def test_deploy_sh_refuses_a_concurrent_run(tmp_path: Path) -> None:
    """A second invocation must not race the first's pull/restart -- hold the same
    `data/dashboard-deploy.lock` externally (as a concurrent deploy would) and confirm
    dashboard-deploy.sh backs off instead of proceeding."""
    origin, clone = _build_origin_and_clone(tmp_path)
    _push_a_new_commit(origin, tmp_path)

    (clone / "data").mkdir(exist_ok=True)
    lock_fd = os.open(clone / "data" / "dashboard-deploy.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = _run_deploy_sh(clone)
        assert result.returncode == 1
        assert "already running" in result.stderr
        assert not _restart_was_called(tmp_path)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def test_deploy_sh_pulls_and_restarts_when_a_new_commit_exists(tmp_path: Path) -> None:
    origin, clone = _build_origin_and_clone(tmp_path)
    _push_a_new_commit(origin, tmp_path)

    result = _run_deploy_sh(clone)

    assert result.returncode == 0
    assert (clone / "NEWS.md").exists()  # the pull actually landed the new commit
    assert _restart_was_called(tmp_path)


def test_deploy_sh_does_not_restart_when_the_restart_script_itself_fails(
    tmp_path: Path,
) -> None:
    origin, clone = _build_origin_and_clone(tmp_path, restart_exit_code=1)
    _push_a_new_commit(origin, tmp_path)

    result = _run_deploy_sh(clone)

    assert result.returncode == 1
    assert (clone / "NEWS.md").exists()  # the pull still landed...
    assert _restart_was_called(tmp_path)  # ...and a restart was attempted...
    # ...it just failed, which the exit code above already confirms.


def test_deploy_sh_fails_when_pull_would_not_fast_forward(tmp_path: Path) -> None:
    """A diverged local commit (never pushed) makes `--ff-only` fail -- dashboard-deploy.sh
    must not force/rebase past it, just refuse and leave the local state untouched."""
    origin, clone = _build_origin_and_clone(tmp_path)
    _push_a_new_commit(origin, tmp_path)
    (clone / "LOCAL.md").write_text("a local commit never pushed\n")
    _git("add", "-A", cwd=clone)
    _git("commit", "-q", "-m", "local divergent commit", cwd=clone)

    result = _run_deploy_sh(clone)

    assert result.returncode == 1
    assert not _restart_was_called(tmp_path)


# --------------------------------------------------------------------- dashboard-restart.sh
# (offline guard; the real tmux-supervision behavior is integration-only, below)


def test_restart_sh_exits_cleanly_when_a_flag_is_missing_its_value() -> None:
    """`--port` with no following value must not blow up as a raw bash "unbound variable"
    trace under `set -u` -- it should fail with a clear, intentional message instead."""
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "dashboard-restart.sh"), "--port"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "unbound variable" not in result.stderr
    assert "--port requires a value" in result.stderr


def test_restart_sh_exits_when_tmux_not_on_path(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for real in ("bash", "dirname", "command"):
        found = shutil.which(real)
        if found:
            (bin_dir / real).symlink_to(found)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "dashboard-restart.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert "tmux not found" in result.stderr


# --------------------------------------------------------------------- integration: real tmux


@pytest.mark.integration
def test_restart_sh_relaunches_a_real_dashboard_and_is_idempotent(tmp_path: Path) -> None:
    session = f"forager-dashboard-test-{os.getpid()}"
    env = os.environ.copy()
    env["FORAGER_DASHBOARD_TMUX_SESSION"] = session
    port = "18765"

    def _restart() -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "bash",
                "scripts/dashboard-restart.sh",
                "--port",
                port,
                "--data-dir",
                str(tmp_path),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    try:
        first = _restart()
        assert first.returncode == 0, first.stderr
        has_session = subprocess.run(
            ["tmux", "has-session", "-t", session], capture_output=True, timeout=10
        )
        assert has_session.returncode == 0

        # Relaunching again must kill the old session and start a fresh one -- idempotent,
        # not "already running, do nothing" (a code change needs the NEW code loaded).
        second = _restart()
        assert second.returncode == 0, second.stderr
        assert "killing existing session" in second.stdout
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, timeout=10)
