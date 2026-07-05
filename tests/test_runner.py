"""Tests for the remote runner (T3.1) — offline & deterministic; ssh/scp mocked at the
subprocess boundary.

Per the DEVPLAN todo: mock subprocess/ssh -> correct command composed + result parsed. Real
ssh is @pytest.mark.integration, gated on MI250_HOST, and skipped by default.
"""

import os
import subprocess

import pytest

from src import runner
from src.runner import RunnerError, run

pytestmark = pytest.mark.m3


class _FakeProcess:
    """Stand-in for a subprocess.Popen handle: an iterable `.stdout`, a `.wait()` that returns
    `returncode` (or raises `wait_exc`), and a `.kill()` flag for the timeout path."""

    def __init__(self, lines=(), returncode=0, wait_exc=None):
        self.stdout = iter(lines)
        self._returncode = returncode
        self._wait_exc = wait_exc
        self.killed = False

    def wait(self, timeout=None):
        if self._wait_exc is not None:
            raise self._wait_exc
        return self._returncode

    def kill(self):
        self.killed = True


# --------------------------------------------------------------------- run: command composition


def test_run_composes_ssh_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append(cmd)
        return _FakeProcess(lines=["ok\n"], returncode=0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    run("mi250-051", "pytest test_rocm.py")

    assert calls == [["ssh", "mi250-051", "pytest test_rocm.py"]]


# --------------------------------------------------------------------- run: streaming + exit code


def test_run_streams_and_captures_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda cmd, **kwargs: _FakeProcess(lines=["line1\n", "line2\n"], returncode=0),
    )

    result = run("mi250-051", "echo hi")

    assert result.exit_code == 0
    assert result.log == "line1\nline2\n"


def test_run_nonzero_exit_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing remote command is a normal result, not an error -- e.g. a repro's failing
    signal is exactly what T3.2 wants captured."""
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda cmd, **kwargs: _FakeProcess(lines=["AssertionError\n"], returncode=1),
    )

    result = run("mi250-051", "pytest test_rocm.py")

    assert result.exit_code == 1
    assert "AssertionError" in result.log


# --------------------------------------------------------------------- run: error paths


def test_run_raises_when_ssh_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_popen(cmd, **kwargs):
        raise FileNotFoundError("no ssh")

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    with pytest.raises(RunnerError, match="ssh"):
        run("mi250-051", "echo hi")


def test_run_raises_and_kills_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_proc = _FakeProcess(lines=[], returncode=0, wait_exc=subprocess.TimeoutExpired("cmd", 1))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda cmd, **kwargs: fake_proc)

    with pytest.raises(RunnerError, match="timed out"):
        run("mi250-051", "sleep 9999", timeout=1)

    assert fake_proc.killed is True


# --------------------------------------------------------------------- run: artifacts


def test_run_no_artifacts_requested_returns_empty_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda cmd, **kwargs: _FakeProcess(returncode=0)
    )

    def fail_if_called(*a, **k):
        raise AssertionError("scp should not run with no artifacts requested")

    monkeypatch.setattr(runner.subprocess, "run", fail_if_called)

    result = run("mi250-051", "echo hi")

    assert result.artifacts == ()


def test_run_fetches_requested_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda cmd, **kwargs: _FakeProcess(returncode=0)
    )
    scp_calls = []

    def fake_scp_run(cmd, **kwargs):
        scp_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_scp_run)

    result = run("mi250-051", "echo hi", artifacts=["/remote/log.txt"], local_dir=tmp_path)

    assert scp_calls == [["scp", "mi250-051:/remote/log.txt", str(tmp_path / "log.txt")]]
    assert result.artifacts == (str(tmp_path / "log.txt"),)


def test_fetch_skips_failed_artifact_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda cmd, **kwargs: _FakeProcess(returncode=0)
    )

    def fake_scp_run(cmd, **kwargs):
        if "missing.txt" in cmd[1]:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_scp_run)

    result = run(
        "mi250-051",
        "echo hi",
        artifacts=["/remote/missing.txt", "/remote/ok.txt"],
        local_dir=tmp_path,
    )

    assert result.artifacts == (str(tmp_path / "ok.txt"),)


# --------------------------------------------------------------------- live smoke


@pytest.mark.integration
def test_run_live_smoke() -> None:
    host = os.getenv("MI250_HOST")
    if not host:
        pytest.skip("MI250_HOST not set")
    result = run(host, "echo hello-from-mi250")
    assert result.exit_code == 0
    assert "hello-from-mi250" in result.log
