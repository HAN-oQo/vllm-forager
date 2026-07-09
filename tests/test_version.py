"""Tests for the build/version stamp (T5.15) — offline & deterministic.

Per the DEVPLAN todo: a smoke check asserts the served page carries the current build/commit
stamp. `GIT_SHA` itself is computed once, at import time, so it's tested here as
`_git_sha()` (the underlying, independently-callable function) rather than by trying to
re-trigger module-level import-time computation.
"""

from __future__ import annotations

import subprocess

import pytest

from src import version

pytestmark = pytest.mark.m5


def test_git_sha_returns_a_real_short_sha_in_this_repo() -> None:
    """This test itself runs inside a real git checkout -- `_git_sha()` should return a real,
    short (7-char) hex SHA, not the "unknown" fallback."""
    sha = version._git_sha()
    assert sha != "unknown"
    assert len(sha) >= 7
    assert all(c in "0123456789abcdef" for c in sha)


def test_git_sha_degrades_to_unknown_when_git_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*a, **k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(version.subprocess, "run", _fail)

    assert version._git_sha() == "unknown"


def test_git_sha_degrades_to_unknown_when_git_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(version.subprocess, "run", _fail)

    assert version._git_sha() == "unknown"


def test_git_sha_degrades_to_unknown_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 5))

    monkeypatch.setattr(version.subprocess, "run", _hang)

    assert version._git_sha() == "unknown"


def test_git_sha_degrades_to_unknown_on_blank_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blank(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="\n")

    monkeypatch.setattr(version.subprocess, "run", _blank)

    assert version._git_sha() == "unknown"


def test_module_level_git_sha_constant_is_a_real_string() -> None:
    """GIT_SHA is computed once at import time -- sanity-check it's already populated with a
    real value from this actual repo checkout, not left as a lazy/None placeholder."""
    assert isinstance(version.GIT_SHA, str)
    assert version.GIT_SHA != ""
