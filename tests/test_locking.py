"""Tests for the run lock (T4.3) — offline & deterministic.

Per the DEVPLAN todo: a second concurrent run backs off (no crash, no duplicate work) rather
than racing the first. `flock` is scoped to the *open file description*, not the process, so a
held lock from a fresh `open()` of the same path conflicts even within a single test process --
no subprocess/multiprocessing needed to exercise this deterministically.
"""

from __future__ import annotations

import fcntl

import pytest

from src import config, locking, orchestrator
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m4


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Every test in this file gets its own scratch `config.DATA_DIR` -- `run_lock` writes a
    lock file there, and a test must never touch the real repo's `data/` directory."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def test_run_lock_raises_if_already_held(tmp_path) -> None:
    with locking.run_lock("test"), pytest.raises(locking.LockHeld), locking.run_lock("test"):
        pass  # pragma: no cover -- LockHeld raises before the block body ever runs


def test_run_lock_releases_after_the_with_block(tmp_path) -> None:
    with locking.run_lock("test"):
        pass

    with locking.run_lock("test"):  # must succeed -- the prior lock was released
        pass


def test_run_lock_releases_even_if_the_with_block_raises(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="boom"), locking.run_lock("test"):
        raise RuntimeError("boom")

    with locking.run_lock("test"):  # must still succeed -- the finally: block released it
        pass


def test_run_lock_different_names_dont_conflict(tmp_path) -> None:
    with locking.run_lock("a"), locking.run_lock("b"):
        pass  # no LockHeld -- distinct lock files


def test_run_lock_conflicts_even_within_one_process(tmp_path) -> None:
    """flock is scoped to the open file description, not the process -- two independent
    `open()` calls on the same path from the same process still conflict."""
    path = tmp_path / "manual.lock"
    fh = path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(locking.LockHeld), locking.run_lock("manual"):
            pass  # pragma: no cover
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def test_orchestrator_main_backs_off_when_a_tick_is_already_in_flight(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """DEVPLAN's own worked example: 'a second run starts while the first is mid-flight -> it
    backs off; no item written twice.' `run_tick` is stubbed here (already covered by
    tests/test_orchestrator.py), so the store just needs an identity for `get_store()` to
    return -- no taxonomy/policy bootstrap is needed since `run_tick` never actually runs."""
    monkeypatch.setattr(orchestrator, "get_store", lambda: JsonlStore(tmp_path / "kb"))
    calls: list[str] = []
    monkeypatch.setattr(orchestrator, "run_tick", lambda *a, **k: calls.append("ran"))

    with locking.run_lock("orchestrator"):  # simulates a tick already mid-flight
        rc = orchestrator.main(["--once"])

    assert rc == 0
    assert "skipping" in capsys.readouterr().err
    assert calls == []  # run_tick was never even called -- nothing to double-write
