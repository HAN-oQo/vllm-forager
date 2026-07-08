"""Tests for orchestrator run events (T4.4) — offline & deterministic.

Per the DEVPLAN todo: a pipeline tick writes a run event with the expected fields --
`{stage, status, items/error, dur_s, policy_version, recorded_at}` -- to the KB's `runs`
collection, for both a stage that succeeds and one that raises.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src import orchestrator, policy, taxonomy
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m4

_NOW = datetime(2026, 1, 8, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path) -> JsonlStore:
    s = JsonlStore(tmp_path)
    taxonomy.create_taxonomy(s, ["rocm-build"])
    policy.create_policy(s, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1)
    return s


def test_run_tick_writes_a_run_event_with_the_expected_fields(store: JsonlStore) -> None:
    """DEVPLAN's own worked example: `{stage: collect, status: ok, items: 42, dur_s: 31}`."""
    stage = orchestrator.Stage("collect", lambda _s: 42, interval_hours=24)

    orchestrator.run_tick(store, [stage], now=_NOW)

    runs = store.list_runs(stage="collect")
    assert len(runs) == 1
    run = runs[0]
    assert run["stage"] == "collect"
    assert run["status"] == "ok"
    assert run["items"] == 42
    assert run["dur_s"] >= 0
    assert run["policy_version"] == 1
    assert run["recorded_at"] == "2026-01-08T00:00:00Z"


def test_run_tick_writes_a_failed_run_event_when_a_stage_raises(store: JsonlStore) -> None:
    def _boom(_store) -> None:
        raise RuntimeError("boom")

    stage = orchestrator.Stage("intel", _boom, interval_hours=24)

    with pytest.raises(RuntimeError, match="boom"):
        orchestrator.run_tick(store, [stage], now=_NOW)

    runs = store.list_runs(stage="intel")
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["error"] == "RuntimeError: boom"  # exception type included, not just str(exc)
    assert "items" not in runs[0]


def test_run_tick_writes_both_an_ok_and_a_failed_event_in_the_same_tick(
    store: JsonlStore,
) -> None:
    """An earlier stage's success must be recorded even though a later stage in the same tick
    raises -- run_tick has no per-stage exception isolation (still aborts the tick), but the
    run-event log for stages that already finished must not be lost."""

    def _boom(_store) -> None:
        raise RuntimeError("boom")

    stages = [
        orchestrator.Stage("collect", lambda _s: 1, interval_hours=24),
        orchestrator.Stage("intel", _boom, interval_hours=24),
    ]

    with pytest.raises(RuntimeError, match="boom"):
        orchestrator.run_tick(store, stages, now=_NOW)

    assert store.list_runs(stage="collect")[0]["status"] == "ok"
    assert store.list_runs(stage="intel")[0]["status"] == "failed"


def test_run_tick_writes_no_event_for_a_skipped_stage(store: JsonlStore) -> None:
    stage = orchestrator.Stage("contribution", lambda _s: 1, trigger=lambda _s: False)

    orchestrator.run_tick(store, [stage], now=_NOW)

    assert store.list_runs(stage="contribution") == []


def test_run_tick_writes_one_event_per_stage_this_tick(store: JsonlStore) -> None:
    stages = [
        orchestrator.Stage("collect", lambda _s: 1, interval_hours=24),
        orchestrator.Stage("contribution", lambda _s: 2, trigger=lambda _s: True),
    ]

    orchestrator.run_tick(store, stages, now=_NOW)

    assert len(store.list_runs()) == 2
    assert {r["stage"] for r in store.list_runs()} == {"collect", "contribution"}


def test_run_tick_survives_a_record_run_failure_but_the_event_is_genuinely_lost(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A KB-write hiccup while recording the run event must not crash the tick -- the stage's
    own real work already happened and must not be discarded over this. But "best effort"
    means exactly that: the run event itself is genuinely never written when record_run fails,
    not silently retried or queued -- `TickResult.ran` and the KB's own run history can
    legitimately disagree about a stage that "ran" but wasn't recorded."""
    monkeypatch.setattr(
        store, "record_run", lambda _r: (_ for _ in ()).throw(RuntimeError("disk full"))
    )
    calls: list[str] = []
    stage = orchestrator.Stage("collect", lambda _s: calls.append("ran"), interval_hours=24)

    result = orchestrator.run_tick(store, [stage], now=_NOW)

    assert calls == ["ran"]
    assert result.ran == ("collect",)
    assert store.list_runs(stage="collect") == []  # the event really is missing, not queued
    assert "failed to record run event" in capsys.readouterr().err


def test_run_tick_survives_a_set_state_failure_after_a_successful_run_event(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A cursor-write hiccup right after a stage's run event was already recorded must not
    crash the tick either -- otherwise a KB reader would see a clean "ok" event for a tick that
    the CLI itself reports as a crash."""
    monkeypatch.setattr(
        store, "set_state", lambda *a: (_ for _ in ()).throw(RuntimeError("disk full"))
    )
    stage = orchestrator.Stage("collect", lambda _s: 1, interval_hours=24)

    result = orchestrator.run_tick(store, [stage], now=_NOW)

    assert result.ran == ("collect",)
    assert store.list_runs(stage="collect")[0]["status"] == "ok"
    assert "failed to advance last-run cursor" in capsys.readouterr().err
