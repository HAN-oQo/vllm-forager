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


def test_run_tick_writes_an_item_count_of_none_when_a_stage_doesnt_report_one(
    store: JsonlStore,
) -> None:
    stage = orchestrator.Stage("collect", lambda _s: None, interval_hours=24)

    orchestrator.run_tick(store, [stage], now=_NOW)

    assert store.list_runs(stage="collect")[0]["items"] is None


def test_run_tick_writes_a_failed_run_event_when_a_stage_raises(store: JsonlStore) -> None:
    def _boom(_store) -> None:
        raise RuntimeError("boom")

    stage = orchestrator.Stage("intel", _boom, interval_hours=24)

    with pytest.raises(RuntimeError, match="boom"):
        orchestrator.run_tick(store, [stage], now=_NOW)

    runs = store.list_runs(stage="intel")
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["error"] == "boom"
    assert "items" not in runs[0]


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


def test_run_tick_still_records_a_run_event_when_store_record_run_itself_fails(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A KB-write hiccup while recording the run event must not crash the tick -- the stage's
    own real work already happened and must not be discarded over this."""
    monkeypatch.setattr(
        store, "record_run", lambda _r: (_ for _ in ()).throw(RuntimeError("disk full"))
    )
    calls: list[str] = []
    stage = orchestrator.Stage("collect", lambda _s: calls.append("ran"), interval_hours=24)

    result = orchestrator.run_tick(store, [stage], now=_NOW)

    assert calls == ["ran"]
    assert result.ran == ("collect",)
    assert "failed to record run event" in capsys.readouterr().err
