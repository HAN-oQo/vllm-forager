"""Tests for the dashboard live health panel (T5.8) — offline & deterministic.

Per the DEVPLAN todo: seeded run events → a running stage renders active with its step +
output tail; a stale heartbeat renders `stalled`; a `failed` event renders red.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dashboard import health
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5

_NOW = datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)


def test_stage_health_running_shows_step_and_output_tail(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(
        {
            "stage": "intel",
            "status": "heartbeat",
            "step": "item 40/120",
            "output_tail": "classifying...",
            "recorded_at": "2026-01-08T11:59:00Z",
        }
    )

    result = health.stage_health(store, "intel", now=_NOW, stale_after_s=600)

    assert result["status"] == "running"
    assert result["step"] == "item 40/120"
    assert result["output_tail"] == "classifying..."
    assert result["elapsed_s"] == pytest.approx(60.0)


def test_stage_health_stale_heartbeat_is_stalled(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(
        {
            "stage": "contribution",
            "status": "heartbeat",
            "step": "reviewing",
            "recorded_at": "2026-01-08T11:00:00Z",  # 3600s old, stale_after_s=600
        }
    )

    result = health.stage_health(store, "contribution", now=_NOW, stale_after_s=600)

    assert result["status"] == "stalled"


def test_stage_health_failed_event_carries_the_error(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(
        {
            "stage": "collect",
            "status": "failed",
            "error": "ConnectionError: timed out",
            "recorded_at": "2026-01-08T11:59:00Z",
        }
    )

    result = health.stage_health(store, "collect", now=_NOW, stale_after_s=600)

    assert result["status"] == "failed"
    assert result["error"] == "ConnectionError: timed out"


def test_stage_health_never_run(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    result = health.stage_health(store, "collect", now=_NOW, stale_after_s=600)
    assert result == {
        "stage": "collect",
        "status": "never run",
        "step": None,
        "output_tail": None,
        "elapsed_s": None,
        "error": None,
    }


def test_health_panel_a_failed_stage_makes_overall_red(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(
        {"stage": "collect", "status": "failed", "recorded_at": "2026-01-08T11:59:00Z"}
    )

    panel = health.health_panel(store, now=_NOW, stale_after_s=600)

    assert panel["overall"] == "red"
    assert {s["stage"] for s in panel["stages"]} == {"collect", "intel", "contribution"}


def test_health_panel_all_healthy_is_green(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    for stage in ("collect", "intel", "contribution"):
        store.record_run({"stage": stage, "status": "ok", "recorded_at": "2026-01-08T11:59:00Z"})

    panel = health.health_panel(store, now=_NOW, stale_after_s=600)

    assert panel["overall"] == "green"


def test_health_panel_a_stalled_stage_makes_overall_red(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(
        {"stage": "intel", "status": "heartbeat", "recorded_at": "2026-01-08T11:00:00Z"}
    )

    panel = health.health_panel(store, now=_NOW, stale_after_s=600)

    assert panel["overall"] == "red"


def test_stage_health_accepts_a_naive_now(tmp_path: Path) -> None:
    """Regression: an earlier version didn't normalize a naive `now` to UTC (unlike
    src.orchestrator.run_tick's own identical handling of this same common idiom), crashing
    with 'can't subtract offset-naive and offset-aware datetimes'."""
    store = JsonlStore(tmp_path)
    store.record_run({"stage": "collect", "status": "ok", "recorded_at": "2026-01-08T11:59:00Z"})
    naive_now = datetime(2026, 1, 8, 12, 0, 0)  # no tzinfo

    result = health.stage_health(store, "collect", now=naive_now, stale_after_s=600)  # no raise

    assert result["status"] == "ok"
    assert result["elapsed_s"] == pytest.approx(60.0)


def test_health_panel_fetches_runs_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an earlier version had health_panel call stage_health once per plane, each
    doing its own store.list_runs(stage=...) call -- 3 separate reads instead of 1, the exact
    anti-pattern dashboard.pipeline_diagram's own review already found and fixed."""
    store = JsonlStore(tmp_path)
    store.record_run({"stage": "collect", "status": "ok", "recorded_at": "2026-01-08T11:59:00Z"})
    calls = []
    real_list_runs = store.list_runs

    def _counting_list_runs(*args, **kwargs):
        calls.append((args, kwargs))
        return real_list_runs(*args, **kwargs)

    monkeypatch.setattr(store, "list_runs", _counting_list_runs)

    health.health_panel(store, now=_NOW, stale_after_s=600)

    assert len(calls) == 1
    assert calls[0] == ((), {})  # unfiltered -- grouped by stage locally, not per-stage calls


def test_stage_health_error_and_status_agree_on_a_same_second_tie(tmp_path: Path) -> None:
    """Regression: an earlier version derived 'the latest record' two independent ways
    (liveness.latest_status's own tie-aware pick vs. a second, tie-unaware stages.latest_run
    call), which could disagree on a same-second tie between a terminal and non-terminal
    event -- reporting status='failed' alongside error=None from the wrong record."""
    store = JsonlStore(tmp_path)
    same_second = "2026-01-08T11:59:00Z"
    store.record_run({"stage": "collect", "status": "heartbeat", "recorded_at": same_second})
    store.record_run(
        {
            "stage": "collect",
            "status": "failed",
            "error": "boom",
            "recorded_at": same_second,
        }
    )

    result = health.stage_health(store, "collect", now=_NOW, stale_after_s=600)

    assert result["status"] == "failed"
    assert result["error"] == "boom"
