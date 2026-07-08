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
