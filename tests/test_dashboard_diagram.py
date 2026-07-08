"""Tests for the pipeline data-flow diagram (T5.5) — offline & deterministic.

Per the DEVPLAN todo: diagram data builds from live stage metadata (smoke).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dashboard import pipeline_diagram
from src import orchestrator
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5

_NOW = datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "recorded_run, stage_under_test, expected_status",
    [
        (
            {"stage": "collect", "status": "ok", "recorded_at": "2026-01-08T11:59:00Z"},
            "collect",
            "ok",
        ),
        (
            {"stage": "intel", "status": "heartbeat", "recorded_at": "2026-01-08T11:00:00Z"},
            "intel",
            "stalled",
        ),
    ],
)
def test_pipeline_diagram_reflects_a_recorded_runs_status(
    tmp_path: Path, recorded_run: dict, stage_under_test: str, expected_status: str
) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(recorded_run)

    diagram = pipeline_diagram.pipeline_diagram(store, now=_NOW, stale_after_s=600)

    nodes = {n["id"]: n for n in diagram["nodes"]}
    assert nodes[stage_under_test]["status"] == expected_status


def test_pipeline_diagram_kb_node_and_edges(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run({"stage": "collect", "status": "ok", "recorded_at": "2026-01-08T11:59:00Z"})

    diagram = pipeline_diagram.pipeline_diagram(store, now=_NOW, stale_after_s=600)

    nodes = {n["id"]: n for n in diagram["nodes"]}
    assert nodes["kb"]["status"] is None
    assert nodes["intel"]["status"] == "never run"
    assert nodes["contribution"]["status"] == "never run"
    assert set(diagram["edges"]) == {("collect", "kb"), ("intel", "kb"), ("contribution", "kb")}


def test_pipeline_diagram_defaults_now_to_the_real_clock(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    diagram = pipeline_diagram.pipeline_diagram(store, stale_after_s=600)  # must not raise
    assert {n["id"] for n in diagram["nodes"]} == {"kb", "collect", "intel", "contribution"}


def test_pipeline_diagram_accepts_a_naive_now(tmp_path: Path) -> None:
    """Regression: an earlier version didn't normalize a naive `now` to UTC (unlike
    src.orchestrator.run_tick's own identical handling of this same common idiom), crashing
    with 'can't subtract offset-naive and offset-aware datetimes' (found while reviewing
    dashboard.health, T5.8, which shares this same gap)."""
    store = JsonlStore(tmp_path)
    store.record_run({"stage": "collect", "status": "ok", "recorded_at": "2026-01-08T11:59:00Z"})
    naive_now = datetime(2026, 1, 8, 12, 0, 0)  # no tzinfo

    diagram = pipeline_diagram.pipeline_diagram(store, now=naive_now, stale_after_s=600)

    assert {n["id"]: n["status"] for n in diagram["nodes"] if n["id"] == "collect"} == {
        "collect": "ok"
    }


def test_pipeline_diagram_plane_ids_match_the_real_orchestrator_stages() -> None:
    """Regression: `PLANES`'s ids are a second, hand-maintained copy of `_real_stages()`'s
    own `Stage.name` values (a code-review finding) -- this guards against the two drifting
    apart silently if a real orchestrator stage is ever renamed or added."""
    real_names = {stage.name for stage in orchestrator._real_stages()}
    assert {plane["id"] for plane in pipeline_diagram.PLANES} == real_names
