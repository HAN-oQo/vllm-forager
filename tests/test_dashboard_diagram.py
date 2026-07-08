"""Tests for the pipeline data-flow diagram (T5.5) — offline & deterministic.

Per the DEVPLAN todo: diagram data builds from live stage metadata (smoke).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dashboard import pipeline_diagram
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5

_NOW = datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)


def test_pipeline_diagram_builds_from_live_stage_metadata(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run({"stage": "collect", "status": "ok", "recorded_at": "2026-01-08T11:59:00Z"})

    diagram = pipeline_diagram.pipeline_diagram(store, now=_NOW, stale_after_s=600)

    nodes = {n["id"]: n for n in diagram["nodes"]}
    assert nodes["kb"]["status"] is None
    assert nodes["collect"]["status"] == "ok"
    assert nodes["intel"]["status"] == "never run"
    assert nodes["contribution"]["status"] == "never run"
    assert set(diagram["edges"]) == {("collect", "kb"), ("intel", "kb"), ("contribution", "kb")}


def test_pipeline_diagram_flags_a_stale_heartbeat(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(
        {"stage": "intel", "status": "heartbeat", "recorded_at": "2026-01-08T11:00:00Z"}
    )

    diagram = pipeline_diagram.pipeline_diagram(store, now=_NOW, stale_after_s=600)

    nodes = {n["id"]: n for n in diagram["nodes"]}
    assert nodes["intel"]["status"] == "stalled"


def test_pipeline_diagram_defaults_now_to_the_real_clock(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    diagram = pipeline_diagram.pipeline_diagram(store, stale_after_s=600)  # must not raise
    assert {n["id"] for n in diagram["nodes"]} == {"kb", "collect", "intel", "contribution"}
