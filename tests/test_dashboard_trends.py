"""Tests for the dashboard trend-visualization data layer (T5.3) — offline & deterministic.

Per the DEVPLAN todo: the series endpoint returns bucketed (dense, gap-filled) points.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dashboard import trend_charts
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5

_NOW = datetime(2026, 1, 19, tzinfo=timezone.utc)  # 2026-W04


def _item(repo: str, number: int, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": f"item {number}",
        "state": "open",
        "labels": [],
        "created_at": "2026-01-05T00:00:00Z",
        "updated_at": "2026-01-05T00:00:00Z",
        "url": f"http://x/{repo}/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


def test_category_series_is_dense_and_gap_filled(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            # 2026-W02
            _item("o/r", 1, category="build", created_at="2026-01-05T00:00:00Z"),
            # 2026-W03 has no "build" activity -- must still appear, as a zero
            # 2026-W04 (== _NOW's own week)
            _item("o/r", 2, category="build", created_at="2026-01-19T00:00:00Z"),
        ]
    )

    series = trend_charts.category_series(store, ["build"], weeks=3, now=_NOW)

    assert series == {
        "build": [
            {"week": "2026-W02", "count": 1},
            {"week": "2026-W03", "count": 0},
            {"week": "2026-W04", "count": 1},
        ]
    }


def test_category_series_defaults_to_every_known_category(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, category="build", created_at="2026-01-05T00:00:00Z"),
            _item("o/r", 2, category="quantization", created_at="2026-01-05T00:00:00Z"),
        ]
    )

    series = trend_charts.category_series(store, weeks=1, now=_NOW)

    assert series.keys() == {"build", "quantization"}


def test_category_series_requested_category_with_no_activity_is_all_zeros(
    tmp_path: Path,
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, category="build", created_at="2026-01-05T00:00:00Z")])

    series = trend_charts.category_series(store, ["build", "never-active"], weeks=2, now=_NOW)

    assert series["never-active"] == [
        {"week": "2026-W03", "count": 0},
        {"week": "2026-W04", "count": 0},
    ]


def test_category_series_no_activity_at_all_returns_empty_dict(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert trend_charts.category_series(store, now=_NOW) == {}
