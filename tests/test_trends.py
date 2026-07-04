"""Tests for the per-category trend series (T1.7) — offline & deterministic.

Per the DEVPLAN todo: synthetic items across weeks -> correct bucketed counts per category.
Also covers the uncategorized-item exclusion, the explicit-Other inclusion, malformed/missing
``created_at`` tolerance, and the store-backed wrapper.
"""

from datetime import datetime, timezone

import pytest

from src import trends
from src.report import week_stamp
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m1

_WEEK1_TS = "2026-01-05T00:00:00Z"
_WEEK2_TS = "2026-01-12T00:00:00Z"
_W1 = week_stamp(datetime(2026, 1, 5, tzinfo=timezone.utc))
_W2 = week_stamp(datetime(2026, 1, 12, tzinfo=timezone.utc))


def _item(repo: str, number: int, created_at: str, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": f"t{number}",
        "state": "open",
        "labels": [],
        "created_at": created_at,
        "updated_at": created_at,
        "url": f"http://x/{repo}/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


def test_category_trends_buckets_by_creation_week_per_category() -> None:
    """The DEVPLAN's named scenario: synthetic items across weeks -> correct bucketed counts."""
    items = [
        _item("o/r", 1, _WEEK1_TS, category="ROCm / AMD"),
        _item("o/r", 2, _WEEK1_TS, category="ROCm / AMD"),
        _item("o/r", 3, _WEEK2_TS, category="ROCm / AMD"),
        _item("o/r", 4, _WEEK1_TS, category="Quantization"),
    ]

    result = trends.category_trends(items)

    assert result == {
        "ROCm / AMD": {_W1: 2, _W2: 1},
        "Quantization": {_W1: 1},
    }


def test_category_trends_excludes_uncategorized_items() -> None:
    items = [_item("o/r", 1, _WEEK1_TS)]  # no `category` field yet
    assert trends.category_trends(items) == {}


def test_category_trends_includes_explicit_other_category() -> None:
    items = [_item("o/r", 1, _WEEK1_TS, category="Other")]
    assert trends.category_trends(items) == {"Other": {_W1: 1}}


def test_category_trends_skips_missing_or_malformed_created_at() -> None:
    items = [
        _item("o/r", 1, "", category="ROCm / AMD"),
        _item("o/r", 2, "not-a-date", category="ROCm / AMD"),
    ]
    assert trends.category_trends(items) == {}


def test_category_trends_empty_input_returns_empty_dict() -> None:
    assert trends.category_trends([]) == {}


def test_trends_from_store_reads_all_items(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, _WEEK1_TS, category="ROCm / AMD"),
            _item("o/r", 2, _WEEK2_TS, category="ROCm / AMD"),
        ]
    )

    assert trends.trends_from_store(store) == {"ROCm / AMD": {_W1: 1, _W2: 1}}
