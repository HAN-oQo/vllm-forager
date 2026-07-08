"""Tests for the per-category trend series (T1.7) — offline & deterministic.

Per the DEVPLAN todo: synthetic items across weeks -> correct bucketed counts per category.
Also covers the uncategorized-item exclusion, the explicit-Other inclusion, malformed/missing
``created_at`` tolerance, and the store-backed wrapper.
"""

from datetime import datetime, timezone

import pytest

from src import trends
from src.store.jsonl_store import JsonlStore
from src.trends import week_stamp

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


def test_category_trends_skips_non_string_created_at(capsys) -> None:
    """Regression: a non-string created_at (e.g. an int) raised an uncaught TypeError."""
    items = [_item("o/r", 1, 1735689600, category="ROCm / AMD")]  # type: ignore[arg-type]
    assert trends.category_trends(items) == {}
    assert "skipping o/r#1" in capsys.readouterr().err


def test_category_trends_skips_non_string_category() -> None:
    """Regression: a non-string truthy category (e.g. a list) crashed with unhashable type."""
    items = [_item("o/r", 1, _WEEK1_TS, category=["ROCm / AMD"])]
    assert trends.category_trends(items) == {}


def test_category_trends_logs_skipped_malformed_timestamp(capsys) -> None:
    items = [_item("o/r", 1, "not-a-date", category="ROCm / AMD")]
    trends.category_trends(items)
    assert "skipping o/r#1" in capsys.readouterr().err


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


# --------------------------------------------------------------------- last_n_weeks


def test_last_n_weeks_is_oldest_first_and_includes_nows_own_week() -> None:
    now = datetime(2026, 1, 19, tzinfo=timezone.utc)  # 2026-W04

    assert trends.last_n_weeks(3, now=now) == ["2026-W02", "2026-W03", "2026-W04"]


def test_last_n_weeks_single_week_is_just_nows_own_week() -> None:
    now = datetime(2026, 1, 19, tzinfo=timezone.utc)  # 2026-W04

    assert trends.last_n_weeks(1, now=now) == ["2026-W04"]


def test_last_n_weeks_zero_weeks_returns_empty_list() -> None:
    assert trends.last_n_weeks(0, now=datetime(2026, 1, 19, tzinfo=timezone.utc)) == []
