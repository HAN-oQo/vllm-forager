"""Tests for the dashboard reports archive (T5.9) — offline & deterministic.

Per the DEVPLAN todo: seeded daily reports → archive lists them newest-first; selecting a
date returns that report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dashboard import archive
from src import report
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5


def _seeded_store(tmp_path: Path) -> JsonlStore:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            {
                "repo": "o/r",
                "number": 1,
                "type": "issue",
                "title": "x",
                "state": "open",
                "category": "build",
                "path": ["build"],
                "created_at": "2026-01-05T00:00:00Z",
                "updated_at": "2026-01-05T00:00:00Z",
                "url": "http://x/1",
                "body": "",
            }
        ]
    )
    return store


def test_list_archived_reports_returns_newest_first(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    report.generate_tree(
        store, tmp_path / "reports", when=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )
    report.generate_tree(
        store, tmp_path / "reports", when=datetime(2026, 6, 1, tzinfo=timezone.utc)
    )

    reports = archive.list_archived_reports(store)

    assert [r["stamp"] for r in reports] == ["2026-W23", "2026-W02"]


def test_list_archived_reports_empty_before_any_report_generated(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert archive.list_archived_reports(store) == []


def test_open_archived_report_returns_the_tree_content(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    report.generate_tree(
        store, tmp_path / "reports", when=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )

    tree = archive.open_archived_report(store, "2026-W02")

    assert tree is not None
    assert tree[0]["name"] == "build"
    assert tree[0]["count"] == 1


def test_open_archived_report_unknown_stamp_returns_none(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    report.generate_tree(
        store, tmp_path / "reports", when=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )

    assert archive.open_archived_report(store, "2026-W99") is None


def test_open_archived_report_no_reports_dir_returns_none(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert archive.open_archived_report(store, "2026-W02") is None


def test_archive_degrades_for_a_store_with_no_data_dir() -> None:
    class NoDirStore:
        pass  # no `data_dir` attribute — mimics a future non-JSONL Store

    assert archive.list_archived_reports(NoDirStore()) == []
    assert archive.open_archived_report(NoDirStore(), "2026-W02") is None
