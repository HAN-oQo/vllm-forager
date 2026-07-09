"""Tests for the dashboard candidate-selection surface (T5.10) — offline & deterministic.

Per the DEVPLAN todo: a select action writes a decision record; the engineer/orchestrator
query returns only selected candidates (write path mocked; everything else stays read-only).

This module is a thin wrapper over `src.selection`; these tests confirm the wrapper composes
correctly, mirroring `src.selection`'s own already-thorough coverage in `tests/test_selection.py`
rather than re-deriving it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard import select
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5


def _seed_item(store: JsonlStore, repo: str, number: int) -> None:
    store.upsert_items(
        [
            {
                "repo": repo,
                "number": number,
                "type": "issue",
                "title": "x",
                "state": "open",
                "url": f"https://github.com/{repo}/issues/{number}",
            }
        ]
    )


def test_select_candidate_writes_a_selected_decision(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)

    record = select.select_candidate(store, "o/r", 1, by="hankyu")

    assert record is not None
    assert record["decision"] == "selected"
    assert select.candidate_decision_status(store, "o/r", 1) == "selected"


def test_skip_candidate_writes_a_skip_decision(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)

    record = select.skip_candidate(store, "o/r", 1, by="hankyu")

    assert record is not None
    assert record["decision"] == "skip"
    assert select.candidate_decision_status(store, "o/r", 1) == "skip"


def test_candidate_decision_status_none_before_any_decision(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)
    assert select.candidate_decision_status(store, "o/r", 1) is None


def test_select_candidate_returns_none_for_a_nonexistent_item(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert select.select_candidate(store, "o/r", 1) is None
