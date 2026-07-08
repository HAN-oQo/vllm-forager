"""Tests for the dashboard monitoring panels (T5.2) — offline & deterministic.

Per the DEVPLAN todo: panel data functions return expected shapes from fixtures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dashboard import panels
from src import taxonomy
from src.agents import scout
from src.agents.forecaster import Prediction, record_prediction
from src.agents.grader import Grade
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5

_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)  # 2026-W23


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


# --------------------------------------------------------------------- collection_stats


def test_collection_stats_counts_per_repo_and_total(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, type="issue"),
            _item("o/r", 2, type="pr", labels=["ROCm"]),
            _item("o/other", 3, type="issue"),
        ]
    )

    stats = panels.collection_stats(store)

    assert stats["total"] == 3
    assert stats["repos"]["o/r"] == {"total": 2, "issue": 1, "pr": 1, "rocm": 1}
    assert stats["repos"]["o/other"] == {"total": 1, "issue": 1, "pr": 0, "rocm": 0}


def test_collection_stats_empty_store(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert panels.collection_stats(store) == {"repos": {}, "total": 0}


# --------------------------------------------------------------------- taxonomy_timeline


def test_taxonomy_timeline_no_taxonomy_returns_empty(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert panels.taxonomy_timeline(store, now=_NOW) == []


def test_taxonomy_timeline_reports_added_version(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    taxonomy.create_taxonomy(store, ["build"])
    taxonomy.add_category(store, "quantization")
    store.upsert_items(
        [
            _item("o/r", 1, category="build", created_at="2026-05-25T00:00:00Z"),
            _item("o/r", 2, category="quantization", created_at="2026-05-25T00:00:00Z"),
        ]
    )

    timeline = panels.taxonomy_timeline(store, now=_NOW)

    by_category = {e["category"]: e for e in timeline}
    assert by_category["build"]["added_in_version"] == 1
    assert by_category["quantization"]["added_in_version"] == 2


def test_taxonomy_timeline_flags_inactive_category_as_retirement_proposal(
    tmp_path: Path,
) -> None:
    store = JsonlStore(tmp_path)
    taxonomy.create_taxonomy(store, ["build", "stale"])
    store.upsert_items([_item("o/r", 1, category="build", created_at="2026-05-25T00:00:00Z")])

    timeline = panels.taxonomy_timeline(store, now=_NOW)

    by_category = {e["category"]: e for e in timeline}
    assert by_category["build"]["retirement"] is None
    stale = by_category["stale"]["retirement"]
    assert stale is not None
    assert stale["weeks_inactive"] is None  # never active -> math.inf normalized to None
    assert stale["evidence"] is None


# --------------------------------------------------------------------- prediction_scoreboard


def test_prediction_scoreboard_buckets_by_graded_week(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    prediction = Prediction(
        claim="x",
        resolution_rule="r",
        prob=0.9,
        due_date="2026-01-02T00:00:00Z",
        evidence=(),
        created_at="2026-01-01T00:00:00Z",
    )
    record_prediction(store, prediction)
    grade = Grade(prediction=prediction, outcome=True, graded_at="2026-01-08T00:00:00Z")
    store.set_state("grade@1", grade.to_json())

    scoreboard = panels.prediction_scoreboard(store)

    assert scoreboard.keys() == {"2026-W02"}
    week = scoreboard["2026-W02"]
    assert week["precision"] == 1.0
    assert week["recall"] == 1.0
    assert week["n"] == 1
    assert week["brier"] == pytest.approx(0.01)


def test_prediction_scoreboard_no_grades_returns_empty_dict(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert panels.prediction_scoreboard(store) == {}


# --------------------------------------------------------------------- candidate_queue


def test_candidate_queue_defaults_to_empty_without_calling_the_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, labels=["good first issue"])])

    def _boom(*a, **k):
        raise AssertionError("llm.complete should not be called")

    monkeypatch.setattr(scout.llm, "complete", _boom)

    assert panels.candidate_queue(store) == []


def test_candidate_queue_computes_when_explicitly_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, labels=["good first issue"])])
    monkeypatch.setattr(
        scout.llm, "complete", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "high"}
    )

    found = panels.candidate_queue(store, compute=True)

    assert len(found) == 1
    assert found[0]["source"] == "good-first-issue"
