"""Tests for candidate selection (T5.10) — offline & deterministic.

Per the DEVPLAN todo: a select action writes a decision record; the engineer/orchestrator
query returns only selected candidates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import selection
from src.agents.scout import Candidate
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5


def _ts(date: str) -> datetime:
    return datetime.fromisoformat(date).replace(tzinfo=timezone.utc)


def _candidate(evidence: str, **overrides) -> Candidate:
    base = {
        "title": "x",
        "source": "good-first-issue",
        "risk": "low",
        "effort": "low",
        "impact": "high",
    }
    base.update(overrides)
    return Candidate(evidence=evidence, **base)


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


def test_record_decision_writes_a_run_and_returns_it(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)

    record = selection.record_decision(store, "o/r", 1, decision="selected", by="hankyu")

    assert record is not None
    assert record["repo"] == "o/r"
    assert record["number"] == 1
    assert record["decision"] == "selected"
    assert record["by"] == "hankyu"
    runs = store.list_runs(repo="o/r", number=1, stage="selection")
    assert len(runs) == 1
    assert runs[0]["decision"] == "selected"


def test_record_decision_rejects_an_invalid_value(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)
    with pytest.raises(selection.SelectionError, match="decision must be one of"):
        selection.record_decision(store, "o/r", 1, decision="maybe")


def test_record_decision_returns_none_for_a_nonexistent_item(tmp_path: Path) -> None:
    """Regression: a code-review finding -- an earlier version wrote a decision for ANY
    (repo, number), even one never collected into the KB, creating a permanent orphan record.
    Mirrors src.gate's own get_item_or_skip precondition."""
    store = JsonlStore(tmp_path)

    assert selection.record_decision(store, "o/r", 1, decision="selected") is None
    assert store.list_runs(repo="o/r", number=1, stage="selection") == []


def test_latest_decision_none_before_any_decision(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)
    assert selection.latest_decision(store, "o/r", 1) is None


def test_latest_decision_reflects_a_changed_mind(tmp_path: Path) -> None:
    """A human can change their mind -- the most recently recorded decision wins, not the
    first."""
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)
    selection.record_decision(store, "o/r", 1, decision="selected", now=_ts("2026-01-01"))
    selection.record_decision(store, "o/r", 1, decision="skip", now=_ts("2026-01-02"))

    assert selection.latest_decision(store, "o/r", 1) == "skip"


def test_latest_decision_a_same_second_correction_wins_over_the_first_click(
    tmp_path: Path,
) -> None:
    """Regression, empirically reproduced during code review: a human clicking "Work this"
    then immediately "Skip" within the same wall-clock second (an ordinary UI correction, not
    a rare edge case) must resolve to the LATER click, not stages.latest_run's own
    first-tied-wins tie-break."""
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)
    same_second = _ts("2026-01-01")
    selection.record_decision(store, "o/r", 1, decision="selected", now=same_second)
    selection.record_decision(store, "o/r", 1, decision="skip", now=same_second)

    assert selection.latest_decision(store, "o/r", 1) == "skip"
    assert selection.is_selected(store, "o/r", 1) is False


def test_is_selected_true_only_when_the_latest_decision_is_selected(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)
    assert selection.is_selected(store, "o/r", 1) is False

    selection.record_decision(store, "o/r", 1, decision="selected", now=_ts("2026-01-01"))
    assert selection.is_selected(store, "o/r", 1) is True

    selection.record_decision(store, "o/r", 1, decision="skip", now=_ts("2026-01-02"))
    assert selection.is_selected(store, "o/r", 1) is False


def test_filter_selected_returns_only_selected_candidates(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)
    _seed_item(store, "o/r", 2)
    selection.record_decision(store, "o/r", 1, decision="selected")
    selection.record_decision(store, "o/r", 2, decision="skip")
    candidates = [
        _candidate("https://github.com/o/r/issues/1"),
        _candidate("https://github.com/o/r/issues/2"),
        _candidate("https://github.com/o/r/issues/3"),  # never decided
    ]

    result = selection.filter_selected(store, candidates)

    assert [c.evidence for c in result] == ["https://github.com/o/r/issues/1"]


def test_filter_selected_excludes_a_candidate_with_unparseable_evidence(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    candidates = [_candidate("not-a-github-url")]

    assert selection.filter_selected(store, candidates) == []


def test_filter_selected_fetches_runs_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression, confirmed by two independent finder angles: an earlier version called
    is_selected once per candidate, each issuing its own store.list_runs query -- N separate
    reads instead of one, the exact anti-pattern already found and fixed twice in this
    milestone (dashboard.pipeline_diagram, dashboard.health)."""
    store = JsonlStore(tmp_path)
    _seed_item(store, "o/r", 1)
    selection.record_decision(store, "o/r", 1, decision="selected")
    calls = []
    real_list_runs = store.list_runs

    def _counting_list_runs(*args, **kwargs):
        calls.append((args, kwargs))
        return real_list_runs(*args, **kwargs)

    monkeypatch.setattr(store, "list_runs", _counting_list_runs)
    candidates = [
        _candidate("https://github.com/o/r/issues/1"),
        _candidate("https://github.com/o/r/issues/2"),
        _candidate("https://github.com/o/r/issues/3"),
    ]

    selection.filter_selected(store, candidates)

    assert len(calls) == 1
    assert calls[0] == ((), {"stage": "selection"})
