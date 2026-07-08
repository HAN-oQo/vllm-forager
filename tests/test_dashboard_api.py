"""Tests for the dashboard read layer (T5.1) — offline & deterministic.

Per the DEVPLAN todo: a seeded store → each endpoint returns the expected shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dashboard import api
from src import parity as parity_module
from src.agents import scout
from src.agents.forecaster import Prediction, record_prediction
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5


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


def _seeded_store(tmp_path: Path) -> JsonlStore:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, path=["cat"], category="cat", created_at="2026-01-05T00:00:00Z"),
            _item("o/r", 2, path=["cat"], category="cat", created_at="2026-01-06T00:00:00Z"),
        ]
    )
    return store


# --------------------------------------------------------------------- items / tree


def test_items_returns_a_page_and_the_true_total(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    page, total = api.items(store, ("cat",), limit=1)

    assert total == 2
    assert len(page) == 1


def test_items_defaults_to_the_other_bucket(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1)])  # no `path` -> Other

    page, total = api.items(store)

    assert total == 1
    assert page[0]["number"] == 1


def test_tree_returns_capped_nodes_with_counts(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    nodes = api.tree(store)

    assert len(nodes) == 1
    assert nodes[0]["name"] == "cat"
    assert nodes[0]["count"] == 2


def test_tree_is_unaffected_by_a_corrupt_prediction_record(tmp_path: Path) -> None:
    """Regression: an earlier version of tree() routed through build_snapshot(), which also
    reads the prediction log -- a corrupt prediction_count state value must not crash a
    tree-only read that has nothing to do with predictions."""
    store = _seeded_store(tmp_path)
    store.set_state("prediction_count", "not-a-number")

    nodes = api.tree(store)  # must not raise

    assert len(nodes) == 1


# --------------------------------------------------------------------- trends


def test_trends_returns_the_full_series(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    series = api.trends(store)

    assert series == {"cat": {"2026-W02": 2}}


def test_trends_for_category_matches_the_devplan_worked_example(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    assert api.trends(store, "cat") == {"2026-W02": 2}


def test_trends_for_category_with_no_activity_returns_empty_dict(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    assert api.trends(store, "quantization") == {}


# --------------------------------------------------------------------- predictions


_PREDICTION = Prediction(
    claim="PR #3 will merge by end of Q1",
    resolution_rule="merged into main",
    prob=0.7,
    due_date="2026-03-31T00:00:00Z",
    evidence=("https://github.com/o/r/pull/3",),
    created_at="2026-01-15T00:00:00Z",
)


def test_predictions_returns_recorded_predictions_as_dicts(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    record_prediction(store, _PREDICTION)

    predictions = api.predictions(store)

    assert predictions == [
        {
            "claim": "PR #3 will merge by end of Q1",
            "resolution_rule": "merged into main",
            "prob": 0.7,
            "due_date": "2026-03-31T00:00:00Z",
            "evidence": ("https://github.com/o/r/pull/3",),
            "created_at": "2026-01-15T00:00:00Z",
        }
    ]


def test_predictions_pagination(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    for i in range(3):
        record_prediction(
            store,
            Prediction(
                claim=f"claim {i}",
                resolution_rule="r",
                prob=0.5,
                due_date="2026-03-31T00:00:00Z",
                evidence=(),
                created_at="2026-01-01T00:00:00Z",
            ),
        )

    page = api.predictions(store, offset=1, limit=1)

    assert [p["claim"] for p in page] == ["claim 1"]


def test_predictions_no_predictions_returns_empty_list(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert api.predictions(store) == []


# --------------------------------------------------------------------- candidates


def test_candidates_defaults_to_empty_without_calling_the_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dashboard page view must never silently trigger LLM spend -- compute=False (the
    default) never even imports/calls the scoring path."""
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, labels=["good first issue"])])

    def _boom(*a, **k):
        raise AssertionError("llm.complete should not be called")

    monkeypatch.setattr(scout.llm, "complete", _boom)

    assert api.candidates(store) == []


def test_candidates_computes_when_explicitly_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, labels=["good first issue"])])
    monkeypatch.setattr(
        scout.llm, "complete", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "high"}
    )

    found = api.candidates(store, compute=True)

    assert len(found) == 1
    assert found[0]["source"] == "good-first-issue"
    assert (
        found[0]["priority"]
        == scout.Candidate(
            title="item 1",
            source="good-first-issue",
            risk="low",
            effort="low",
            impact="high",
            evidence="http://x/o/r/1",
        ).priority
    )


# --------------------------------------------------------------------- parity


def test_parity_returns_cells_and_a_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        parity_module.config,
        "REPOS",
        [
            {"slug": "vllm-project/vllm", "role": "primary", "domain": "speech"},
            {"slug": "other/engine", "role": "source", "domain": "speech"},
        ],
    )
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            # shipped on the source engine, not on the primary target -> a gap
            _item("other/engine", 1, type="pr", state="closed", category="fp8"),
        ]
    )

    result = api.parity(store)

    assert {k: v for k, v in result["cells"][0].items() if k != "evidence"} == {
        "engine": "other/engine",
        "capability": "fp8",
        "present": True,
    }
    assert len(result["gaps"]) == 1
    assert result["gaps"][0]["target_engine"] == "vllm-project/vllm"
    assert result["gaps"][0]["source_engine"] == "other/engine"


def test_parity_degrades_to_no_gaps_without_a_primary_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        parity_module.config, "REPOS", [{"slug": "o/r", "role": "source", "domain": "speech"}]
    )
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, type="pr", state="closed", category="fp8")])

    result = api.parity(store)  # must not raise

    assert result["gaps"] == []
    assert len(result["cells"]) == 1
    assert "skipping parity gaps" in capsys.readouterr().err


# --------------------------------------------------------------------- runs


def test_runs_filters_by_stage(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run({"stage": "collect", "status": "ok"})
    store.record_run({"stage": "intel", "status": "ok"})

    assert [r["stage"] for r in api.runs(store, stage="collect")] == ["collect"]


def test_runs_with_no_filters_returns_everything(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run({"stage": "collect", "status": "ok"})
    store.record_run({"stage": "intel", "status": "ok"})

    assert len(api.runs(store)) == 2


# --------------------------------------------------------------------- stage_status


_NOW = datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)


def test_stage_status_classifies_a_recent_run_as_ok(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run({"stage": "collect", "status": "ok", "recorded_at": "2026-01-08T11:59:00Z"})

    status = api.stage_status(store, "collect", now=_NOW, stale_after_s=600)

    assert status == "ok"


def test_stage_status_classifies_a_stale_heartbeat_as_stalled(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(
        {"stage": "intel", "status": "heartbeat", "recorded_at": "2026-01-08T11:00:00Z"}
    )

    status = api.stage_status(store, "intel", now=_NOW, stale_after_s=600)

    assert status == "stalled"


def test_stage_status_never_run(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)

    assert api.stage_status(store, "collect", now=_NOW, stale_after_s=600) == "never run"
