"""Tests for the dashboard guardrail panels (T5.7) — offline & deterministic.

Per the DEVPLAN todo: seeded metrics → panels return series + threshold flags.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard import guardrails
from src import audit, rag_eval
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5

_FAKE_REMOTE = lambda repo, since: {"total": 1}  # noqa: E731


# --------------------------------------------------------------------- data_quality_series


def test_data_quality_series_returns_recorded_checks_with_their_flags(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [{"repo": "o/r", "number": 1, "type": "issue", "updated_at": "2026-01-05T00:00:00Z"}]
    )
    audit.audit_repo(
        store,
        "o/r",
        "2025-01-01T00:00:00Z",
        remote_fetcher=_FAKE_REMOTE,
        checked_at="2026-01-06T00:00:00Z",
    )

    series = guardrails.data_quality_series(store)

    assert len(series) == 1
    assert series[0]["reason"] == "reconciliation"
    assert "flagged" in series[0]


def test_data_quality_series_empty_before_any_check_has_run(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert guardrails.data_quality_series(store) == []


# --------------------------------------------------------------------- rag_eval_series


def _score(**overrides) -> rag_eval.RagEvalScore:
    base = {
        "recall_at_k": 0.9,
        "mrr": 0.8,
        "ndcg_at_k": 0.85,
        "citation_accuracy": 1.0,
        "hallucination_rate": 0.0,
        "faithfulness": 1.0,
        "created_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return rag_eval.RagEvalScore(**base)


def test_rag_eval_series_includes_the_passed_flag(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    rag_eval.record_score(store, _score())
    rag_eval.record_score(store, _score(recall_at_k=0.5))  # below RECALL_THRESHOLD -> fails

    series = guardrails.rag_eval_series(store)

    assert [s["passed"] for s in series] == [True, False]


def test_rag_eval_series_empty_before_any_run_recorded(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert guardrails.rag_eval_series(store) == []
