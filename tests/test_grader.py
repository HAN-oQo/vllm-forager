"""Tests for the Grader agent (T2.1) — offline & deterministic.

Per the DEVPLAN todo: synthetic predictions + outcomes → known metric values. Also covers
maturity gating, the evidence-URL-to-item lookup, per-prediction failure isolation, the
grade/prediction 1:1 index correspondence, and the KB round-trip.
"""

import json
from datetime import datetime, timezone

import pytest

from src import llm
from src.agents import forecaster, grader
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m2

_NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _prediction(**overrides) -> forecaster.Prediction:
    base = {
        "claim": "this will be merged",
        "resolution_rule": "resolved true if the PR is merged",
        "prob": 0.7,
        "due_date": "2026-02-01T00:00:00Z",
        "evidence": ("https://github.com/o/r/pull/1",),
        "created_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return forecaster.Prediction(**base)


def _item(repo: str, number: int, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "pr",
        "title": "x",
        "state": "closed",
        "url": f"https://github.com/{repo}/pull/{number}",
    }
    rec.update(overrides)
    return rec


# --------------------------------------------------------------------- is_matured


def test_is_matured_true_after_due_date() -> None:
    p = _prediction(due_date="2026-02-01T00:00:00Z")
    assert grader.is_matured(p, now=_NOW)


def test_is_matured_false_before_due_date() -> None:
    p = _prediction(due_date="2026-06-01T00:00:00Z")
    assert not grader.is_matured(p, now=_NOW)


# --------------------------------------------------------------------- resolve_prediction


def test_resolve_prediction_not_matured_returns_none(tmp_path) -> None:
    p = _prediction(due_date="2026-06-01T00:00:00Z")
    assert grader.resolve_prediction(p, JsonlStore(tmp_path), now=_NOW) is None


def test_resolve_prediction_looks_up_evidence_item_by_url(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, state="closed", title="fix the bug")])
    seen_prompt = {}

    def fake_complete(prompt: str, **kwargs) -> dict:
        seen_prompt["text"] = prompt
        return {"outcome": True}

    monkeypatch.setattr(llm, "complete", fake_complete)

    grade = grader.resolve_prediction(_prediction(), store, now=_NOW)

    assert grade is not None
    assert grade.outcome is True
    assert grade.prediction.prob == 0.7
    assert "o/r#1" in seen_prompt["text"]
    assert "fix the bug" in seen_prompt["text"]
    assert "closed" in seen_prompt["text"]


def test_resolve_prediction_missing_evidence_item_still_asks_llm(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cited item no longer exists in the KB — still resolves (via the "none found" note
    in the prompt) rather than crashing; the model can only guess, but that's a quality
    concern, not a correctness one."""
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"outcome": False})

    grade = grader.resolve_prediction(_prediction(), store, now=_NOW)

    assert grade is not None
    assert grade.outcome is False


def test_resolve_prediction_non_github_evidence_url_is_skipped_not_crashed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"outcome": True})
    p = _prediction(evidence=("not-a-github-url",))

    grade = grader.resolve_prediction(p, store, now=_NOW)

    assert grade is not None


def test_resolve_prediction_non_boolean_outcome_raises(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"outcome": "yes"})
    with pytest.raises(grader.GradeError, match="expected a boolean"):
        grader.resolve_prediction(_prediction(), store, now=_NOW)


def test_resolve_prediction_missing_outcome_raises(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {})
    with pytest.raises(grader.GradeError, match="expected a boolean"):
        grader.resolve_prediction(_prediction(), store, now=_NOW)


# --------------------------------------------------------------------- Grade json roundtrip


def test_grade_json_roundtrip() -> None:
    prediction = _prediction()
    grade = grader.Grade(prediction=prediction, outcome=True, graded_at="2026-03-01T00:00:00Z")
    assert grader.Grade.from_json(grade.to_json(), prediction) == grade


def test_grade_to_json_does_not_reembed_the_prediction() -> None:
    """Regression: to_json used to nest a full copy of the prediction's own fields — a second
    copy of every prediction's data in the state map, on top of prediction@<index>."""
    grade = grader.Grade(prediction=_prediction(), outcome=True, graded_at="2026-03-01T00:00:00Z")
    assert set(json.loads(grade.to_json())) == {"outcome", "graded_at"}


def test_grade_from_json_malformed_raises() -> None:
    with pytest.raises(grader.GradeError, match="corrupt grade record"):
        grader.Grade.from_json("not valid json", _prediction())


def test_grade_from_json_missing_field_raises() -> None:
    with pytest.raises(grader.GradeError, match="corrupt grade record"):
        grader.Grade.from_json('{"foo": true}', _prediction())


def test_grade_construction_rejects_malformed_graded_at() -> None:
    with pytest.raises(grader.GradeError, match="not a valid"):
        grader.Grade(prediction=_prediction(), outcome=True, graded_at="not-a-date")


def test_grade_brier_term() -> None:
    grade = grader.Grade(
        prediction=_prediction(prob=0.7), outcome=True, graded_at="2026-03-01T00:00:00Z"
    )
    assert grade.brier_term == pytest.approx((0.7 - 1.0) ** 2)
    grade2 = grader.Grade(
        prediction=_prediction(prob=0.7), outcome=False, graded_at="2026-03-01T00:00:00Z"
    )
    assert grade2.brier_term == pytest.approx((0.7 - 0.0) ** 2)


# --------------------------------------------------------------------- grade_store


def test_grade_store_grades_matured_and_skips_unmatured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    forecaster.record_prediction(store, _prediction(due_date="2026-02-01T00:00:00Z"))  # matured
    forecaster.record_prediction(store, _prediction(due_date="2026-06-01T00:00:00Z"))  # not yet
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"outcome": True})

    graded = grader.grade_store(store, now=_NOW)

    assert len(graded) == 1
    assert grader.list_grades(store) == graded


def test_grade_store_skips_already_graded(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlStore(tmp_path)
    forecaster.record_prediction(store, _prediction())
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"outcome": True})

    first = grader.grade_store(store, now=_NOW)
    second = grader.grade_store(store, now=_NOW)

    assert len(first) == 1
    assert second == []
    assert len(grader.list_grades(store)) == 1


def test_grade_store_skips_failing_prediction_and_persists_the_rest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    forecaster.record_prediction(store, _prediction(claim="good"))
    forecaster.record_prediction(store, _prediction(claim="bad"))

    def flaky_complete(prompt: str, **kwargs) -> dict:
        if "bad" in prompt:
            raise llm.LLMError("simulated transient failure")
        return {"outcome": True}

    monkeypatch.setattr(llm, "complete", flaky_complete)

    graded = grader.grade_store(store, now=_NOW)

    assert len(graded) == 1
    assert graded[0].prediction.claim == "good"
    # the failing one is left pending, not permanently skipped
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"outcome": False})
    graded_again = grader.grade_store(store, now=_NOW)
    assert len(graded_again) == 1
    assert graded_again[0].prediction.claim == "bad"


def test_grade_store_no_predictions_returns_empty(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    assert grader.grade_store(store, now=_NOW) == []


# --------------------------------------------------------------------- compute_metrics


def test_compute_metrics_empty_returns_zeros() -> None:
    metrics = grader.compute_metrics([])
    assert metrics == grader.GradeMetrics(precision=0.0, recall=0.0, brier=0.0, n=0)


def test_compute_metrics_known_values() -> None:
    ts = "2026-03-01T00:00:00Z"
    grades = [
        grader.Grade(prediction=_prediction(prob=0.9), outcome=True, graded_at=ts),  # TP
        grader.Grade(prediction=_prediction(prob=0.8), outcome=False, graded_at=ts),  # FP
        grader.Grade(prediction=_prediction(prob=0.2), outcome=False, graded_at=ts),  # TN
        grader.Grade(prediction=_prediction(prob=0.3), outcome=True, graded_at=ts),  # FN
    ]

    metrics = grader.compute_metrics(grades)

    # precision = TP/(TP+FP) = 1/2; recall = TP/(TP+FN) = 1/2
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    expected_brier = ((0.9 - 1) ** 2 + (0.8 - 0) ** 2 + (0.2 - 0) ** 2 + (0.3 - 1) ** 2) / 4
    assert metrics.brier == pytest.approx(expected_brier)
    assert metrics.n == 4


def test_compute_metrics_no_positive_predictions_precision_is_zero_not_nan() -> None:
    ts = "2026-03-01T00:00:00Z"
    grades = [grader.Grade(prediction=_prediction(prob=0.1), outcome=False, graded_at=ts)]
    metrics = grader.compute_metrics(grades)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
