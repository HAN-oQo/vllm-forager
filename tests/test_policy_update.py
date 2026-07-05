"""Tests for policy update from grades (T2.2) — offline & deterministic.

Per the DEVPLAN todo: a low-precision category → its weight decreases in the new version; a
reliable one gains. Also covers category attribution via a grade's evidence, grouping,
un-attributable grades, and the KB round-trip through `policy.get_active`.
"""

from datetime import datetime, timezone

import pytest

from src import llm, policy, taxonomy
from src.agents import forecaster, grader, policy_update
from src.store.jsonl_store import JsonlStore

_NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)

pytestmark = pytest.mark.m2

_TS = "2026-03-01T00:00:00Z"


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


def _prediction(url: str, **overrides) -> forecaster.Prediction:
    base = {
        "claim": "this will be merged",
        "resolution_rule": "resolved true if merged",
        "prob": 0.9,
        "due_date": "2026-02-01T00:00:00Z",
        "evidence": (url,),
        "created_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return forecaster.Prediction(**base)


def _grade(url: str, *, outcome: bool, prob: float = 0.9) -> grader.Grade:
    return grader.Grade(prediction=_prediction(url, prob=prob), outcome=outcome, graded_at=_TS)


def _seeded_store(tmp_path) -> JsonlStore:
    store = JsonlStore(tmp_path)
    taxonomy.create_taxonomy(store, ["build", "quantization"])
    policy.create_policy(
        store,
        scoring_weights={"build": 1.0, "quantization": 1.0},
        prompt_templates={},
        active_taxonomy_version=1,
    )
    return store


# --------------------------------------------------------------------- grade_category


def test_grade_category_resolves_via_evidence_item(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, category="build")])
    grade = _grade("https://github.com/o/r/pull/1", outcome=True)

    assert policy_update.grade_category(grade, store) == "build"


def test_grade_category_none_when_item_missing(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    grade = _grade("https://github.com/o/r/pull/1", outcome=True)

    assert policy_update.grade_category(grade, store) is None


def test_grade_category_none_when_item_uncategorized(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1)])  # no category field
    grade = _grade("https://github.com/o/r/pull/1", outcome=True)

    assert policy_update.grade_category(grade, store) is None


# --------------------------------------------------------------------- group_by_category


def test_group_by_category_buckets_and_drops_unattributable(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, category="build"),
            _item("o/r", 2, category="quantization"),
        ]
    )
    g1 = _grade("https://github.com/o/r/pull/1", outcome=True)
    g2 = _grade("https://github.com/o/r/pull/2", outcome=False)
    g3 = _grade("https://github.com/o/r/pull/999", outcome=True)  # unresolvable

    grouped = policy_update.group_by_category([g1, g2, g3], store)

    assert grouped == {"build": [g1], "quantization": [g2]}


# --------------------------------------------------------------------- propose_scoring_weights


def test_propose_scoring_weights_low_precision_decreases(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", n, category="build") for n in range(1, 11)])
    # predicted-true (prob=0.9) for all 10; only 3 actually resolved true -> precision 0.3
    grades = [_grade(f"https://github.com/o/r/pull/{n}", outcome=(n <= 3)) for n in range(1, 11)]

    updated = policy_update.propose_scoring_weights(grades, store, {"build": 1.0})

    # EMA: (1 - 0.5) * 1.0 + 0.5 * 0.3 = 0.65 — down from the prior 1.0
    assert updated["build"] == pytest.approx(0.65)


def test_propose_scoring_weights_high_precision_increases(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", n, category="quantization") for n in range(1, 11)])
    # 9/10 predicted-true resolve true -> precision 0.9
    grades = [_grade(f"https://github.com/o/r/pull/{n}", outcome=(n <= 9)) for n in range(1, 11)]

    updated = policy_update.propose_scoring_weights(grades, store, {"quantization": 0.5})

    # EMA: (1 - 0.5) * 0.5 + 0.5 * 0.9 = 0.7 — up from the prior 0.5
    assert updated["quantization"] == pytest.approx(0.7)


def test_propose_scoring_weights_new_category_blends_from_default(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, category="new-cat")])
    grades = [_grade("https://github.com/o/r/pull/1", outcome=True)]

    updated = policy_update.propose_scoring_weights(grades, store, {})

    # precision = 1.0; no prior weight -> defaults to 1.0 -> EMA blend stays 1.0
    assert updated["new-cat"] == pytest.approx(1.0)


def test_propose_scoring_weights_ungraded_category_gets_no_entry(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, category="build")])
    grades = [_grade("https://github.com/o/r/pull/1", outcome=True)]

    updated = policy_update.propose_scoring_weights(
        grades, store, {"build": 1.0, "quantization": 1.0}
    )

    assert "quantization" not in updated


def test_propose_scoring_weights_empty_grades_returns_empty(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    assert policy_update.propose_scoring_weights([], store, {"build": 1.0}) == {}


# --------------------------------------------------------------------- update_policy_from_grades


def test_update_policy_from_grades_applies_new_version(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    store.upsert_items([_item("o/r", n, category="build") for n in range(1, 11)])
    for n in range(1, 11):
        forecaster.record_prediction(store, _prediction(f"https://github.com/o/r/pull/{n}"))

    def fake_complete(prompt: str, **kwargs) -> dict:
        # the resolution prompt cites "o/r#N" — outcome true for the first 3 -> precision 0.3
        n = int(prompt.split("o/r#")[1].split(" ")[0])
        return {"outcome": n <= 3}

    monkeypatch.setattr(llm, "complete", fake_complete)
    grader.grade_store(store, now=_NOW)

    updated = policy_update.update_policy_from_grades(store)

    assert updated is not None
    assert updated.version == 2
    assert updated.scoring_weights["build"] == pytest.approx(0.65)
    assert updated.scoring_weights["quantization"] == pytest.approx(1.0)  # carried over
    assert policy.get_active(store).version == 2


def test_update_policy_from_grades_returns_none_when_nothing_attributable(tmp_path) -> None:
    store = _seeded_store(tmp_path)
    # a grade whose evidence item has no category
    store.upsert_items([_item("o/r", 1)])
    forecaster.record_prediction(store, _prediction("https://github.com/o/r/pull/1"))
    grade = grader.Grade(
        prediction=_prediction("https://github.com/o/r/pull/1"), outcome=True, graded_at=_TS
    )
    store.set_state("grade@1", grade.to_json())

    assert policy_update.update_policy_from_grades(store) is None
    assert policy.get_active(store).version == 1  # unchanged


def test_update_policy_from_grades_raises_if_no_policy_exists(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    with pytest.raises(policy.PolicyError, match="no policy exists yet"):
        policy_update.update_policy_from_grades(store)
