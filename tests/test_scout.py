"""Tests for the Scout agent (T2.5) — offline & deterministic.

Per the DEVPLAN todo: fixture items → every ranked candidate carries risk, effort, and impact
scores + evidence present; ranking reflects the combination of all three, not risk alone (a
low-risk/low-impact item doesn't outrank a medium-risk/high-impact one).
"""

import pytest

from src import llm, parity
from src.agents import scout
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m2


def _item(repo: str, number: int, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": "x",
        "state": "open",
        "labels": [],
        "url": f"https://github.com/{repo}/issues/{number}",
    }
    rec.update(overrides)
    return rec


def _reply(risk="low", effort="low", impact="low") -> dict:
    return {"risk": risk, "effort": effort, "impact": impact}


# --------------------------------------------------------------------- discover_candidates


def test_discover_candidates_from_parity_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    gap = parity.Gap(capability="fp8-kv-cache", evidence="https://github.com/ROCm/vllm/pull/1")

    candidates = scout.discover_candidates([], [gap])

    assert len(candidates) == 1
    assert candidates[0].source == "parity-gap"
    assert candidates[0].evidence == "https://github.com/ROCm/vllm/pull/1"
    assert "fp8-kv-cache" in candidates[0].title


def test_discover_candidates_good_first_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="fix typo", labels=["good first issue"])]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].source == "good-first-issue"
    assert candidates[0].evidence == "https://github.com/o/r/issues/1"


def test_discover_candidates_rocm_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="hipBLAS build fails on gfx90a")]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].source == "rocm-reproducible"


def test_discover_candidates_ignores_closed_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="hipBLAS bug on gfx90a", state="closed")]

    assert scout.discover_candidates(items) == []


def test_discover_candidates_ignores_prs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="hipBLAS fix on gfx90a", type="pr")]

    assert scout.discover_candidates(items) == []


def test_discover_candidates_ignores_unrelated_open_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="typo in docs")]

    assert scout.discover_candidates(items) == []


def test_discover_candidates_dedupes_across_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """An item matching both good-first-issue AND rocm-reproducible is only counted once."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    items = [_item("o/r", 1, title="hipBLAS bug on gfx90a", labels=["good first issue"])]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].source == "good-first-issue"  # checked first


def test_discover_candidates_ranking_reflects_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEVPLAN's named scenario: a low-risk/low-impact item doesn't outrank a
    medium-risk/high-impact one."""

    def fake_complete(prompt: str, **kwargs) -> dict:
        if "low-impact-item" in prompt:
            return _reply(risk="low", effort="low", impact="low")
        return _reply(risk="medium", effort="high", impact="high")

    monkeypatch.setattr(llm, "complete", fake_complete)
    items = [
        _item("o/r", 1, title="low-impact-item hipBLAS", labels=["good first issue"]),
        _item("o/r", 2, title="high-impact-item hipBLAS", labels=["good first issue"]),
    ]

    candidates = scout.discover_candidates(items)

    assert [c.title for c in candidates] == ["high-impact-item hipBLAS", "low-impact-item hipBLAS"]


def test_discover_candidates_skips_llm_failure_keeps_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def flaky_complete(prompt: str, **kwargs) -> dict:
        if "bad" in prompt:
            raise llm.LLMError("simulated transient failure")
        return _reply()

    monkeypatch.setattr(llm, "complete", flaky_complete)
    items = [
        _item("o/r", 1, title="bad hipBLAS issue", labels=["good first issue"]),
        _item("o/r", 2, title="good hipBLAS issue", labels=["good first issue"]),
    ]

    candidates = scout.discover_candidates(items)

    assert len(candidates) == 1
    assert candidates[0].title == "good hipBLAS issue"


def test_discover_candidates_skips_malformed_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"risk": "extreme"})
    items = [_item("o/r", 1, title="hipBLAS bug", labels=["good first issue"])]

    assert scout.discover_candidates(items) == []


def test_discover_candidates_empty_returns_empty() -> None:
    assert scout.discover_candidates([]) == []


# --------------------------------------------------------------------- Candidate/priority


def test_candidate_rejects_invalid_level() -> None:
    with pytest.raises(scout.ScoutError, match="risk must be one of"):
        scout.Candidate(
            title="x", source="parity-gap", risk="extreme", effort="low", impact="low", evidence=""
        )


def test_priority_low_risk_low_impact_never_outranks_medium_risk_high_impact() -> None:
    low = scout.Candidate(
        title="a", source="x", risk="low", effort="low", impact="low", evidence=""
    )
    for effort in ("low", "medium", "high"):
        medium_high = scout.Candidate(
            title="b", source="x", risk="medium", effort=effort, impact="high", evidence=""
        )
        assert medium_high.priority > low.priority


# --------------------------------------------------------------------- discover_from_store


def test_discover_from_store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("ROCm/vllm", 1, title="hipBLAS bug on gfx90a", category="build"),
            _item(
                "ROCm/vllm",
                2,
                title="shipped fp8 kv-cache",
                type="pr",
                state="closed",
                category="quantization > FP8",
            ),
        ]
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())

    candidates = scout.discover_from_store(store)

    sources = {c.source for c in candidates}
    assert "rocm-reproducible" in sources
    assert "parity-gap" in sources
