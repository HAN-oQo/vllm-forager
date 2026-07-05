"""Tests for the ensemble self-review gate (T3.4) — offline & deterministic; llm votes mocked.

Per the DEVPLAN todo: mock llm votes -- majority-approve => advance; split/reject => hold
(no gate).
"""

from datetime import datetime, timezone

import pytest

from src import llm, self_review
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m3

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _store_with_verified_patch(tmp_path, *, verified=True, repo="o/r", number=1):
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            {
                "repo": repo,
                "number": number,
                "type": "issue",
                "title": "vLLM crashes on gfx90a with fp8",
                "body": "Running fp8 quant on MI250 raises an assertion.",
                "state": "open",
            }
        ]
    )
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "verify",
            "patch": "--- a/x.py\n+++ b/x.py\n",
            "log": "1 passed\n",
            "verified": verified,
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    return store


def _votes(*bools):
    """A `llm.complete` side_effect cycling through `bools`, one per call, as
    `{"looks_correct": bool, "reason": "..."}`."""
    replies = iter({"looks_correct": b, "reason": "because"} for b in bools)

    def _complete(*a, **k):
        return next(replies)

    return _complete


def _fail_if_llm_called(*a, **k):
    raise AssertionError("llm.complete should not be called")


# --------------------------------------------------------------------- run_self_review: voting


def test_majority_approve_advances(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DEVPLAN's own example: 5 critiques, 4 say correct -> advance."""
    store = _store_with_verified_patch(tmp_path)
    monkeypatch.setattr(llm, "complete", _votes(True, True, True, True, False))

    result = self_review.run_self_review(store, "o/r", 1, now=_NOW)

    assert result is not None
    assert result.approve_count == 4
    assert result.total_votes == 5
    assert result.advance is True
    assert result.verify_recorded_at == "2025-12-31T00:00:00Z"

    runs = store.list_runs(repo="o/r", number=1, stage="self_review")
    assert len(runs) == 1
    assert runs[0]["advance"] is True
    assert runs[0]["verify_recorded_at"] == "2025-12-31T00:00:00Z"


def test_split_vote_holds(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DEVPLAN's own example: a 3-2 split -> hold, don't gate (3/5 = 0.6 is a plain majority
    but below the 2/3 supermajority this module actually requires)."""
    store = _store_with_verified_patch(tmp_path)
    monkeypatch.setattr(llm, "complete", _votes(True, True, True, False, False))

    result = self_review.run_self_review(store, "o/r", 1, now=_NOW)

    assert result is not None
    assert result.approve_count == 3
    assert result.total_votes == 5
    assert result.advance is False
    assert store.list_runs(repo="o/r", number=1, stage="self_review")[0]["advance"] is False


def test_mostly_reject_holds(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_with_verified_patch(tmp_path)
    monkeypatch.setattr(llm, "complete", _votes(False, False, False, False, True))

    result = self_review.run_self_review(store, "o/r", 1, now=_NOW)

    assert result.advance is False
    assert result.approve_count == 1


def test_unanimous_approve_advances(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_with_verified_patch(tmp_path)
    monkeypatch.setattr(llm, "complete", _votes(True, True, True, True, True))

    result = self_review.run_self_review(store, "o/r", 1, now=_NOW)

    assert result.advance is True
    assert result.approve_count == 5


def test_all_critiques_failing_holds_not_advances(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero votes cast must never be treated as unanimous approval (fails safe)."""
    store = _store_with_verified_patch(tmp_path)

    def _raise(*a, **k):
        raise llm.LLMError("provider unavailable")

    monkeypatch.setattr(llm, "complete", _raise)

    result = self_review.run_self_review(store, "o/r", 1, now=_NOW)

    assert result is not None
    assert result.total_votes == 0
    assert result.approve_count == 0
    assert result.advance is False


def test_threshold_is_overridable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """3/5 holds against the default 2/3 threshold, but must advance against a looser one."""
    store = _store_with_verified_patch(tmp_path)
    monkeypatch.setattr(llm, "complete", _votes(True, True, True, False, False))

    result = self_review.run_self_review(store, "o/r", 1, threshold=0.5, now=_NOW)

    assert result.advance is True


def test_a_failed_critique_is_excluded_not_counted_as_reject(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One flaky call among five shouldn't silently tip a real 4-approve verdict into a
    recorded 4-of-4 (still a supermajority) rather than being misread as a rejection."""
    store = _store_with_verified_patch(tmp_path)
    replies = iter(
        [
            {"looks_correct": True, "reason": "ok"},
            {"looks_correct": True, "reason": "ok"},
            {"looks_correct": True, "reason": "ok"},
            {"looks_correct": True, "reason": "ok"},
        ]
    )
    calls = {"n": 0}

    def _complete(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise llm.LLMError("transient failure")
        return next(replies)

    monkeypatch.setattr(llm, "complete", _complete)

    result = self_review.run_self_review(store, "o/r", 1, n=5, now=_NOW)

    assert result.total_votes == 4  # one call failed and was excluded, not counted
    assert result.approve_count == 4
    assert result.advance is True


# --------------------------------------------------------------------- run_self_review: skip paths


def test_returns_none_when_no_verified_patch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_with_verified_patch(tmp_path, verified=False)
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)

    assert self_review.run_self_review(store, "o/r", 1) is None


def test_ignores_stale_verified_run_when_latest_is_not_verified(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a candidate verified successfully once, then regressed on a later retry --
    self-review must not fall back to the older success and must skip, since the CURRENT patch
    state is unverified."""
    store = _store_with_verified_patch(tmp_path)  # verified=True, recorded_at=2025-12-31
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "verify",
            "patch": "--- a/x.py\n+++ b/x.py\n(different, worse patch)\n",
            "log": "1 failed\n",
            "verified": False,
            "recorded_at": "2026-01-02T00:00:00Z",  # newer than the verified=True run
        }
    )
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)

    assert self_review.run_self_review(store, "o/r", 1) is None


def test_returns_none_when_no_verify_run_at_all(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([{"repo": "o/r", "number": 1, "type": "issue", "title": "t"}])
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)

    assert self_review.run_self_review(store, "o/r", 1) is None


def test_returns_none_when_item_missing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "verify",
            "verified": True,
            "patch": "x",
            "log": "x",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)

    assert self_review.run_self_review(store, "o/r", 1) is None


def test_raises_for_n_less_than_one(tmp_path) -> None:
    store = _store_with_verified_patch(tmp_path)
    with pytest.raises(self_review.SelfReviewError, match=r"n must be >= 1"):
        self_review.run_self_review(store, "o/r", 1, n=0)


# --------------------------------------------------------------------- _critique


def test_critique_rejects_non_dict_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "not a dict")
    assert self_review._critique("t", "b", "patch", "log") is None


def test_critique_rejects_missing_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"looks_correct": True})
    assert self_review._critique("t", "b", "patch", "log") is None


def test_critique_rejects_non_bool_looks_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm, "complete", lambda *a, **k: {"looks_correct": "yes", "reason": "because"}
    )
    assert self_review._critique("t", "b", "patch", "log") is None


def test_critique_rejects_blank_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """A degenerate empty/whitespace-only reason must be excluded, not silently counted as a
    real vote -- matching engineer.py/repro.py's own blank-string rejection for their single
    string field."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"looks_correct": True, "reason": "   "})
    assert self_review._critique("t", "b", "patch", "log") is None


def test_critique_strips_reason_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm, "complete", lambda *a, **k: {"looks_correct": True, "reason": "  ok  "}
    )
    critique = self_review._critique("t", "b", "patch", "log")
    assert critique is not None
    assert critique.reason == "ok"
