"""Tests for the PR-quality gate (T3.10) — offline & deterministic; llm votes mocked.

Per the DEVPLAN todo: a thin/boilerplate body fails; a complete, evidence-backed one passes
(mock judges).
"""

from datetime import datetime, timezone

import pytest

from src import llm, pr_quality
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m3

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


_VERIFY_RECORDED_AT = "2025-12-31T00:00:00Z"


def _store_ready_for_quality(
    tmp_path,
    *,
    repo="o/r",
    number=1,
    pr_author_recorded_at="2026-01-01T00:00:00Z",
    pr_author_verify_recorded_at=_VERIFY_RECORDED_AT,
    title="[Bugfix] Fix fp8 assertion on gfx90a",
    body="## Problem\n...\n\nFixes #1",
):
    """A store where (repo, number) is gate-ready (verify+self_review advanced, matching
    tests/test_gate.py's own fixture) AND already has a composed `stage="pr_author"` run --
    the two preconditions `run_pr_quality` requires."""
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
                "url": f"https://github.com/{repo}/issues/{number}",
            }
        ]
    )
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "verify",
            "branch": f"forager/{repo.replace('/', '-')}-{number}",
            "patch": "--- a/x.py\n+++ b/x.py\n",
            "log": "1 passed\n",
            "verified": True,
            "recorded_at": _VERIFY_RECORDED_AT,
        }
    )
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "self_review",
            "critiques": [{"looks_correct": True, "reason": "ok"}],
            "approve_count": 4,
            "total_votes": 5,
            "advance": True,
            "verify_recorded_at": _VERIFY_RECORDED_AT,
            "recorded_at": "2025-12-31T12:00:00Z",
        }
    )
    if pr_author_recorded_at is not None:
        store.record_run(
            {
                "repo": repo,
                "number": number,
                "stage": "pr_author",
                "title": title,
                "body": body,
                "verify_recorded_at": pr_author_verify_recorded_at,
                "recorded_at": pr_author_recorded_at,
            }
        )
    return store


def _votes(*bools):
    replies = iter({"acceptable": b, "reason": "because"} for b in bools)

    def _complete(*a, **k):
        return next(replies)

    return _complete


def _fail_if_llm_called(*a, **k):
    raise AssertionError("llm.complete should not be called")


@pytest.fixture(autouse=True)
def _no_profile_fetch(monkeypatch: pytest.MonkeyPatch):
    """`get_profile` hits live GitHub on a cache miss -- keep every test in this file offline by
    defaulting to no profile unless a test overrides `pr_quality.get_profile` (or passes
    `profile=`) itself."""
    monkeypatch.setattr(pr_quality, "get_profile", lambda repo, **k: None)


# --------------------------------------------------------------------- run_pr_quality: voting


def test_majority_acceptable_passes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_quality(tmp_path)
    monkeypatch.setattr(llm, "complete", _votes(True, True, True, True, False))

    result = pr_quality.run_pr_quality(store, "o/r", 1, now=_NOW)

    assert result is not None
    assert result.approve_count == 4
    assert result.total_votes == 5
    assert result.passes is True
    assert result.pr_author_recorded_at == "2026-01-01T00:00:00Z"

    runs = store.list_runs(repo="o/r", number=1, stage="pr_quality")
    assert len(runs) == 1
    assert runs[0]["passes"] is True


def test_mostly_reject_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_quality(tmp_path)
    monkeypatch.setattr(llm, "complete", _votes(False, False, False, False, True))

    result = pr_quality.run_pr_quality(store, "o/r", 1, now=_NOW)

    assert result.passes is False
    assert result.approve_count == 1


def test_unanimous_acceptable_passes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_quality(tmp_path)
    monkeypatch.setattr(llm, "complete", _votes(True, True, True, True, True))

    result = pr_quality.run_pr_quality(store, "o/r", 1, now=_NOW)

    assert result.passes is True
    assert result.approve_count == 5


def test_all_judges_failing_never_passes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero votes cast must never be treated as unanimous approval (fails safe)."""
    store = _store_ready_for_quality(tmp_path)

    def _raise(*a, **k):
        raise llm.LLMError("provider unavailable")

    monkeypatch.setattr(llm, "complete", _raise)

    result = pr_quality.run_pr_quality(store, "o/r", 1, now=_NOW)

    assert result is not None
    assert result.total_votes == 0
    assert result.approve_count == 0
    assert result.passes is False


def test_threshold_is_overridable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_quality(tmp_path)
    monkeypatch.setattr(llm, "complete", _votes(True, True, True, False, False))

    result = pr_quality.run_pr_quality(store, "o/r", 1, threshold=0.5, now=_NOW)

    assert result.passes is True


def test_a_failed_judge_is_excluded_not_counted_as_reject(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_quality(tmp_path)
    replies = iter(
        [
            {"acceptable": True, "reason": "ok"},
            {"acceptable": True, "reason": "ok"},
            {"acceptable": True, "reason": "ok"},
            {"acceptable": True, "reason": "ok"},
        ]
    )
    calls = {"n": 0}

    def _complete(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise llm.LLMError("transient failure")
        return next(replies)

    monkeypatch.setattr(llm, "complete", _complete)

    result = pr_quality.run_pr_quality(store, "o/r", 1, n=5, now=_NOW)

    assert result.total_votes == 4
    assert result.approve_count == 4
    assert result.passes is True


# --------------------------------------------------------------------- run_pr_quality: skip paths


def test_returns_none_when_no_pr_author_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_quality(tmp_path, pr_author_recorded_at=None)
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)

    assert pr_quality.run_pr_quality(store, "o/r", 1) is None


def test_returns_none_when_not_gate_ready(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pr_author run exists, but the candidate regressed (no advancing self-review) --
    pr_quality must not judge a narrative for a candidate that's no longer gate-ready."""
    store = JsonlStore(tmp_path)
    store.upsert_items([{"repo": "o/r", "number": 1, "type": "issue", "title": "t"}])
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "t",
            "body": "b",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)

    assert pr_quality.run_pr_quality(store, "o/r", 1) is None


def test_ignores_stale_pr_author_run_recorded_at(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`latest_run` must pick the most recent `stage=\"pr_author\"` record, not an older one --
    both composed against the same (current) verify run here, so this exercises only the
    latest-run selection, not the separate verify_recorded_at staleness guard below."""
    store = _store_ready_for_quality(tmp_path, pr_author_recorded_at="2025-01-01T00:00:00Z")
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "newer title",
            "body": "newer body",
            "verify_recorded_at": _VERIFY_RECORDED_AT,
            "recorded_at": "2026-06-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(llm, "complete", _votes(True, True, True, True, True))

    result = pr_quality.run_pr_quality(store, "o/r", 1, now=_NOW)

    assert result is not None
    assert result.pr_author_recorded_at == "2026-06-01T00:00:00Z"


def test_returns_none_when_pr_author_composed_against_a_different_verify_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: a candidate re-verified (e.g. a follow-up fix) after T3.9 composed its
    narrative must not have that stale narrative judged against the new diff."""
    store = _store_ready_for_quality(tmp_path, pr_author_verify_recorded_at="2025-01-01T00:00:00Z")
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)

    assert pr_quality.run_pr_quality(store, "o/r", 1) is None


def test_judges_when_pr_author_verify_recorded_at_matches_current(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_quality(tmp_path)  # default already matches _VERIFY_RECORDED_AT
    monkeypatch.setattr(llm, "complete", _votes(True, True, True, True, True))

    result = pr_quality.run_pr_quality(store, "o/r", 1, now=_NOW)

    assert result is not None
    assert result.passes is True


def test_raises_for_n_less_than_one(tmp_path) -> None:
    store = _store_ready_for_quality(tmp_path)
    with pytest.raises(pr_quality.PRQualityError, match=r"n must be >= 1"):
        pr_quality.run_pr_quality(store, "o/r", 1, n=0)


# --------------------------------------------------------------------- _judge


def test_judge_rejects_non_dict_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "not a dict")
    assert pr_quality._judge("t", "b", "diff", None) is None


def test_judge_rejects_missing_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"acceptable": True})
    assert pr_quality._judge("t", "b", "diff", None) is None


def test_judge_rejects_non_bool_acceptable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"acceptable": "yes", "reason": "because"})
    assert pr_quality._judge("t", "b", "diff", None) is None


def test_judge_rejects_blank_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"acceptable": True, "reason": "   "})
    assert pr_quality._judge("t", "b", "diff", None) is None


def test_judge_strips_reason_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"acceptable": True, "reason": "  ok  "})
    vote = pr_quality._judge("t", "b", "diff", None)
    assert vote is not None
    assert vote.reason == "ok"


# --------------------------------------------------------------------- _profile_context


def test_profile_context_handles_none() -> None:
    assert "No contribution-norms profile" in pr_quality._profile_context(None)


def test_profile_context_includes_profile_fields() -> None:
    from src.pr_profile import RepoProfile

    profile = RepoProfile(
        repo="o/r",
        contributing="Please sign your commits per our DCO policy.",
        pr_template="## Description\n",
        exemplars=({"title": "[Bugfix] fix x", "url": "u", "body": "b"},) * 3,
    )
    context = pr_quality._profile_context(profile)
    assert "DCO/Signed-off-by required: True" in context
    assert "Please sign your commits per our DCO policy." in context
