"""Tests for the Upstream PRs tracker (T5.14) — offline & deterministic; `gh`/subprocess mocked.

Per the DEVPLAN todo: seeded stage="gate" (submitted) + stage="review_response"/review_post
records, gh mocked -> the tab lists only open real PRs, correctly flags un-responded comments
as outstanding, and renders live check status.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from src import upstream_prs
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5


def _submit_gate_run(
    store: JsonlStore, *, repo="o/r", number=1, pr_url="https://github.com/o/r/pull/1"
) -> None:
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "gate",
            "approved": True,
            "submitted": True,
            "pr_url": pr_url,
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )


def _gh_comment(comment_id, author, body="hi"):
    return {"id": comment_id, "user": {"login": author}, "body": body}


def _fake_gh(*, pr_json, comments=None, login="forager-bot"):
    comments = comments if comments is not None else []

    def _run(cmd, **kwargs):
        if cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(pr_json))
        if cmd[:2] == ["gh", "api"] and cmd[2] == "user":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{login}\n")
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/comments"):
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(comments))
        raise AssertionError(f"unexpected gh invocation: {cmd}")

    return _run


_OPEN_PR_JSON = {
    "url": "https://github.com/o/r/pull/1",
    "state": "OPEN",
    "mergeable": "MERGEABLE",
    "reviewDecision": "REVIEW_REQUIRED",
    "updatedAt": "2026-01-05T00:00:00Z",
    "statusCheckRollup": [{"conclusion": "SUCCESS", "status": "COMPLETED"}],
    "reviewRequests": [{"login": "octocat"}, {"name": "vllm-maintainers"}],
}


# --------------------------------------------------------------------- find_submitted_candidates


def test_find_submitted_candidates_returns_only_submitted_with_pr_url(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    _submit_gate_run(store, repo="o/r", number=1)
    store.record_run(
        {
            "repo": "o/r",
            "number": 2,
            "stage": "gate",
            "approved": True,
            "submitted": False,
            "pr_url": None,
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )

    assert upstream_prs.find_submitted_candidates(store) == [("o/r", 1)]


def test_find_submitted_candidates_empty_store(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    assert upstream_prs.find_submitted_candidates(store) == []


def test_find_submitted_candidates_dedupes_multiple_gate_runs_for_the_same_candidate(
    tmp_path,
) -> None:
    store = JsonlStore(tmp_path)
    _submit_gate_run(store, repo="o/r", number=1)
    _submit_gate_run(store, repo="o/r", number=1)

    assert upstream_prs.find_submitted_candidates(store) == [("o/r", 1)]


# --------------------------------------------------------------------- fetch_upstream_pr


def test_fetch_upstream_pr_builds_state_from_gh_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(
        upstream_prs.subprocess, "run", _fake_gh(pr_json=_OPEN_PR_JSON, comments=[])
    )

    pr = upstream_prs.fetch_upstream_pr(store, "o/r", 1)

    assert pr is not None
    assert pr.state == "open"
    assert pr.ci_status == "success"
    assert pr.mergeable is True
    assert pr.review_decision == "REVIEW_REQUIRED"
    assert pr.requested_reviewers == ("octocat", "vllm-maintainers")


def test_fetch_upstream_pr_returns_none_for_a_closed_pr(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    closed_json = {**_OPEN_PR_JSON, "state": "MERGED"}
    monkeypatch.setattr(upstream_prs.subprocess, "run", _fake_gh(pr_json=closed_json))

    assert upstream_prs.fetch_upstream_pr(store, "o/r", 1) is None


def test_fetch_upstream_pr_returns_none_on_gh_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)

    def _fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(upstream_prs.subprocess, "run", _fail)

    assert upstream_prs.fetch_upstream_pr(store, "o/r", 1) is None


def test_fetch_upstream_pr_marks_a_comment_outstanding_with_no_posted_reply(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(
        upstream_prs.subprocess,
        "run",
        _fake_gh(pr_json=_OPEN_PR_JSON, comments=[_gh_comment(101, "maintainer")]),
    )

    pr = upstream_prs.fetch_upstream_pr(store, "o/r", 1)

    assert pr is not None
    assert len(pr.comments) == 1
    assert pr.comments[0].outstanding is True
    assert pr.outstanding_count == 1


def test_fetch_upstream_pr_a_posted_reply_clears_outstanding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: only a real stage="review_post" record with posted=True clears a comment --
    a merely-composed (stage="review_response") draft is not enough, per this todo's own
    stricter-than-T3.11 "outstanding" definition."""
    store = JsonlStore(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 101,
            "recorded_at": "2026-01-02T00:00:00Z",
        }
    )
    monkeypatch.setattr(
        upstream_prs.subprocess,
        "run",
        _fake_gh(pr_json=_OPEN_PR_JSON, comments=[_gh_comment(101, "maintainer")]),
    )

    pr = upstream_prs.fetch_upstream_pr(store, "o/r", 1)

    assert pr is not None
    assert pr.comments[0].outstanding is True  # composed, not posted -- still outstanding

    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_post",
            "comment_id": 101,
            "posted": True,
            "posted_comment_url": "https://github.com/o/r/pull/1#issuecomment-1",
            "recorded_at": "2026-01-03T00:00:00Z",
        }
    )

    pr2 = upstream_prs.fetch_upstream_pr(store, "o/r", 1)

    assert pr2 is not None
    assert pr2.comments[0].outstanding is False
    assert pr2.outstanding_count == 0


def test_fetch_upstream_pr_excludes_the_bots_own_comment_from_outstanding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(
        upstream_prs.subprocess,
        "run",
        _fake_gh(
            pr_json=_OPEN_PR_JSON,
            comments=[_gh_comment(101, "forager-bot")],
            login="forager-bot",
        ),
    )

    pr = upstream_prs.fetch_upstream_pr(store, "o/r", 1)

    assert pr is not None
    assert pr.comments == ()
    assert pr.outstanding_count == 0


def test_fetch_upstream_pr_returns_none_when_comments_fetch_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)

    def _run(cmd, **kwargs):
        if cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_OPEN_PR_JSON))
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(upstream_prs.subprocess, "run", _run)

    assert upstream_prs.fetch_upstream_pr(store, "o/r", 1) is None


# --------------------------------------------------------------------- list_upstream_prs


def test_list_upstream_prs_lists_only_open_submitted_candidates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    _submit_gate_run(store, repo="o/r", number=1, pr_url="https://github.com/o/r/pull/1")
    _submit_gate_run(store, repo="o/r", number=2, pr_url="https://github.com/o/r/pull/2")

    def _run(cmd, **kwargs):
        if cmd[:3] == ["gh", "pr", "view"] and cmd[3] == "1" and cmd[5] == "o/r":
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_OPEN_PR_JSON))
        if cmd[:3] == ["gh", "pr", "view"] and cmd[3] == "2" and cmd[5] == "o/r":
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps({**_OPEN_PR_JSON, "state": "CLOSED"})
            )
        if cmd[:2] == ["gh", "api"] and cmd[2] == "user":
            return subprocess.CompletedProcess(cmd, 0, stdout="forager-bot\n")
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/comments"):
            return subprocess.CompletedProcess(cmd, 0, stdout="[]")
        raise AssertionError(f"unexpected gh invocation: {cmd}")

    monkeypatch.setattr(upstream_prs.subprocess, "run", _run)

    prs = upstream_prs.list_upstream_prs(store)

    assert [(pr.repo, pr.number) for pr in prs] == [("o/r", 1)]


def test_list_upstream_prs_empty_when_nothing_submitted(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    assert upstream_prs.list_upstream_prs(store) == []
