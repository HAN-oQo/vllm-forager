"""Tests for the Upstream PRs tracker (T5.14) — offline & deterministic; `gh`/subprocess mocked.

Per the DEVPLAN todo: seeded stage="gate" (submitted) + stage="review_response"/review_post
records, gh mocked -> the tab lists only open real PRs, correctly flags un-responded comments
as outstanding, and renders live check status.

`gh` calls are mocked by patching `pr_followup.subprocess.run` (not `upstream_prs.subprocess`
-- this module has no `subprocess` import of its own; it composes over
`pr_followup._gh_pr_view`/`review_loop._fetch_comments`/`review_loop._current_gh_login`, all
of which share the one real `subprocess` module object, so patching it via any of their own
module references is equivalent).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from src import pr_followup, upstream_prs
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


def test_fetch_upstream_pr_builds_state_from_gh_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_followup.subprocess, "run", _fake_gh(pr_json=_OPEN_PR_JSON, comments=[]))

    pr = upstream_prs.fetch_upstream_pr(
        "o/r", 1, self_login="forager-bot", posted_comment_ids=frozenset()
    )

    assert pr is not None
    assert pr.state == "open"
    assert pr.ci_status == "success"
    assert pr.mergeable is True
    assert pr.review_decision == "REVIEW_REQUIRED"
    assert pr.requested_reviewers == ("octocat", "vllm-maintainers")


def test_fetch_upstream_pr_returns_none_for_a_closed_pr(monkeypatch: pytest.MonkeyPatch) -> None:
    closed_json = {**_OPEN_PR_JSON, "state": "MERGED"}
    monkeypatch.setattr(pr_followup.subprocess, "run", _fake_gh(pr_json=closed_json))

    pr = upstream_prs.fetch_upstream_pr(
        "o/r", 1, self_login="forager-bot", posted_comment_ids=frozenset()
    )
    assert pr is None


def test_fetch_upstream_pr_returns_none_on_gh_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(pr_followup.subprocess, "run", _fail)

    pr = upstream_prs.fetch_upstream_pr(
        "o/r", 1, self_login="forager-bot", posted_comment_ids=frozenset()
    )
    assert pr is None


def test_fetch_upstream_pr_marks_a_comment_outstanding_with_no_posted_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pr_followup.subprocess,
        "run",
        _fake_gh(pr_json=_OPEN_PR_JSON, comments=[_gh_comment(101, "maintainer")]),
    )

    pr = upstream_prs.fetch_upstream_pr(
        "o/r", 1, self_login="forager-bot", posted_comment_ids=frozenset()
    )

    assert pr is not None
    assert len(pr.comments) == 1
    assert pr.comments[0].outstanding is True
    assert pr.outstanding_count == 1


def test_fetch_upstream_pr_a_posted_reply_clears_outstanding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: only a real stage="review_post" record with posted=True clears a comment --
    a merely-composed (stage="review_response") draft is not enough, per this todo's own
    stricter-than-T3.11 "outstanding" definition."""
    monkeypatch.setattr(
        pr_followup.subprocess,
        "run",
        _fake_gh(pr_json=_OPEN_PR_JSON, comments=[_gh_comment(101, "maintainer")]),
    )

    pr = upstream_prs.fetch_upstream_pr(
        "o/r", 1, self_login="forager-bot", posted_comment_ids=frozenset()
    )
    assert pr is not None
    assert pr.comments[0].outstanding is True  # composed, not posted -- still outstanding

    pr2 = upstream_prs.fetch_upstream_pr(
        "o/r", 1, self_login="forager-bot", posted_comment_ids=frozenset({101})
    )
    assert pr2 is not None
    assert pr2.comments[0].outstanding is False
    assert pr2.outstanding_count == 0


def test_fetch_upstream_pr_excludes_the_bots_own_comment_from_outstanding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pr_followup.subprocess,
        "run",
        _fake_gh(
            pr_json=_OPEN_PR_JSON,
            comments=[_gh_comment(101, "forager-bot")],
            login="forager-bot",
        ),
    )

    pr = upstream_prs.fetch_upstream_pr(
        "o/r", 1, self_login="forager-bot", posted_comment_ids=frozenset()
    )

    assert pr is not None
    assert pr.comments == ()
    assert pr.outstanding_count == 0


def test_fetch_upstream_pr_returns_none_when_comments_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _run(cmd, **kwargs):
        if cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_OPEN_PR_JSON))
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(pr_followup.subprocess, "run", _run)

    pr = upstream_prs.fetch_upstream_pr(
        "o/r", 1, self_login="forager-bot", posted_comment_ids=frozenset()
    )
    assert pr is None


def test_fetch_upstream_pr_mergeable_none_when_gh_has_not_computed_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: GitHub's own "UNKNOWN" mergeable state (not yet computed) must map to
    `None`, distinct from a confirmed `True`/`False` -- see pr_followup._MERGEABLE's own
    docstring for why this three-state distinction matters."""
    unknown_json = {**_OPEN_PR_JSON, "mergeable": "UNKNOWN"}
    monkeypatch.setattr(pr_followup.subprocess, "run", _fake_gh(pr_json=unknown_json))

    pr = upstream_prs.fetch_upstream_pr(
        "o/r", 1, self_login="forager-bot", posted_comment_ids=frozenset()
    )

    assert pr is not None
    assert pr.mergeable is None


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

    monkeypatch.setattr(pr_followup.subprocess, "run", _run)

    prs = upstream_prs.list_upstream_prs(store)

    assert [(pr.repo, pr.number) for pr in prs] == [("o/r", 1)]


def test_list_upstream_prs_empty_when_nothing_submitted(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    assert upstream_prs.list_upstream_prs(store) == []


def test_list_upstream_prs_resolves_identity_and_review_post_scan_only_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (a code-review finding): the bot's own gh identity and the KB's
    stage="review_post" history must be resolved once for the whole poll, not once per
    candidate -- confirmed here by counting each kind of call across 3 open candidates."""
    store = JsonlStore(tmp_path)
    for n in (1, 2, 3):
        _submit_gate_run(store, repo="o/r", number=n, pr_url=f"https://github.com/o/r/pull/{n}")

    identity_calls = 0
    review_post_scans = 0
    real_list_runs = store.list_runs

    def _counting_list_runs(*args, **kwargs):
        nonlocal review_post_scans
        if kwargs.get("stage") == "review_post":
            review_post_scans += 1
        return real_list_runs(*args, **kwargs)

    monkeypatch.setattr(store, "list_runs", _counting_list_runs)

    def _run(cmd, **kwargs):
        nonlocal identity_calls
        if cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_OPEN_PR_JSON))
        if cmd[:2] == ["gh", "api"] and cmd[2] == "user":
            identity_calls += 1
            return subprocess.CompletedProcess(cmd, 0, stdout="forager-bot\n")
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/comments"):
            return subprocess.CompletedProcess(cmd, 0, stdout="[]")
        raise AssertionError(f"unexpected gh invocation: {cmd}")

    monkeypatch.setattr(pr_followup.subprocess, "run", _run)

    prs = upstream_prs.list_upstream_prs(store)

    assert len(prs) == 3
    assert identity_calls == 1
    assert review_post_scans == 1


def test_list_upstream_prs_fails_closed_when_identity_resolution_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    _submit_gate_run(store)

    def _run(cmd, **kwargs):
        if cmd[:2] == ["gh", "api"] and cmd[2] == "user":
            raise subprocess.CalledProcessError(1, cmd)
        raise AssertionError(f"unexpected gh invocation: {cmd}")

    monkeypatch.setattr(pr_followup.subprocess, "run", _run)

    assert upstream_prs.list_upstream_prs(store) == []
