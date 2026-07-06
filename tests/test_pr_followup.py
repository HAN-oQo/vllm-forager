"""Tests for the PR follow-up steward (T3.13) — offline & deterministic; `gh`/subprocess mocked.

Per the DEVPLAN todo: mock PR states -> the steward picks the correct action per state
(ci_fail->engineer · comment->review-response · approved->notify · stale->draft-nudge ·
conflict->rebase); nothing is pushed/posted without an explicit approve.
"""

import json
import subprocess
from datetime import datetime, timezone

import pytest

from src import pr_followup, review_loop
from src.pr_followup import PRState
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m3

_NOW = datetime(2026, 1, 10, tzinfo=timezone.utc)


def _pr(
    *,
    repo="o/r",
    number=1,
    url="https://github.com/o/r/pull/1",
    state="open",
    ci_status="success",
    review_decision=None,
    has_new_comments=False,
    mergeable=True,
    last_activity_at="2026-01-09T00:00:00Z",
) -> PRState:
    return PRState(
        repo=repo,
        number=number,
        url=url,
        state=state,
        ci_status=ci_status,
        review_decision=review_decision,
        has_new_comments=has_new_comments,
        mergeable=mergeable,
        last_activity_at=last_activity_at,
    )


def _fail_if_subprocess_called(*a, **k):
    raise AssertionError("subprocess.run should not be called")


# --------------------------------------------------------------------- next_action


def test_conflict_triggers_rebase() -> None:
    action = pr_followup.next_action(_pr(mergeable=False), now=_NOW)
    assert action.action == pr_followup.ACTION_REBASE


def test_ci_failure_triggers_ci_fix() -> None:
    action = pr_followup.next_action(_pr(ci_status="failure"), now=_NOW)
    assert action.action == pr_followup.ACTION_CI_FIX


def test_changes_requested_triggers_review_response() -> None:
    action = pr_followup.next_action(_pr(review_decision="CHANGES_REQUESTED"), now=_NOW)
    assert action.action == pr_followup.ACTION_REVIEW_RESPONSE


def test_new_comment_alone_triggers_review_response() -> None:
    action = pr_followup.next_action(_pr(has_new_comments=True), now=_NOW)
    assert action.action == pr_followup.ACTION_REVIEW_RESPONSE


def test_approved_triggers_notify_ready() -> None:
    action = pr_followup.next_action(_pr(review_decision="APPROVED"), now=_NOW)
    assert action.action == pr_followup.ACTION_NOTIFY_READY


def test_stale_with_no_other_signal_triggers_nudge() -> None:
    action = pr_followup.next_action(_pr(last_activity_at="2026-01-01T00:00:00Z"), now=_NOW)
    assert action.action == pr_followup.ACTION_NUDGE_STALE


def test_fresh_with_no_signal_needs_no_action() -> None:
    action = pr_followup.next_action(_pr(), now=_NOW)
    assert action.action == pr_followup.ACTION_NONE


def test_closed_pr_needs_no_action_even_if_stale() -> None:
    action = pr_followup.next_action(
        _pr(state="closed", last_activity_at="2020-01-01T00:00:00Z"), now=_NOW
    )
    assert action.action == pr_followup.ACTION_NONE


def test_merged_pr_needs_no_action() -> None:
    action = pr_followup.next_action(_pr(state="merged"), now=_NOW)
    assert action.action == pr_followup.ACTION_NONE


def test_conflict_outranks_ci_failure() -> None:
    action = pr_followup.next_action(_pr(mergeable=False, ci_status="failure"), now=_NOW)
    assert action.action == pr_followup.ACTION_REBASE


def test_review_response_outranks_a_stale_approved_decision() -> None:
    """A comment landing after an approval must be answered before notifying "ready" --
    otherwise the human is told to merge a PR with unaddressed feedback sitting on it."""
    action = pr_followup.next_action(
        _pr(review_decision="APPROVED", has_new_comments=True), now=_NOW
    )
    assert action.action == pr_followup.ACTION_REVIEW_RESPONSE


# --------------------------------------------------------------------- apply_action


def test_apply_review_response_calls_run_review_loop_even_without_approve(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    calls = []
    monkeypatch.setattr(
        review_loop,
        "run_review_loop",
        lambda store, repo, number, **k: calls.append((repo, number)) or (),
    )
    action = pr_followup.FollowupAction("o/r", 1, pr_followup.ACTION_REVIEW_RESPONSE, "x")

    message = pr_followup.apply_action(store, action, _pr(), approve=False)

    assert calls == [("o/r", 1)]
    assert "no new maintainer comment" in message


def test_apply_review_response_reports_drafted_count(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    fake_response = object()
    monkeypatch.setattr(
        review_loop, "run_review_loop", lambda *a, **k: (fake_response, fake_response)
    )
    action = pr_followup.FollowupAction("o/r", 1, pr_followup.ACTION_REVIEW_RESPONSE, "x")

    message = pr_followup.apply_action(store, action, _pr(), approve=False)

    assert "drafted 2 replies" in message


def test_apply_notify_ready_without_approve_never_calls_subprocess(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(pr_followup.subprocess, "run", _fail_if_subprocess_called)
    action = pr_followup.FollowupAction("o/r", 1, pr_followup.ACTION_NOTIFY_READY, "x")

    message = pr_followup.apply_action(store, action, _pr(), approve=False)

    assert "pass --approve" in message


def test_apply_notify_ready_with_approve_calls_notify_script(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    calls = []
    monkeypatch.setattr(
        pr_followup.subprocess,
        "run",
        lambda cmd, **k: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )
    action = pr_followup.FollowupAction("o/r", 1, pr_followup.ACTION_NOTIFY_READY, "x")

    message = pr_followup.apply_action(store, action, _pr(), approve=True)

    assert len(calls) == 1
    assert calls[0][0] == str(pr_followup._NOTIFY_SCRIPT)
    assert "notified" in message


def test_apply_nudge_stale_without_approve_never_calls_subprocess(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(pr_followup.subprocess, "run", _fail_if_subprocess_called)
    action = pr_followup.FollowupAction("o/r", 1, pr_followup.ACTION_NUDGE_STALE, "x")

    message = pr_followup.apply_action(store, action, _pr(), approve=False)

    assert "pass --approve" in message


def test_apply_nudge_stale_with_approve_calls_notify_script(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    calls = []
    monkeypatch.setattr(
        pr_followup.subprocess,
        "run",
        lambda cmd, **k: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )
    action = pr_followup.FollowupAction("o/r", 1, pr_followup.ACTION_NUDGE_STALE, "x")

    pr_followup.apply_action(store, action, _pr(), approve=True)

    assert len(calls) == 1


def test_apply_ci_fix_never_calls_subprocess_regardless_of_approve(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(pr_followup.subprocess, "run", _fail_if_subprocess_called)
    action = pr_followup.FollowupAction("o/r", 1, pr_followup.ACTION_CI_FIX, "x")

    message = pr_followup.apply_action(store, action, _pr(), approve=True)

    assert "not automated" in message


def test_apply_rebase_never_calls_subprocess_regardless_of_approve(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(pr_followup.subprocess, "run", _fail_if_subprocess_called)
    action = pr_followup.FollowupAction("o/r", 1, pr_followup.ACTION_REBASE, "x")

    message = pr_followup.apply_action(store, action, _pr(), approve=True)

    assert "not automated" in message


def test_apply_none_action_is_a_no_op(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(pr_followup.subprocess, "run", _fail_if_subprocess_called)
    action = pr_followup.FollowupAction("o/r", 1, pr_followup.ACTION_NONE, "x")

    message = pr_followup.apply_action(store, action, _pr(), approve=True)

    assert "no action needed" in message


# --------------------------------------------------------------------- fetch_pr_state


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


def test_fetch_pr_state_builds_state_from_gh_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    pr_json = {
        "url": "https://github.com/o/r/pull/1",
        "state": "OPEN",
        "mergeable": "CONFLICTING",
        "reviewDecision": "CHANGES_REQUESTED",
        "updatedAt": "2026-01-05T00:00:00Z",
        "statusCheckRollup": [{"conclusion": "FAILURE", "status": "COMPLETED"}],
    }
    monkeypatch.setattr(pr_followup.subprocess, "run", _fake_gh(pr_json=pr_json, comments=[]))

    pr = pr_followup.fetch_pr_state(store, "o/r", 1)

    assert pr is not None
    assert pr.state == "open"
    assert pr.mergeable is False
    assert pr.ci_status == "failure"
    assert pr.review_decision == "CHANGES_REQUESTED"
    assert pr.last_activity_at == "2026-01-05T00:00:00Z"


def test_fetch_pr_state_marks_a_genuinely_new_comment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    pr_json = {
        "url": "u",
        "state": "OPEN",
        "mergeable": "MERGEABLE",
        "reviewDecision": "",
        "updatedAt": "2026-01-05T00:00:00Z",
        "statusCheckRollup": [],
    }
    monkeypatch.setattr(
        pr_followup.subprocess,
        "run",
        _fake_gh(pr_json=pr_json, comments=[_gh_comment(1, "maintainer")]),
    )

    pr = pr_followup.fetch_pr_state(store, "o/r", 1)

    assert pr is not None
    assert pr.has_new_comments is True
    assert pr.review_decision is None
    assert pr.ci_status == "unknown"


def test_fetch_pr_state_excludes_an_already_answered_comment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    pr_json = {
        "url": "u",
        "state": "OPEN",
        "mergeable": "MERGEABLE",
        "reviewDecision": "",
        "updatedAt": "2026-01-05T00:00:00Z",
        "statusCheckRollup": [],
    }
    monkeypatch.setattr(
        pr_followup.subprocess,
        "run",
        _fake_gh(pr_json=pr_json, comments=[_gh_comment(1, "maintainer")]),
    )

    pr = pr_followup.fetch_pr_state(store, "o/r", 1)

    assert pr is not None
    assert pr.has_new_comments is False


def test_fetch_pr_state_excludes_the_bots_own_comment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    pr_json = {
        "url": "u",
        "state": "OPEN",
        "mergeable": "MERGEABLE",
        "reviewDecision": "",
        "updatedAt": "2026-01-05T00:00:00Z",
        "statusCheckRollup": [],
    }
    monkeypatch.setattr(
        pr_followup.subprocess,
        "run",
        _fake_gh(pr_json=pr_json, comments=[_gh_comment(1, "forager-bot")], login="forager-bot"),
    )

    pr = pr_followup.fetch_pr_state(store, "o/r", 1)

    assert pr is not None
    assert pr.has_new_comments is False


def test_fetch_pr_state_returns_none_on_gh_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)

    def _fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(pr_followup.subprocess, "run", _fail)

    assert pr_followup.fetch_pr_state(store, "o/r", 1) is None


# --------------------------------------------------------------------- CLI


def test_cli_dry_run_does_not_notify_without_approve(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(pr_followup, "resolve_store", lambda data_dir: (store, tmp_path))
    monkeypatch.setattr(
        pr_followup,
        "fetch_pr_state",
        lambda store, repo, number: _pr(review_decision="APPROVED"),
    )
    monkeypatch.setattr(pr_followup.subprocess, "run", _fail_if_subprocess_called)

    rc = pr_followup.main(["--candidate", "o/r#1"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "[notify_ready]" in out
    assert "pass --approve" in out


def test_cli_with_approve_notifies(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(pr_followup, "resolve_store", lambda data_dir: (store, tmp_path))
    monkeypatch.setattr(
        pr_followup,
        "fetch_pr_state",
        lambda store, repo, number: _pr(review_decision="APPROVED"),
    )
    calls = []
    monkeypatch.setattr(
        pr_followup.subprocess,
        "run",
        lambda cmd, **k: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    rc = pr_followup.main(["--candidate", "o/r#1", "--approve"])

    assert rc == 0
    assert len(calls) == 1
    assert "notified" in capsys.readouterr().out


def test_cli_returns_nonzero_when_fetch_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(pr_followup, "resolve_store", lambda data_dir: (store, tmp_path))
    monkeypatch.setattr(pr_followup, "fetch_pr_state", lambda store, repo, number: None)

    rc = pr_followup.main(["--candidate", "o/r#1"])

    assert rc == 1
    assert "could not fetch PR state" in capsys.readouterr().out
