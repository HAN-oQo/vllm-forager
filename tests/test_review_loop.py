"""Tests for the review-response loop (T3.11) — offline & deterministic; `gh`/llm mocked.

Per the DEVPLAN todo: a mock review comment -> a response draft + a proposed diff are produced;
nothing is pushed without approval.
"""

import subprocess
from datetime import datetime, timezone

import pytest

from src import gate, llm, review_loop
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m3

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _store_with_item(tmp_path, *, repo="o/r", number=1):
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            {
                "repo": repo,
                "number": number,
                "type": "issue",
                "title": "vLLM crashes on gfx90a with fp8",
                "body": "assertion",
                "state": "open",
                "url": f"https://github.com/{repo}/issues/{number}",
            }
        ]
    )
    return store


def _gh_comment(comment_id, author, body):
    return {"id": comment_id, "user": {"login": author}, "body": body}


def _fake_gh(*, comments=None, login="forager-bot", post_url="https://github.com/o/r/pull/1#c1"):
    """A `subprocess.run` stand-in dispatching on the `gh` sub-command actually invoked."""
    comments = comments if comments is not None else []

    def _run(cmd, **kwargs):
        if cmd[:2] == ["gh", "api"] and cmd[2] == "user":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{login}\n")
        if len(cmd) >= 3 and cmd[2].endswith("/comments") and "-f" not in cmd:
            import json

            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(comments))
        if "-f" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{post_url}\n")
        raise AssertionError(f"unexpected gh invocation: {cmd}")

    return _run


def _fail_if_gh_called(*a, **k):
    raise AssertionError("gh (subprocess.run) should not be called")


def _fail_if_llm_called(*a, **k):
    raise AssertionError("llm.complete should not be called")


@pytest.fixture(autouse=True)
def _isolate_review_drafts_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(review_loop, "_REVIEW_DRAFTS_DIR", tmp_path / "review_drafts")


@pytest.fixture(autouse=True)
def _mock_risk_score(monkeypatch: pytest.MonkeyPatch):
    """`gate.assemble_bundle` (called for diff context) triggers a real LLM call via
    `_risk_badge` -- isolate every test from it, matching tests/test_gate.py's own fixture."""
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )


# --------------------------------------------------------------------- run_review_loop


def test_composes_a_reply_for_a_new_maintainer_comment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        _fake_gh(comments=[_gh_comment(1, "maintainer", "Please add a test for the empty case")]),
    )
    monkeypatch.setattr(
        llm,
        "complete",
        lambda *a, **k: {"reply": "Good catch, adding a test now.", "proposed_diff": None},
    )

    responses = review_loop.run_review_loop(store, "o/r", 1, now=_NOW)

    assert len(responses) == 1
    assert responses[0].comment_id == 1
    assert responses[0].comment_author == "maintainer"
    assert responses[0].reply == "Good catch, adding a test now."
    assert responses[0].proposed_diff is None

    runs = store.list_runs(repo="o/r", number=1, stage="review_response")
    assert len(runs) == 1
    assert runs[0]["comment_id"] == 1


def test_composed_response_includes_a_proposed_diff_when_given(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        _fake_gh(comments=[_gh_comment(1, "maintainer", "Add a null check here")]),
    )
    monkeypatch.setattr(
        llm,
        "complete",
        lambda *a, **k: {
            "reply": "Added the null check.",
            "proposed_diff": "--- a/x.py\n+++ b/x.py\n+if x is None: return\n",
        },
    )

    responses = review_loop.run_review_loop(store, "o/r", 1, now=_NOW)

    assert responses[0].proposed_diff == "--- a/x.py\n+++ b/x.py\n+if x is None: return\n"


def test_writes_a_draft_file_with_the_comment_and_reply(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        _fake_gh(comments=[_gh_comment(1, "maintainer", "Please add a test")]),
    )
    monkeypatch.setattr(
        llm, "complete", lambda *a, **k: {"reply": "Adding a test now.", "proposed_diff": None}
    )

    responses = review_loop.run_review_loop(store, "o/r", 1, now=_NOW)

    path = review_loop._draft_path("o/r", 1, 1)
    assert path.exists()
    text = path.read_text()
    assert "Please add a test" in text
    assert "Adding a test now." in text
    assert responses[0].reply in text


def test_skips_a_comment_already_responded_to(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "already replied",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        _fake_gh(comments=[_gh_comment(1, "maintainer", "Please add a test")]),
    )
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)

    responses = review_loop.run_review_loop(store, "o/r", 1, now=_NOW)

    assert responses == ()


def test_skips_the_bots_own_previously_posted_comments(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        _fake_gh(
            comments=[_gh_comment(1, "forager-bot", "Thanks, addressed in the latest push.")],
            login="forager-bot",
        ),
    )
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)

    responses = review_loop.run_review_loop(store, "o/r", 1, now=_NOW)

    assert responses == ()


def test_returns_empty_when_no_item(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(review_loop.subprocess, "run", _fail_if_gh_called)
    assert review_loop.run_review_loop(store, "o/r", 1) == ()


def test_returns_empty_when_fetch_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_with_item(tmp_path)

    def _raise(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(review_loop.subprocess, "run", _raise)

    assert review_loop.run_review_loop(store, "o/r", 1) == ()


def test_works_without_a_gate_ready_bundle(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A submitted candidate might no longer be "gate ready" (e.g. a later regression) -- the
    review loop must still compose a response, just with empty diff context."""
    store = _store_with_item(tmp_path)  # no verify/self_review runs at all
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        _fake_gh(comments=[_gh_comment(1, "maintainer", "Please add a test")]),
    )
    captured = {}

    def _fake_complete(prompt, **k):
        captured["prompt"] = prompt
        return {"reply": "ok", "proposed_diff": None}

    monkeypatch.setattr(llm, "complete", _fake_complete)

    responses = review_loop.run_review_loop(store, "o/r", 1, now=_NOW)

    assert len(responses) == 1
    assert "Current diff (for context):\n\n\n" in captured["prompt"]


def test_composed_response_none_on_malformed_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"reply": "", "proposed_diff": None})
    assert review_loop._compose_response("c", "t", "d") is None


def test_composed_response_treats_blank_proposed_diff_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"reply": "ok", "proposed_diff": "   "})
    result = review_loop._compose_response("c", "t", "d")
    assert result == ("ok", None)


# --------------------------------------------------------------------- post_reply


def test_post_reply_returns_none_when_nothing_composed(tmp_path) -> None:
    store = _store_with_item(tmp_path)
    assert review_loop.post_reply(store, "o/r", 1, 1, post=True) is None


def test_post_reply_does_not_call_gh_when_post_is_false(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "Adding a test now.",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    monkeypatch.setattr(review_loop.subprocess, "run", _fail_if_gh_called)

    result = review_loop.post_reply(store, "o/r", 1, 1, post=False)

    assert result is not None
    assert result.posted is False
    assert result.posted_comment_url is None


def test_post_reply_calls_gh_when_post_is_true(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "Adding a test now.",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="https://github.com/o/r/pull/1#c1\n")

    monkeypatch.setattr(review_loop.subprocess, "run", _run)

    result = review_loop.post_reply(store, "o/r", 1, 1, post=True)

    assert result.posted is True
    assert result.posted_comment_url == "https://github.com/o/r/pull/1#c1"
    assert len(calls) == 1
    assert "body=Adding a test now." in calls[0]


def test_post_reply_records_kb_outcome(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "ok",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="https://x/c1\n"),
    )

    review_loop.post_reply(store, "o/r", 1, 1, post=True)

    runs = store.list_runs(repo="o/r", number=1, stage="review_post")
    assert len(runs) == 1
    assert runs[0]["posted"] is True


def test_post_reply_picks_the_latest_composed_response_for_that_comment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "older draft",
            "recorded_at": "2025-01-01T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "newer draft",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    calls = []
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        lambda cmd, **k: (calls.append(cmd), subprocess.CompletedProcess(cmd, 0, stdout="u\n"))[1],
    )

    review_loop.post_reply(store, "o/r", 1, 1, post=True)

    assert "body=newer draft" in calls[0]


# --------------------------------------------------------------------- _current_gh_login


def test_current_gh_login_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(review_loop.subprocess, "run", _raise)
    assert review_loop._current_gh_login() is None


def test_current_gh_login_returns_the_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="forager-bot\n"),
    )
    assert review_loop._current_gh_login() == "forager-bot"


# --------------------------------------------------------------------- fail-safe behavior


def test_run_review_loop_fails_closed_when_login_lookup_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: if the bot can't resolve its own gh identity, it must compose NOTHING rather
    than risk treating its own past replies as new maintainer comments (fail closed, not open)."""
    store = _store_with_item(tmp_path)

    def _run(cmd, **kwargs):
        if cmd[2] == "user":
            raise subprocess.CalledProcessError(1, cmd)
        import json

        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps([_gh_comment(1, "maintainer", "Please add a test")])
        )

    monkeypatch.setattr(review_loop.subprocess, "run", _run)
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)

    assert review_loop.run_review_loop(store, "o/r", 1) == ()


def test_run_review_loop_skips_a_malformed_comment_without_crashing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A comment with a non-dict `user` field (a malformed/unexpected API response) must be
    skipped, not crash the whole batch -- the other, well-formed comment still gets a reply."""
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        _fake_gh(
            comments=[
                {"id": 1, "user": "not-a-dict", "body": "malformed"},
                _gh_comment(2, "maintainer", "Please add a test"),
            ]
        ),
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"reply": "ok", "proposed_diff": None})

    responses = review_loop.run_review_loop(store, "o/r", 1, now=_NOW)

    assert len(responses) == 1
    assert responses[0].comment_id == 2


def test_run_review_loop_uses_verified_diff_not_assemble_bundle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: must use the cheaper gate.verified_diff (no risk-badge LLM call), not
    gate.assemble_bundle, matching pr_quality.py's own established fix for the identical
    'just need the diff' need."""
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        _fake_gh(comments=[_gh_comment(1, "maintainer", "Please add a test")]),
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"reply": "ok", "proposed_diff": None})

    def _fail_if_assemble_bundle_called(*a, **k):
        raise AssertionError("gate.assemble_bundle should not be called")

    monkeypatch.setattr(gate, "assemble_bundle", _fail_if_assemble_bundle_called)

    responses = review_loop.run_review_loop(store, "o/r", 1, now=_NOW)

    assert len(responses) == 1


# --------------------------------------------------------------------- post_reply: idempotency


def test_post_reply_refuses_to_post_twice(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "ok",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_post",
            "comment_id": 1,
            "posted": True,
            "posted_comment_url": "https://x/c1",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(review_loop.subprocess, "run", _fail_if_gh_called)

    result = review_loop.post_reply(store, "o/r", 1, 1, post=True)

    assert result is not None
    assert result.posted is False
    assert result.posted_comment_url == "https://x/c1"


def test_post_reply_allows_posting_after_a_failed_prior_attempt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prior review_post record with posted=False (the gh call failed) must not block a
    later, real retry."""
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "ok",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_post",
            "comment_id": 1,
            "posted": False,
            "posted_comment_url": None,
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="https://x/c2\n"),
    )

    result = review_loop.post_reply(store, "o/r", 1, 1, post=True)

    assert result.posted is True
    assert result.posted_comment_url == "https://x/c2"


# --------------------------------------------------------------------- post_reply: draft-file edits


def test_post_reply_posts_a_hand_edited_draft_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a human editing the on-disk draft's '## Proposed reply' section must have
    that edit actually posted, not the original, unedited KB record text."""
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "original LLM-composed reply",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()
    path = review_loop._draft_path("o/r", 1, 1, drafts_dir=drafts_dir)
    path.write_text(
        "# Reply to maintainer's comment on o/r#1\n\n"
        "## Their comment\nPlease add a test\n\n"
        "## Proposed reply\nHand-edited reply text.\n"
    )
    calls = []
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        lambda cmd, **k: (calls.append(cmd), subprocess.CompletedProcess(cmd, 0, stdout="u\n"))[1],
    )

    review_loop.post_reply(store, "o/r", 1, 1, post=True, drafts_dir=drafts_dir)

    assert "body=Hand-edited reply text." in calls[0]


def test_post_reply_falls_back_to_kb_record_when_no_draft_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "original LLM-composed reply",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    calls = []
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        lambda cmd, **k: (calls.append(cmd), subprocess.CompletedProcess(cmd, 0, stdout="u\n"))[1],
    )

    review_loop.post_reply(
        store, "o/r", 1, 1, post=True, drafts_dir=tmp_path / "nonexistent_drafts"
    )

    assert "body=original LLM-composed reply" in calls[0]


# --------------------------------------------------------------------- CLI


def test_cli_compose_prints_drafted_comment_ids(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        _fake_gh(comments=[_gh_comment(1, "maintainer", "Please add a test")]),
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"reply": "ok", "proposed_diff": None})
    monkeypatch.setattr(review_loop, "resolve_store", lambda data_dir: (store, tmp_path))

    rc = review_loop.main(["compose", "--candidate", "o/r#1"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "comment 1" in out


def test_cli_compose_reports_nothing_new(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(review_loop.subprocess, "run", _fake_gh(comments=[]))
    monkeypatch.setattr(review_loop, "resolve_store", lambda data_dir: (store, tmp_path))

    rc = review_loop.main(["compose", "--candidate", "o/r#1"])

    assert rc == 0
    assert "No new maintainer comments" in capsys.readouterr().out


def test_cli_post_without_flag_does_not_call_gh(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "ok",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    monkeypatch.setattr(review_loop.subprocess, "run", _fail_if_gh_called)
    monkeypatch.setattr(review_loop, "resolve_store", lambda data_dir: (store, tmp_path))

    rc = review_loop.main(["post", "--candidate", "o/r#1", "--comment-id", "1"])

    assert rc == 0
    assert "Not posted" in capsys.readouterr().out


def test_cli_post_with_flag_calls_gh(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    store = _store_with_item(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "review_response",
            "comment_id": 1,
            "reply": "ok",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    monkeypatch.setattr(
        review_loop.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="https://x/c1\n"),
    )
    monkeypatch.setattr(review_loop, "resolve_store", lambda data_dir: (store, tmp_path))

    rc = review_loop.main(["post", "--candidate", "o/r#1", "--comment-id", "1", "--post"])

    assert rc == 0
    assert "Posted: https://x/c1" in capsys.readouterr().out


def test_cli_post_returns_nonzero_when_nothing_composed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(review_loop, "resolve_store", lambda data_dir: (store, tmp_path))

    rc = review_loop.main(["post", "--candidate", "o/r#1", "--comment-id", "99"])

    assert rc == 1
    assert "No composed response" in capsys.readouterr().out
