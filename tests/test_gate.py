"""Tests for the human gate (T3.5/T3.7) — offline & deterministic; `gh` subprocess mocked.

Per the DEVPLAN todo: unapproved => gh never called; approved-without-submit => a draft file is
written but gh is still never called; approved AND submitted => gh invoked exactly once
(subprocess mocked). HARD: nothing reaches upstream without both an explicit approval AND an
explicit submission.
"""

import subprocess

import pytest

from src import gate
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m3


def _store_ready_for_gate(
    tmp_path,
    *,
    verified=True,
    advance=True,
    matching_verify_recorded_at=True,
    repo="o/r",
    number=1,
):
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
            "stage": "repro",
            "command": "pytest test_fp8.py",
            "log": "AssertionError\n",
            "reproduced": True,
            "recorded_at": "2025-12-30T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "verify",
            "branch": f"forager/{repo.replace('/', '-')}-{number}",
            "patch": "--- a/x.py\n+++ b/x.py\n",
            "log": "1 passed\n",
            "verified": verified,
            "recorded_at": "2025-12-31T00:00:00Z",
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
            "advance": advance,
            "verify_recorded_at": "2025-12-31T00:00:00Z" if matching_verify_recorded_at else "x",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    return store


def _add_passing_narrative(
    store,
    *,
    repo="o/r",
    number=1,
    body="composed body",
    pr_author_recorded_at="2026-01-02T00:00:00Z",
):
    """Add a `stage=\"pr_author\"` run plus a matching, passing `stage=\"pr_quality\"` run --
    what `gate.main()`'s `_current_narrative` needs to use `body` and allow `--submit`."""
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "pr_author",
            "title": "[Bugfix] composed title",
            "body": body,
            "verify_recorded_at": "2025-12-31T00:00:00Z",
            "recorded_at": pr_author_recorded_at,
        }
    )
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "pr_quality",
            "votes": [{"acceptable": True, "reason": "ok"}],
            "approve_count": 1,
            "total_votes": 1,
            "passes": True,
            "pr_author_recorded_at": pr_author_recorded_at,
            "recorded_at": "2026-01-03T00:00:00Z",
        }
    )


def _fail_if_gh_called(*a, **k):
    raise AssertionError("gh (subprocess.run) should not be called")


def _approve_then_submit(store, repo, number, *, quality_passed=True, **kwargs):
    """T3.7's `draft_is_new` check requires the draft to already exist from an earlier, separate
    call before `submit=True` actually submits (see `gate.py`'s own module docstring) -- do the
    required first bare `approve=True` call (creates the draft) and return the second call's
    result (the draft already exists by then, so this one actually attempts submission).
    `quality_passed=True` by default (T3.10.5's own required condition) so every existing
    "submission succeeds" test doesn't have to know about it; tests of that condition itself
    override it."""
    gate.run_gate(store, repo, number, approve=True)
    return gate.run_gate(
        store, repo, number, approve=True, submit=True, quality_passed=quality_passed, **kwargs
    )


@pytest.fixture(autouse=True)
def _mock_risk_score(monkeypatch: pytest.MonkeyPatch):
    """Isolate every test from the real LLM call `_risk_badge` makes (scout's own `_score`) --
    individual tests can still override this via their own `monkeypatch.setattr(gate, "_score",
    ...)` afterward. Without this, `gate.subprocess.run` mocks below would also intercept
    scout's real underlying `claude -p` subprocess call (both go through the same `subprocess`
    module object), corrupting call counts and assertions unrelated to `gh`."""
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )


@pytest.fixture(autouse=True)
def _isolate_pr_drafts_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Every test that approves a candidate writes a real PR-draft file (that's the whole point
    of T3.7's fix) -- redirect the module-level default to `tmp_path` so tests never touch the
    real shared `config.DATA_DIR`. Tests calling `run_gate`/`_finalize` directly with their own
    explicit `pr_drafts_dir=...` aren't affected by this (their own argument wins); this only
    changes what the *default* (`None` -> `_PR_DRAFTS_DIR`) resolves to."""
    monkeypatch.setattr(gate, "_PR_DRAFTS_DIR", tmp_path / "pr_drafts")


# --------------------------------------------------------------------- run_gate: the HARD rule


def test_unapproved_never_calls_gh(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    result = gate.run_gate(store, "o/r", 1, approve=False)

    assert result is not None
    assert result.approved is False
    assert result.pr_url is None


def test_approved_and_submitted_calls_gh(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_gate(tmp_path)
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="https://github.com/o/r/pull/99\n")

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)

    result = _approve_then_submit(store, "o/r", 1)

    assert result is not None
    assert result.approved is True
    assert result.submitted is True
    assert result.pr_url == "https://github.com/o/r/pull/99"
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:4] == ["gh", "pr", "create", "--draft"]
    assert "--repo" in cmd and "o/r" in cmd
    assert "--head" in cmd and "forager/o-r-1" in cmd


def test_approve_and_submit_together_on_first_call_does_not_submit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core residual gap a code-review finding raised on this very fix: two required flags
    checked in one function call are still just as easy to pass together on a single command
    line as the one flag they replaced. `submit=True` must be inert the first time a candidate
    is ever approved -- the draft has to exist from an *earlier*, separate call first."""
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    result = gate.run_gate(store, "o/r", 1, approve=True, submit=True)

    assert result is not None
    assert result.approved is True
    assert result.submitted is False
    assert result.pr_url is None
    assert result.draft_is_new is True
    assert result.draft_path is not None
    assert result.draft_path.exists()


def test_approve_and_submit_on_a_later_call_does_submit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a draft already exists from an earlier call, a later approve+submit call actually
    submits -- the fix requires a temporal gap between the two, not a permanent block."""
    store = _store_ready_for_gate(tmp_path)
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="https://github.com/o/r/pull/99\n")

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)

    gate.run_gate(store, "o/r", 1, approve=True)  # first call: writes the draft only
    result = gate.run_gate(
        store, "o/r", 1, approve=True, submit=True, quality_passed=True
    )  # second: submits

    assert result is not None
    assert result.submitted is True
    assert result.draft_is_new is False
    assert result.pr_url == "https://github.com/o/r/pull/99"
    assert len(calls) == 1


def test_approved_without_submit_never_calls_gh_but_writes_draft(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core T3.7 safety property: T3.6's real incident was a single flag opening a real,
    public PR. `approve` alone must now only ever produce a local artifact -- `gh` is never
    invoked without a *separate*, explicit `submit=True` too."""
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    result = gate.run_gate(store, "o/r", 1, approve=True)

    assert result is not None
    assert result.approved is True
    assert result.submitted is False
    assert result.pr_url is None
    assert result.draft_path is not None
    assert result.draft_path.exists()
    assert "o/r#1" in result.draft_path.read_text()


def test_submit_alone_without_approve_never_calls_gh(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `submit=True` with `approve` left at its default (`False`) must not submit
    either -- BOTH flags are required, not just `submit`."""
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    result = gate.run_gate(store, "o/r", 1, submit=True)

    assert result is not None
    assert result.approved is False
    assert result.submitted is False
    assert result.pr_url is None
    assert result.draft_path is None


def test_submit_requires_the_literal_true_not_just_truthiness(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `submit` is checked with `is True`, mirroring `approve`'s own hardening --
    a caller passing any other truthy value must NOT open a PR."""
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    result = gate.run_gate(store, "o/r", 1, approve=True, submit="yes")  # type: ignore[arg-type]

    assert result is not None
    assert result.submitted is False
    assert result.pr_url is None


def test_approved_with_fork_owner_prefixes_head(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: `--head <branch>` alone makes `gh` look for that branch *inside* `--repo`
    itself, which fails for a fork-hosted branch (`gh` errors "No commits between main and
    <branch>" / "Head ref must be a branch") -- T3.6's own first real run hit this. `gh`'s
    required form for a fork-hosted head is `owner:branch`."""
    store = _store_ready_for_gate(tmp_path)
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="https://github.com/o/r/pull/99\n")

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)

    result = _approve_then_submit(store, "o/r", 1, fork_owner="HAN-oQo")

    assert result is not None
    assert result.pr_url == "https://github.com/o/r/pull/99"
    cmd = calls[0]
    assert "--head" in cmd
    assert cmd[cmd.index("--head") + 1] == "HAN-oQo:forager/o-r-1"


def test_approved_without_fork_owner_leaves_head_unprefixed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default (`fork_owner=None`) preserves prior behavior: a bare branch name, for a
    same-repo head."""
    store = _store_ready_for_gate(tmp_path)
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="https://github.com/o/r/pull/99\n")

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)

    _approve_then_submit(store, "o/r", 1)

    cmd = calls[0]
    assert cmd[cmd.index("--head") + 1] == "forager/o-r-1"


def test_gh_failure_still_returns_result_with_no_pr_url(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)

    def _raise(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="not found")

    monkeypatch.setattr(gate.subprocess, "run", _raise)

    result = _approve_then_submit(store, "o/r", 1)

    assert result is not None
    assert result.approved is True
    assert result.submitted is True
    assert result.pr_url is None


def test_approve_requires_the_literal_true_not_just_truthiness(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `approve` is checked with `is True`, not plain truthiness -- a caller
    passing any other truthy value (a non-empty string, an int) must NOT write a draft or open
    a PR, even with `submit=True` also passed."""
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    result = gate.run_gate(store, "o/r", 1, approve="yes", submit=True)  # type: ignore[arg-type]

    assert result is not None
    assert result.approved is False
    assert result.submitted is False
    assert result.pr_url is None
    assert result.draft_path is None


def test_run_gate_records_its_own_outcome(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="https://x/pull/1\n"),
    )

    _approve_then_submit(store, "o/r", 1)

    gate_runs = store.list_runs(repo="o/r", number=1, stage="gate")
    assert len(gate_runs) == 2  # one per call: first approve-only, then approve+submit
    assert gate_runs[-1]["approved"] is True
    assert gate_runs[-1]["submitted"] is True
    assert gate_runs[-1]["pr_url"] == "https://x/pull/1"
    assert gate_runs[-1]["draft_path"] is not None


def test_run_gate_records_fork_owner_in_its_own_outcome(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The KB's own audit trail for a cross-repo PR should show which fork the head branch was
    addressed against, not just that approval happened."""
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="https://x/pull/1\n"),
    )

    result = gate.run_gate(store, "o/r", 1, approve=True, submit=True, fork_owner="HAN-oQo")

    assert result.fork_owner == "HAN-oQo"
    gate_runs = store.list_runs(repo="o/r", number=1, stage="gate")
    assert gate_runs[0]["fork_owner"] == "HAN-oQo"


def test_run_gate_records_draft_path_and_unsubmitted_when_approved_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    gate.run_gate(store, "o/r", 1, approve=True)

    gate_runs = store.list_runs(repo="o/r", number=1, stage="gate")
    assert gate_runs[0]["approved"] is True
    assert gate_runs[0]["submitted"] is False
    assert gate_runs[0]["pr_url"] is None
    assert gate_runs[0]["draft_path"] is not None


def test_run_gate_records_outcome_even_when_unapproved(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    gate.run_gate(store, "o/r", 1, approve=False)

    gate_runs = store.list_runs(repo="o/r", number=1, stage="gate")
    assert len(gate_runs) == 1
    assert gate_runs[0]["approved"] is False
    assert gate_runs[0]["pr_url"] is None


# --------------------------------------------------------------------- assemble_bundle: skip paths


def test_returns_none_when_not_verified(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_gate(tmp_path, verified=False)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    assert gate.run_gate(store, "o/r", 1, approve=True) is None


def test_returns_none_when_self_review_does_not_advance(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T3.4's own 'Why': a hold must not reach the human's attention at all."""
    store = _store_ready_for_gate(tmp_path, advance=False)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    assert gate.run_gate(store, "o/r", 1, approve=True) is None


def test_returns_none_when_self_review_is_for_a_different_verify_attempt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a self-review whose verify_recorded_at doesn't match the current verify
    run's recorded_at must not be treated as evidence for THIS verify attempt."""
    store = _store_ready_for_gate(tmp_path, matching_verify_recorded_at=False)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    assert gate.run_gate(store, "o/r", 1, approve=True) is None


def test_returns_none_when_no_verify_run_at_all(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([{"repo": "o/r", "number": 1, "type": "issue", "title": "t"}])
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    assert gate.run_gate(store, "o/r", 1, approve=True) is None


def test_returns_none_when_item_missing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    assert gate.run_gate(store, "o/r", 404, approve=True) is None


def test_returns_none_when_both_timestamps_are_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a verify run missing `recorded_at` and a self_review run missing
    `verify_recorded_at` must NOT be treated as matching just because `None == None`."""
    store = JsonlStore(tmp_path)
    store.upsert_items([{"repo": "o/r", "number": 1, "type": "issue", "title": "t"}])
    store.record_run({"repo": "o/r", "number": 1, "stage": "verify", "verified": True})
    store.record_run(
        {"repo": "o/r", "number": 1, "stage": "self_review", "advance": True, "critiques": []}
    )
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    assert gate.run_gate(store, "o/r", 1, approve=True) is None


def test_bundle_uses_most_recent_reproduced_repro_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a later stray repro run with reproduced=False must not shadow an earlier
    reproduced=True run in the evidence bundle -- matching engineer.py's own baseline-selection
    convention (filter to reproduced=True, then take the latest)."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "repro",
            "command": "pytest test_other.py",
            "log": "no failure here\n",
            "reproduced": False,
            "recorded_at": "2026-01-05T00:00:00Z",  # newer than the reproduced=True run
        }
    )

    bundle = gate.assemble_bundle(store, "o/r", 1)

    assert bundle is not None
    assert bundle.repro_command == "pytest test_fp8.py"
    assert bundle.repro_log == "AssertionError\n"


# --------------------------------------------------------------------- assemble_bundle: content


def test_bundle_includes_risk_badge(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )

    bundle = gate.assemble_bundle(store, "o/r", 1)

    assert bundle is not None
    assert (bundle.risk, bundle.effort, bundle.impact) == ("low", "low", "medium")
    assert bundle.diff == "--- a/x.py\n+++ b/x.py\n"
    assert bundle.branch == "forager/o-r-1"
    assert bundle.evidence_url == "https://github.com/o/r/issues/1"
    assert bundle.verify_recorded_at == "2025-12-31T00:00:00Z"
    assert bundle.approve_count == 4
    assert bundle.total_votes == 5


def test_bundle_risk_badge_is_none_when_scoring_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate, "_score", lambda *a, **k: None)

    bundle = gate.assemble_bundle(store, "o/r", 1)

    assert bundle is not None
    assert (bundle.risk, bundle.effort, bundle.impact) == (None, None, None)


def test_bundle_format_includes_key_sections(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )
    bundle = gate.assemble_bundle(store, "o/r", 1)

    text = bundle.format()

    assert "o/r#1" in text
    assert "risk=low" in text
    assert "1 passed" in text  # verify log
    assert "AssertionError" in text  # repro log
    assert "diff" in text  # the patch fence


def test_critique_missing_reason_does_not_crash_format(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: format() reads each vote defensively -- a malformed critique record missing
    a field must degrade gracefully, not raise, since this is the one function whose entire job
    is safely surfacing evidence to the human approver."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "self_review",
            "critiques": [{"looks_correct": True}],  # missing "reason"
            "approve_count": 1,
            "total_votes": 1,
            "advance": True,
            "verify_recorded_at": "2025-12-31T00:00:00Z",
            "recorded_at": "2026-01-02T00:00:00Z",  # newer -- becomes the matching review
        }
    )

    bundle = gate.assemble_bundle(store, "o/r", 1)
    assert bundle is not None
    bundle.format()  # must not raise


# --------------------------------------------------------------------- verified_diff


def test_verified_diff_returns_diff_and_recorded_at_when_ready(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)

    def _fail_if_scored(*a, **k):
        raise AssertionError("_score should not be called by verified_diff")

    monkeypatch.setattr(gate, "_score", _fail_if_scored)

    result = gate.verified_diff(store, "o/r", 1)

    assert result == ("--- a/x.py\n+++ b/x.py\n", "2025-12-31T00:00:00Z")


def test_verified_diff_returns_none_when_not_verified(tmp_path) -> None:
    store = _store_ready_for_gate(tmp_path, verified=False)
    assert gate.verified_diff(store, "o/r", 1) is None


def test_verified_diff_returns_none_when_self_review_did_not_advance(tmp_path) -> None:
    store = _store_ready_for_gate(tmp_path, advance=False)
    assert gate.verified_diff(store, "o/r", 1) is None


# --------------------------------------------------------------------- T3.10.5: pr_body/quality


def test_finalize_uses_composed_pr_body_when_given(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )
    bundle = gate.assemble_bundle(store, "o/r", 1)
    assert bundle is not None

    result = gate._finalize(store, "o/r", 1, bundle, approve=True, pr_body="composed body")

    assert result.draft_path is not None
    assert result.draft_path.read_text() == "composed body"


def test_finalize_falls_back_to_bundle_format_when_no_pr_body_given(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )
    bundle = gate.assemble_bundle(store, "o/r", 1)
    assert bundle is not None

    result = gate._finalize(store, "o/r", 1, bundle, approve=True)

    assert result.draft_path is not None
    assert result.draft_path.read_text() == bundle.format()


def test_draft_is_new_when_content_changed_even_though_a_file_already_existed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `draft_is_new` must be content-based, not just file-existence-based -- a
    narrative recomposed (T3.9 rerun) between the required `--approve` and `--approve --submit`
    calls must force a fresh read, not silently submit the new content under the old
    'draft already existed' assumption."""
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)
    store = _store_ready_for_gate(tmp_path)

    gate.run_gate(store, "o/r", 1, approve=True, pr_body="narrative A")
    # A recomposition landed between the two calls -- different content, same candidate.
    result = gate.run_gate(
        store, "o/r", 1, approve=True, submit=True, pr_body="narrative B", quality_passed=True
    )

    assert result.draft_is_new is True  # forces a fresh read cycle, not a silent submit
    assert result.submitted is False
    assert result.draft_path.read_text() == "narrative B"


def test_draft_is_new_false_when_content_is_unchanged(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="https://x/pull/1\n"),
    )
    gate.run_gate(store, "o/r", 1, approve=True, pr_body="same narrative")
    result = gate.run_gate(
        store,
        "o/r",
        1,
        approve=True,
        submit=True,
        pr_body="same narrative",
        quality_passed=True,
    )
    assert result.draft_is_new is False
    assert result.submitted is True


# --------------------------------------------------------------------- T3.10.6: idempotency


def test_refuses_to_submit_when_candidate_already_has_an_open_pr(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core idempotency guard: a prior stage='gate' run already shows a real pr_url --
    a fresh --approve --submit must never call gh again, even with every other condition met
    and draft_is_new=True (e.g. the local draft cache was cleared)."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "gate",
            "approved": True,
            "submitted": True,
            "pr_url": "https://github.com/o/r/pull/99",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    result = gate.run_gate(
        store, "o/r", 1, approve=True, submit=True, pr_body="narrative", quality_passed=True
    )

    assert result.submitted is False
    assert result.already_open_pr_url == "https://github.com/o/r/pull/99"
    assert result.draft_is_new is True  # no local draft existed -- this alone must not matter


def test_already_open_pr_url_picks_the_most_recent_when_multiple_exist(tmp_path) -> None:
    """Regression: must use latest_run (deterministic by recorded_at), not a first-match loop
    over store.list_runs -- Store.list_runs' ordering isn't guaranteed on every backend."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "gate",
            "submitted": True,
            "pr_url": "https://github.com/o/r/pull/1",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "gate",
            "submitted": True,
            "pr_url": "https://github.com/o/r/pull/2",
            "recorded_at": "2026-02-01T00:00:00Z",
        }
    )
    assert gate._already_open_pr_url(store, "o/r", 1) == "https://github.com/o/r/pull/2"


def test_allows_submit_when_prior_gate_run_never_actually_submitted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prior stage='gate' run that was only approved (never submitted, or gh failed and left
    pr_url=None) must not block a later, real submission."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "gate",
            "approved": True,
            "submitted": False,
            "pr_url": None,
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="https://x/pull/1\n"),
    )

    result = _approve_then_submit(store, "o/r", 1)

    assert result.submitted is True
    assert result.already_open_pr_url is None


def test_cli_reports_existing_pr_url_instead_of_resubmitting(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "gate",
            "approved": True,
            "submitted": True,
            "pr_url": "https://github.com/o/r/pull/99",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    rc = gate.main(["--candidate", "o/r#1", "--approve", "--submit", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "already has an open PR: https://github.com/o/r/pull/99" in out


def test_cli_surfaces_existing_pr_even_without_submit_flag(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A human checking status (no --submit, or even no --approve) on an already-submitted
    candidate should still see the existing PR -- not only when they also pass --submit."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "gate",
            "approved": True,
            "submitted": True,
            "pr_url": "https://github.com/o/r/pull/99",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    rc = gate.main(["--candidate", "o/r#1", "--data-dir", str(tmp_path)])  # no flags at all

    assert rc == 0
    out = capsys.readouterr().out
    assert "already has an open PR: https://github.com/o/r/pull/99" in out
    assert "refusing to open a second one" not in out  # --submit wasn't even requested


def test_submit_stays_inert_when_quality_passed_is_false_on_a_later_call(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)
    store = _store_ready_for_gate(tmp_path)

    result = _approve_then_submit(store, "o/r", 1, quality_passed=False)

    assert result.submitted is False
    assert result.draft_is_new is False
    assert result.quality_passed is False


def test_submit_stays_inert_when_quality_passed_is_none_on_a_later_call(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)
    store = _store_ready_for_gate(tmp_path)

    result = _approve_then_submit(store, "o/r", 1, quality_passed=None)

    assert result.submitted is False
    assert result.quality_passed is None


_CURRENT_VERIFY_RECORDED_AT = "2025-12-31T00:00:00Z"  # matches _store_ready_for_gate's verify run


def test_current_narrative_returns_none_body_when_no_pr_author_run(tmp_path) -> None:
    store = _store_ready_for_gate(tmp_path)
    assert gate._current_narrative(
        store, "o/r", 1, current_verify_recorded_at=_CURRENT_VERIFY_RECORDED_AT
    ) == (None, None)


def test_current_narrative_returns_body_and_none_quality_when_no_pr_quality_run(
    tmp_path,
) -> None:
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "t",
            "body": "composed body",
            "verify_recorded_at": _CURRENT_VERIFY_RECORDED_AT,
            "recorded_at": "2026-01-02T00:00:00Z",
        }
    )
    assert gate._current_narrative(
        store, "o/r", 1, current_verify_recorded_at=_CURRENT_VERIFY_RECORDED_AT
    ) == ("composed body", None)


def test_current_narrative_ignores_a_stale_pr_quality_verdict(tmp_path) -> None:
    """A pr_quality run judging an OLDER pr_author run must not be read as covering the
    current (newer) one."""
    store = _store_ready_for_gate(tmp_path)
    _add_passing_narrative(store, pr_author_recorded_at="2025-01-01T00:00:00Z")  # stale verdict
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "t",
            "body": "newer composed body",
            "verify_recorded_at": _CURRENT_VERIFY_RECORDED_AT,
            "recorded_at": "2026-06-01T00:00:00Z",
        }
    )

    pr_body, quality_passed = gate._current_narrative(
        store, "o/r", 1, current_verify_recorded_at=_CURRENT_VERIFY_RECORDED_AT
    )

    assert pr_body == "newer composed body"
    assert quality_passed is None


def test_current_narrative_reads_kb_records_not_live_calls(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `_current_narrative` must never invoke `pr_author`/`pr_quality`'s actual LLM
    calls (it isn't even allowed to import those modules -- see module docstring) -- it only
    reads already-persisted KB records."""
    store = _store_ready_for_gate(tmp_path)
    _add_passing_narrative(store, body="composed body")
    monkeypatch.setattr(gate, "_score", _fail_if_gh_called)

    assert gate._current_narrative(
        store, "o/r", 1, current_verify_recorded_at=_CURRENT_VERIFY_RECORDED_AT
    ) == ("composed body", True)


def test_current_narrative_returns_none_when_narrative_composed_against_older_verify_run(
    tmp_path,
) -> None:
    """TOCTOU regression: a candidate re-verified (a new verify run landed) after T3.9/T3.10
    already ran must not have its stale narrative/verdict used -- neither the body nor the
    quality verdict describe the diff that's actually current now."""
    store = _store_ready_for_gate(tmp_path)
    _add_passing_narrative(store)  # composed against _CURRENT_VERIFY_RECORDED_AT

    pr_body, quality_passed = gate._current_narrative(
        store, "o/r", 1, current_verify_recorded_at="2026-06-01T00:00:00Z"  # a newer verify run
    )

    assert pr_body is None
    assert quality_passed is None


def test_current_narrative_returns_none_when_pr_author_recorded_at_missing(tmp_path) -> None:
    """Defense-in-depth: a malformed pr_author record missing `recorded_at` must not be treated
    as usable, even if it happens to have a body -- there's nothing to correlate a quality
    verdict against."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "t",
            "body": "composed body",
            "verify_recorded_at": _CURRENT_VERIFY_RECORDED_AT,
            "recorded_at": "",
        }
    )
    assert gate._current_narrative(
        store, "o/r", 1, current_verify_recorded_at=_CURRENT_VERIFY_RECORDED_AT
    ) == (None, None)


def test_current_narrative_returns_none_when_body_missing(tmp_path) -> None:
    """A pr_author record with a matching, valid recorded_at but an empty body must not report
    a body OR a quality verdict -- the two must never be independently valid."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "t",
            "body": "",
            "verify_recorded_at": _CURRENT_VERIFY_RECORDED_AT,
            "recorded_at": "2026-01-02T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_quality",
            "passes": True,
            "pr_author_recorded_at": "2026-01-02T00:00:00Z",
            "recorded_at": "2026-01-03T00:00:00Z",
        }
    )
    assert gate._current_narrative(
        store, "o/r", 1, current_verify_recorded_at=_CURRENT_VERIFY_RECORDED_AT
    ) == (None, None)


def test_current_narrative_does_not_coerce_a_truthy_non_bool_passes_value(tmp_path) -> None:
    """`is True`, not `bool(...)` -- a malformed store value like the string "False" is truthy
    in Python but must never be read as an accidental pass."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "t",
            "body": "composed body",
            "verify_recorded_at": _CURRENT_VERIFY_RECORDED_AT,
            "recorded_at": "2026-01-02T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_quality",
            "passes": "False",  # malformed: a non-empty (truthy) string, not the bool False
            "pr_author_recorded_at": "2026-01-02T00:00:00Z",
            "recorded_at": "2026-01-03T00:00:00Z",
        }
    )
    _, quality_passed = gate._current_narrative(
        store, "o/r", 1, current_verify_recorded_at=_CURRENT_VERIFY_RECORDED_AT
    )
    assert quality_passed is False


# --------------------------------------------------------------------- CLI ordering


def test_cli_first_approve_submit_together_does_not_submit(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The CLI-level version of the core T3.7 residual-gap fix: `--approve --submit` together
    on the very first invocation for a candidate must not open a PR."""
    _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    rc = gate.main(["--candidate", "o/r#1", "--approve", "--submit", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "NOT submitted" in out
    assert "first time" in out


def test_cli_prints_bundle_before_opening_pr(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Regression: CLAUDE.md's word is 'seeing' -- the bundle must appear in the human's
    terminal before any PR exists. Requires two separate invocations now (see
    `test_cli_first_approve_submit_together_does_not_submit`): the first creates the draft,
    the second (this test's actual assertion) submits it."""
    store = _store_ready_for_gate(tmp_path)  # populates the JsonlStore the CLI itself will open
    _add_passing_narrative(store)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="https://x/pull/1\n"),
    )
    gate.main(["--candidate", "o/r#1", "--approve", "--data-dir", str(tmp_path)])
    capsys.readouterr()  # discard the first call's output

    rc = gate.main(["--candidate", "o/r#1", "--approve", "--submit", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    bundle_pos = out.index("Candidate o/r#1")
    pr_pos = out.index("Opened PR")
    assert bundle_pos < pr_pos


def test_cli_approve_without_submit_never_calls_gh_and_prints_draft_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The CLI's own default-safe path: `--approve` alone never reaches `gh` (T3.7's whole
    point), and tells the human where the draft landed plus how to actually submit it."""
    _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    rc = gate.main(["--candidate", "o/r#1", "--approve", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Wrote PR draft:" in out
    assert "--submit" in out


def test_cli_submit_blocked_when_quality_gate_not_passed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """CLI-level regression for the quality-gate branch itself (not just the underlying
    run_gate() behavior) -- a narrative exists but its pr_quality verdict is passes=False, so a
    later --approve --submit call must print why and never call gh."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "t",
            "body": "composed body",
            "verify_recorded_at": "2025-12-31T00:00:00Z",
            "recorded_at": "2026-01-02T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_quality",
            "passes": False,
            "pr_author_recorded_at": "2026-01-02T00:00:00Z",
            "recorded_at": "2026-01-03T00:00:00Z",
        }
    )
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    gate.main(["--candidate", "o/r#1", "--approve", "--data-dir", str(tmp_path)])
    capsys.readouterr()
    rc = gate.main(["--candidate", "o/r#1", "--approve", "--submit", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "PR-quality gate (T3.10): NOT PASSED" in out
    assert "NOT submitted: no passing T3.10 PR-quality verdict" in out


def test_cli_writes_draft_under_the_given_data_dir_not_configs_default(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the draft's location must follow the same `--data-dir` override the store
    itself uses -- a caller pointed at a test/alternate store must not have its draft silently
    land in the real shared `config.DATA_DIR` instead (caught by this module's own Demo run)."""
    _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", _fail_if_gh_called)

    result_rc = gate.main(["--candidate", "o/r#1", "--approve", "--data-dir", str(tmp_path)])

    assert result_rc == 0
    draft = tmp_path / "pr_drafts" / "o-r-1.md"
    assert draft.exists()
    assert "o/r#1" in draft.read_text()


def test_cli_fork_owner_flag_prefixes_head(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--fork-owner` on the CLI reaches `gh pr create`'s `--head` the same way the
    programmatic `run_gate(..., fork_owner=...)` path does."""
    store = _store_ready_for_gate(tmp_path)
    _add_passing_narrative(store)
    calls = []

    def _fake_run(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="https://x/pull/1\n")

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)

    gate.main(["--candidate", "o/r#1", "--approve", "--data-dir", str(tmp_path)])
    gate.main(
        [
            "--candidate",
            "o/r#1",
            "--approve",
            "--submit",
            "--fork-owner",
            "HAN-oQo",
            "--data-dir",
            str(tmp_path),
        ]
    )

    cmd = calls[0]
    assert cmd[cmd.index("--head") + 1] == "HAN-oQo:forager/o-r-1"
