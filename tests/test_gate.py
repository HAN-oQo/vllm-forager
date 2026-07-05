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


def _fail_if_gh_called(*a, **k):
    raise AssertionError("gh (subprocess.run) should not be called")


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

    result = gate.run_gate(store, "o/r", 1, approve=True, submit=True)

    assert result is not None
    assert result.approved is True
    assert result.submitted is True
    assert result.pr_url == "https://github.com/o/r/pull/99"
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:4] == ["gh", "pr", "create", "--draft"]
    assert "--repo" in cmd and "o/r" in cmd
    assert "--head" in cmd and "forager/o-r-1" in cmd


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

    result = gate.run_gate(store, "o/r", 1, approve=True, submit=True, fork_owner="HAN-oQo")

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

    gate.run_gate(store, "o/r", 1, approve=True, submit=True)

    cmd = calls[0]
    assert cmd[cmd.index("--head") + 1] == "forager/o-r-1"


def test_gh_failure_still_returns_result_with_no_pr_url(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)

    def _raise(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="not found")

    monkeypatch.setattr(gate.subprocess, "run", _raise)

    result = gate.run_gate(store, "o/r", 1, approve=True, submit=True)

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

    gate.run_gate(store, "o/r", 1, approve=True, submit=True)

    gate_runs = store.list_runs(repo="o/r", number=1, stage="gate")
    assert len(gate_runs) == 1
    assert gate_runs[0]["approved"] is True
    assert gate_runs[0]["submitted"] is True
    assert gate_runs[0]["pr_url"] == "https://x/pull/1"
    assert gate_runs[0]["draft_path"] is not None


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


# --------------------------------------------------------------------- CLI ordering


def test_cli_prints_bundle_before_opening_pr(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Regression: CLAUDE.md's word is 'seeing' -- the bundle must appear in the human's
    terminal before any PR exists, even on a single `--candidate X --approve --submit`
    invocation."""
    _store_ready_for_gate(tmp_path)  # populates the JsonlStore the CLI itself will open
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="https://x/pull/1\n"),
    )

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
    _store_ready_for_gate(tmp_path)
    calls = []

    def _fake_run(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="https://x/pull/1\n")

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)

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
