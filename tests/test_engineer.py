"""Tests for the engineer patch loop (T3.3) — offline & deterministic; llm + runner mocked.

Per the DEVPLAN todo: mock llm+runner -- fail->patch->pass => verified=True;
fail->patch->fail => verified=False and no PR (this module never creates a PR at all; that's
T3.5's job once T3.4's ensemble review has looked at a verified=True result).
"""

import subprocess
from datetime import datetime, timezone

import pytest

from src import engineer, llm, runner
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m3

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _store_with_baseline(tmp_path, *, reproduced=True, repo="o/r", number=1):
    """A store with an item and a prior T3.2 repro run -- the baseline run_engineer needs."""
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
            "stage": "repro",
            "host": "mi250-051",
            "command": "pytest test_fp8.py",
            "exit_code": 1 if reproduced else 0,
            "log": "AssertionError: fp8 mismatch\n",
            "reproduced": reproduced,
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    return store


def _fail_if_llm_called(*a, **k):
    raise AssertionError("llm.complete should not be called")


def _fail_if_runner_called(*a, **k):
    raise AssertionError("runner.run should not be called")


# --------------------------------------------------------------------- run_engineer: happy path


def test_run_engineer_verified_true_when_patched_run_passes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_baseline(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"patch": "--- a\n+++ b\n"})
    monkeypatch.setattr(
        runner, "run", lambda host, command, **k: runner.RunResult(exit_code=0, log="PASSED\n")
    )

    result = engineer.run_engineer(store, "o/r", 1, "mi250-051", now=_NOW)

    assert result is not None
    assert result.verified is True
    assert result.exit_code == 0

    runs = store.list_runs(repo="o/r", number=1, stage="verify")
    assert len(runs) == 1
    assert runs[0]["verified"] is True


def test_run_engineer_verified_false_when_patched_run_still_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fail -> patch -> fail => verified=False -- and since this module never creates a PR at
    all, "no PR" is automatically satisfied regardless of the outcome."""
    store = _store_with_baseline(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"patch": "--- a\n+++ b\n"})
    monkeypatch.setattr(
        runner,
        "run",
        lambda host, command, **k: runner.RunResult(
            exit_code=1, log="AssertionError: fp8 mismatch\n"
        ),
    )

    result = engineer.run_engineer(store, "o/r", 1, "mi250-051", now=_NOW)

    assert result is not None
    assert result.verified is False
    assert store.list_runs(repo="o/r", number=1, stage="verify")[0]["verified"] is False


# --------------------------------------------------------------------- run_engineer: skip paths


def test_run_engineer_returns_none_when_no_baseline(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([{"repo": "o/r", "number": 1, "type": "issue", "title": "t"}])
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)
    monkeypatch.setattr(runner, "run", _fail_if_runner_called)

    assert engineer.run_engineer(store, "o/r", 1, "mi250-051") is None


def test_run_engineer_returns_none_when_baseline_not_reproduced(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repro run exists but never actually reproduced the bug -- nothing to confirm a flip
    against."""
    store = _store_with_baseline(tmp_path, reproduced=False)
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)
    monkeypatch.setattr(runner, "run", _fail_if_runner_called)

    assert engineer.run_engineer(store, "o/r", 1, "mi250-051") is None


def test_run_engineer_returns_none_when_item_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "repro",
            "reproduced": True,
            "command": "pytest",
            "log": "x",
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)
    monkeypatch.setattr(runner, "run", _fail_if_runner_called)

    assert engineer.run_engineer(store, "o/r", 1, "mi250-051") is None


def test_run_engineer_returns_none_when_patch_synthesis_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_baseline(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"unexpected": "shape"})
    monkeypatch.setattr(runner, "run", _fail_if_runner_called)

    assert engineer.run_engineer(store, "o/r", 1, "mi250-051") is None


# --------------------------------------------------------------------- run_engineer: infra errors


def test_run_engineer_propagates_runner_error_without_recording(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_baseline(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"patch": "--- a\n+++ b\n"})

    def _raise(*a, **k):
        raise runner.RunnerError("ssh unreachable")

    monkeypatch.setattr(runner, "run", _raise)

    with pytest.raises(runner.RunnerError):
        engineer.run_engineer(store, "o/r", 1, "mi250-051")

    assert store.list_runs(stage="verify") == []


def test_run_engineer_returns_result_even_if_record_run_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_baseline(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"patch": "--- a\n+++ b\n"})
    monkeypatch.setattr(
        runner, "run", lambda host, command, **k: runner.RunResult(exit_code=0, log="PASSED\n")
    )

    def _raise(*a, **k):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(store, "record_run", _raise)

    result = engineer.run_engineer(store, "o/r", 1, "mi250-051", now=_NOW)

    assert result is not None
    assert result.verified is True


# --------------------------------------------------------------------- command composition


def test_verify_command_includes_branch_base_ref_and_patch() -> None:
    command = engineer._verify_command(
        "forager/o-r-1", "--- a\n+++ b\n", "pytest test_x.py", base_ref="main"
    )

    assert "git checkout -B forager/o-r-1 main" in command
    assert "--- a\n+++ b" in command
    assert "pytest test_x.py" in command


def test_verify_command_is_syntactically_valid_shell() -> None:
    """Regression: the original `cmd1 && cmd2 <<'EOF' ... EOF && cmd3` composition was a real
    bash syntax error (a heredoc's closing delimiter ends the enclosing command; `&&` on the
    next line has nothing to attach to) that every mocked test here was blind to. This pins the
    fix by actually asking a real shell to parse the composed script (`bash -n`, syntax check
    only -- no execution, no git/network needed)."""
    command = engineer._verify_command(
        "forager/o-r-1",
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n",
        "pytest test_x.py",
        base_ref="main",
    )

    result = subprocess.run(
        ["bash", "-n", "-c", command], capture_output=True, text=True, timeout=5
    )

    assert result.returncode == 0, result.stderr


def test_verify_command_stays_valid_shell_when_patch_contains_ampersand_lines() -> None:
    """The heredoc quoting must survive patch content that itself looks like shell syntax."""
    tricky_patch = "--- a\n+++ b\n@@ -1 +1 @@\n-old && rm -rf /\n+new || true\n"
    command = engineer._verify_command("forager/o-r-1", tricky_patch, "pytest x", base_ref="main")

    result = subprocess.run(
        ["bash", "-n", "-c", command], capture_output=True, text=True, timeout=5
    )

    assert result.returncode == 0, result.stderr


def test_verify_command_omits_cd_when_repo_dir_not_given() -> None:
    """Default (`repo_dir=None`) preserves prior behavior: no `cd`, script starts with `set -e`."""
    command = engineer._verify_command(
        "forager/o-r-1", "--- a\n+++ b\n", "pytest test_x.py", base_ref="main"
    )

    assert command.startswith("set -e")


def test_verify_command_cds_into_repo_dir_after_set_dash_e_but_before_git() -> None:
    """Regression: plain `ssh host command` lands in the ssh session's default directory, not
    necessarily the checkout -- without this `cd`, every git command below it silently runs
    against whatever directory ssh happened to default to. The `cd` must come *after* `set -e`,
    not before -- a `cd` that fails before `set -e` takes effect does not abort the script
    (verified directly: `bash -c "cd /no-such-dir; set -e; echo reached"` prints "reached" and
    exits 0), which would silently defeat this fix's whole purpose."""
    command = engineer._verify_command(
        "forager/o-r-1",
        "--- a\n+++ b\n",
        "pytest test_x.py",
        base_ref="main",
        repo_dir="/remote/vast0/herom/vllm",
    )

    lines = command.splitlines()
    assert lines[0] == "set -e"
    assert lines.index("cd /remote/vast0/herom/vllm") == 1
    assert lines.index("cd /remote/vast0/herom/vllm") < lines.index(
        "git checkout -B forager/o-r-1 main"
    )


def test_verify_command_aborts_at_failed_cd_without_running_git_operations() -> None:
    """Direct execution proof of the ordering regression above: a nonexistent `repo_dir` must
    make the whole script fail before any git command runs, not after."""
    command = engineer._verify_command(
        "forager/o-r-1",
        "--- a\n+++ b\n",
        "pytest test_x.py",
        base_ref="main",
        repo_dir="/no/such/directory/xyz",
    )

    result = subprocess.run(
        ["bash", "-x", "-c", command], capture_output=True, text=True, timeout=5
    )

    assert result.returncode != 0
    assert "+ git checkout" not in result.stderr


def test_verify_command_quotes_repo_dir_containing_a_space() -> None:
    command = engineer._verify_command(
        "forager/o-r-1",
        "--- a\n+++ b\n",
        "pytest test_x.py",
        base_ref="main",
        repo_dir="/data/vllm forager",
    )

    assert "cd '/data/vllm forager'" in command.splitlines()


def test_verify_command_with_repo_dir_is_syntactically_valid_shell() -> None:
    command = engineer._verify_command(
        "forager/o-r-1",
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n",
        "pytest test_x.py",
        base_ref="main",
        repo_dir="/remote/vast0/herom/vllm",
    )

    result = subprocess.run(
        ["bash", "-n", "-c", command], capture_output=True, text=True, timeout=5
    )

    assert result.returncode == 0, result.stderr


def test_branch_name_includes_repo_to_avoid_cross_repo_collision() -> None:
    assert engineer._branch_name("vllm-project/vllm", 42) != engineer._branch_name("ROCm/vllm", 42)


def test_run_engineer_uses_baseline_command_to_rerun(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_baseline(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"patch": "--- a\n+++ b\n"})
    calls = []

    def _fake_run(host, command, **k):
        calls.append(command)
        return runner.RunResult(exit_code=0, log="PASSED\n")

    monkeypatch.setattr(runner, "run", _fake_run)

    engineer.run_engineer(store, "o/r", 1, "mi250-051")

    assert "pytest test_fp8.py" in calls[0]  # the same command T3.2's baseline originally ran


def test_run_engineer_passes_repo_dir_through_to_verify_command(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_baseline(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"patch": "--- a\n+++ b\n"})
    calls = []

    def _fake_run(host, command, **k):
        calls.append(command)
        return runner.RunResult(exit_code=0, log="PASSED\n")

    monkeypatch.setattr(runner, "run", _fake_run)

    engineer.run_engineer(store, "o/r", 1, "mi250-051", repo_dir="/remote/vast0/herom/vllm")

    assert "cd /remote/vast0/herom/vllm" in calls[0].splitlines()


# --------------------------------------------------------------------- synthesize_patch


def test_synthesize_patch_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"patch": "  --- a\n+++ b\n  "})
    assert engineer.synthesize_patch("t", "b", "log") == "--- a\n+++ b"


def test_synthesize_patch_rejects_blank_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"patch": "   "})
    assert engineer.synthesize_patch("t", "b", "log") is None


def test_synthesize_patch_rejects_non_dict_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "not a dict")
    assert engineer.synthesize_patch("t", "b", "log") is None


def test_synthesize_patch_returns_none_on_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a, **k):
        raise llm.LLMError("provider unavailable")

    monkeypatch.setattr(llm, "complete", _raise)
    assert engineer.synthesize_patch("t", "b", "log") is None
