"""Tests for the repro harness (T3.2) — offline & deterministic; llm + runner mocked.

Per the DEVPLAN todo: mock runner returns a failing log -> failing signal recorded.
"""

from datetime import datetime, timezone

import pytest

from src import llm, repro, runner
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m3

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _store_with_item(tmp_path, repo="o/r", number=1, **overrides):
    store = JsonlStore(tmp_path)
    item = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": "vLLM crashes on gfx90a with fp8",
        "body": "Running fp8 quant on MI250 raises an assertion.",
        "state": "open",
    }
    item.update(overrides)
    store.upsert_items([item])
    return store


def _fail_if_llm_called(*a, **k):
    raise AssertionError("llm.complete should not be called")


def _fail_if_runner_called(*a, **k):
    raise AssertionError("runner.run should not be called")


# --------------------------------------------------------------------- run_repro: happy path


def test_run_repro_records_failing_signal(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"command": "pytest test_fp8.py"})
    monkeypatch.setattr(
        runner,
        "run",
        lambda host, command, **k: runner.RunResult(
            exit_code=1, log="AssertionError: fp8 mismatch\n"
        ),
    )

    result = repro.run_repro(store, "o/r", 1, "mi250-051", now=_NOW)

    assert result is not None
    assert result.reproduced is True
    assert result.exit_code == 1
    assert result.command == "pytest test_fp8.py"
    assert "AssertionError" in result.log

    runs = store.list_runs(repo="o/r", number=1)
    assert len(runs) == 1
    assert runs[0]["stage"] == "repro"
    assert runs[0]["reproduced"] is True
    assert runs[0]["exit_code"] == 1
    assert runs[0]["host"] == "mi250-051"


def test_run_repro_records_non_reproducing_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"command": "pytest test_fp8.py"})
    monkeypatch.setattr(
        runner, "run", lambda host, command, **k: runner.RunResult(exit_code=0, log="ok\n")
    )

    result = repro.run_repro(store, "o/r", 1, "mi250-051", now=_NOW)

    assert result is not None
    assert result.reproduced is False
    assert store.list_runs(repo="o/r", number=1)[0]["reproduced"] is False


# --------------------------------------------------------------------- run_repro: skip paths


def test_run_repro_returns_none_when_item_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(llm, "complete", _fail_if_llm_called)
    monkeypatch.setattr(runner, "run", _fail_if_runner_called)

    assert repro.run_repro(store, "o/r", 404, "mi250-051") is None
    assert store.list_runs() == []


def test_run_repro_returns_none_when_synthesis_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_item(tmp_path)

    def _raise(*a, **k):
        raise llm.LLMError("provider unavailable")

    monkeypatch.setattr(llm, "complete", _raise)
    monkeypatch.setattr(runner, "run", _fail_if_runner_called)

    assert repro.run_repro(store, "o/r", 1, "mi250-051") is None
    assert store.list_runs() == []


def test_run_repro_returns_none_when_reply_has_no_command(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"unexpected": "shape"})
    monkeypatch.setattr(runner, "run", _fail_if_runner_called)

    assert repro.run_repro(store, "o/r", 1, "mi250-051") is None


# --------------------------------------------------------------------- run_repro: infra errors


def test_run_repro_propagates_runner_error_without_recording(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ssh/timeout failure is an infra problem, not 'no signal' -- it must not be silently
    swallowed as a skip, and nothing partial should be recorded."""
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"command": "pytest test_fp8.py"})

    def _raise(*a, **k):
        raise runner.RunnerError("ssh unreachable")

    monkeypatch.setattr(runner, "run", _raise)

    with pytest.raises(runner.RunnerError):
        repro.run_repro(store, "o/r", 1, "mi250-051")

    assert store.list_runs() == []


def test_run_repro_propagates_ssh_transport_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Specifically pins runner.py's own exit-255 (ssh transport failure, not a remote-command
    result) RunnerError -- distinct from a generic infra error -- still propagates uncaught."""
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"command": "pytest test_fp8.py"})

    def _raise(*a, **k):
        raise runner.RunnerError("ssh to 'mi250-051' failed (exit 255) — Permission denied")

    monkeypatch.setattr(runner, "run", _raise)

    with pytest.raises(runner.RunnerError, match="exit 255"):
        repro.run_repro(store, "o/r", 1, "mi250-051")


def test_run_repro_passes_timeout_to_runner(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"command": "pytest test_fp8.py"})

    def _fake_run(host, command, *, timeout=None, **k):
        calls.append(timeout)
        return runner.RunResult(exit_code=0, log="ok\n")

    monkeypatch.setattr(runner, "run", _fake_run)

    repro.run_repro(store, "o/r", 1, "mi250-051")
    assert calls == [repro.DEFAULT_REPRO_TIMEOUT_S]

    repro.run_repro(store, "o/r", 1, "mi250-051", timeout=30.0)
    assert calls[-1] == 30.0


def test_run_repro_returns_result_even_if_record_run_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A KB write failure must not discard an already-completed MI250 result -- redoing a real
    hardware run just to recover an outcome this process already has would be wasteful."""
    store = _store_with_item(tmp_path)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"command": "pytest test_fp8.py"})
    monkeypatch.setattr(
        runner, "run", lambda host, command, **k: runner.RunResult(exit_code=1, log="boom\n")
    )

    def _raise(*a, **k):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(store, "record_run", _raise)

    result = repro.run_repro(store, "o/r", 1, "mi250-051", now=_NOW)

    assert result is not None
    assert result.reproduced is True
    assert result.log == "boom\n"


# --------------------------------------------------------------------- ReproResult.to_run_record


def test_to_run_record_truncates_persisted_log_but_result_keeps_full_log() -> None:
    huge_log = "x" * (repro._MAX_PERSISTED_LOG_CHARS + 1000) + "TAIL"
    result = repro.ReproResult(
        repo="o/r",
        number=1,
        host="mi250-051",
        command="pytest",
        exit_code=1,
        log=huge_log,
        reproduced=True,
        recorded_at="2026-01-01T00:00:00Z",
    )

    record = result.to_run_record()

    assert len(record["log"]) == repro._MAX_PERSISTED_LOG_CHARS
    assert record["log"].endswith("TAIL")
    assert result.log == huge_log  # the in-memory result itself is never truncated


# --------------------------------------------------------------------- synthesize_repro_command


def test_synthesize_repro_command_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"command": "  pytest test_x.py  "})
    assert repro.synthesize_repro_command("t", "b") == "pytest test_x.py"


def test_synthesize_repro_command_rejects_blank_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"command": "   "})
    assert repro.synthesize_repro_command("t", "b") is None


def test_synthesize_repro_command_rejects_non_dict_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "not a dict")
    assert repro.synthesize_repro_command("t", "b") is None
