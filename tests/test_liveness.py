"""Tests for liveness (T4.5) — offline & deterministic.

Per the DEVPLAN todo: a stage emits started -> heartbeat -> finished with monotonic
timestamps; a stale heartbeat is classified `stalled`; an exception path records `failed`
(nothing left silently "running").

Every run-event timestamp `run_tick`/`emit_heartbeat` write always samples the real wall
clock (see `src/orchestrator.py`'s module docstring: an earlier version stamped the terminal
event with the tick's own `now` instead, which could make a stage's terminal timestamp sort
*before* its own `started` timestamp for a multi-stage tick with an earlier slow stage -- a
real bug, not a test artifact, fixed before this PR merged). `now` only affects cadence math
now, not event timestamps, so the `run_tick` tests below don't need to fix it. The pure
`liveness.latest_status` unit tests, which construct `runs` dicts by hand instead of going
through `run_tick`, choose their own fixed clock for those hand-built timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src import liveness, orchestrator, policy, taxonomy
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m4


@pytest.fixture
def store(tmp_path) -> JsonlStore:
    s = JsonlStore(tmp_path)
    taxonomy.create_taxonomy(s, ["rocm-build"])
    policy.create_policy(s, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1)
    return s


# --------------------------------------------------------------------- write side (run_tick)


def test_run_tick_emits_started_then_heartbeat_then_a_terminal_event_in_order(
    store: JsonlStore,
) -> None:
    """DEVPLAN's own worked example: a stage emits started -> heartbeat -> finished with
    monotonic timestamps. No fixed `now` needed here -- every run-event timestamp samples the
    real wall clock regardless of what `now` is (see this file's own module docstring)."""

    def _work(s: JsonlStore) -> int:
        orchestrator.emit_heartbeat(s, stage="collect", step="paging", output_tail="fetched 10")
        return 10

    stage = orchestrator.Stage("collect", _work, interval_hours=24)

    orchestrator.run_tick(store, [stage])

    runs = store.list_runs(stage="collect")
    assert [r["status"] for r in runs] == ["started", "heartbeat", "ok"]
    timestamps = [r["recorded_at"] for r in runs]
    assert timestamps == sorted(timestamps)  # monotonic (non-decreasing)


def test_run_tick_emits_a_started_event_before_a_failing_stage_runs(store: JsonlStore) -> None:
    def _boom(_s: JsonlStore) -> None:
        raise RuntimeError("boom")

    stage = orchestrator.Stage("intel", _boom, interval_hours=24)

    with pytest.raises(RuntimeError, match="boom"):
        orchestrator.run_tick(store, [stage])

    statuses = [r["status"] for r in store.list_runs(stage="intel")]
    assert statuses == ["started", "failed"]  # nothing left silently "running"


def test_emit_heartbeat_carries_step_output_tail_and_policy_version(
    store: JsonlStore,
) -> None:
    orchestrator.emit_heartbeat(store, stage="intel", step="classifying", output_tail="42/100")

    run = store.list_runs(stage="intel")[0]
    assert run["status"] == "heartbeat"
    assert run["step"] == "classifying"
    assert run["output_tail"] == "42/100"
    assert run["policy_version"] == 1


def test_emit_heartbeat_survives_a_record_run_failure(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        store, "record_run", lambda _r: (_ for _ in ()).throw(RuntimeError("disk full"))
    )

    orchestrator.emit_heartbeat(store, stage="intel", step="classifying")  # must not raise

    assert "failed to record heartbeat" in capsys.readouterr().err


def test_emit_heartbeat_survives_no_active_policy(tmp_path, capsys) -> None:
    """A heartbeat is inherently best-effort -- a PolicyError from the version lookup must
    degrade the same way a record_run failure does, not propagate uncaught."""
    store = JsonlStore(tmp_path)  # no policy.create_policy() call

    orchestrator.emit_heartbeat(store, stage="intel", step="classifying")  # must not raise

    assert "failed to record heartbeat" in capsys.readouterr().err
    assert store.list_runs() == []  # never attempted the write -- there was nothing to record


def test_emit_heartbeat_accepts_an_explicit_policy_version_to_skip_the_lookup(
    store: JsonlStore,
) -> None:
    """A future caller that already knows the tick's own pinned version (run_tick computes it
    once per tick) can pass it directly instead of paying for a fresh get_active(store) lookup
    -- update_policy() to version 2 here, then confirm passing policy_version=1 explicitly is
    honored rather than silently overridden by the (now different) active version."""
    policy.update_policy(store, scoring_weights={"risk": 2.0})  # now active: policy@2

    orchestrator.emit_heartbeat(store, stage="intel", step="classifying", policy_version=1)

    assert store.list_runs(stage="intel")[0]["policy_version"] == 1


# --------------------------------------------------------------------- read side (liveness)

_NOW = datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)


def test_latest_status_never_run_with_no_runs() -> None:
    assert liveness.latest_status([], now=_NOW, stale_after_s=600) == "never run"


def test_latest_status_classifies_a_recent_heartbeat_as_running() -> None:
    runs = [{"status": "heartbeat", "recorded_at": "2026-01-08T11:55:00Z"}]  # 5 min old
    assert liveness.latest_status(runs, now=_NOW, stale_after_s=600) == "running"


def test_latest_status_classifies_a_stale_heartbeat_as_stalled() -> None:
    """DEVPLAN's own worked example: a stale heartbeat is classified `stalled`."""
    runs = [{"status": "heartbeat", "recorded_at": "2026-01-08T11:00:00Z"}]  # 1h old
    assert liveness.latest_status(runs, now=_NOW, stale_after_s=600) == "stalled"


def test_latest_status_classifies_a_stale_started_event_as_stalled() -> None:
    """A stage that never gets past `started` (dies before its first heartbeat, or never
    heartbeats at all) must still go stale -- `started` isn't exempt from the staleness check
    just because it isn't a `heartbeat`."""
    runs = [{"status": "started", "recorded_at": "2026-01-08T11:00:00Z"}]
    assert liveness.latest_status(runs, now=_NOW, stale_after_s=600) == "stalled"


def test_latest_status_terminal_events_ignore_staleness() -> None:
    """An old but genuinely-finished run must read as `"ok"`, not `"stalled"` -- staleness is
    about a lifecycle event with no successor yet, not about how long ago anything happened."""
    runs = [{"status": "ok", "recorded_at": "2020-01-01T00:00:00Z"}]
    assert liveness.latest_status(runs, now=_NOW, stale_after_s=600) == "ok"


def test_latest_status_a_failed_run_is_never_running_or_stalled() -> None:
    """DEVPLAN's own worked example: an exception path records `failed` (nothing left
    silently "running")."""
    runs = [
        {"status": "started", "recorded_at": "2026-01-08T11:00:00Z"},
        {"status": "failed", "recorded_at": "2026-01-08T11:00:01Z"},
    ]
    assert liveness.latest_status(runs, now=_NOW, stale_after_s=600) == "failed"


def test_latest_status_picks_the_most_recent_event_not_the_first(store: JsonlStore) -> None:
    runs = [
        {"status": "started", "recorded_at": "2026-01-08T11:00:00Z"},
        {"status": "heartbeat", "recorded_at": "2026-01-08T11:59:00Z"},
    ]
    assert liveness.latest_status(runs, now=_NOW, stale_after_s=600) == "running"


def test_latest_status_unparseable_timestamp_is_stalled_not_running() -> None:
    """Can't confirm freshness without a parseable timestamp -- fail toward "needs attention"."""
    runs = [{"status": "heartbeat", "recorded_at": "not-a-timestamp"}]
    assert liveness.latest_status(runs, now=_NOW, stale_after_s=600) == "stalled"


def test_latest_status_a_terminal_event_wins_a_tie_at_the_same_recorded_at() -> None:
    """Real same-second ties are possible -- every run-event timestamp has only
    second-resolution, so a fast stage's own `started` and terminal events can legitimately
    land in the same rendered second. `stages.latest_run`'s own `max()` alone would pick
    whichever happens to be first in `runs`' own order (Python `max()` semantics); the
    terminal-status tie-break must win regardless of list order."""
    tied_at = "2026-01-08T11:00:00Z"
    heartbeat_first = [
        {"status": "heartbeat", "recorded_at": tied_at},
        {"status": "ok", "recorded_at": tied_at},
    ]
    ok_first = [
        {"status": "ok", "recorded_at": tied_at},
        {"status": "heartbeat", "recorded_at": tied_at},
    ]
    assert liveness.latest_status(heartbeat_first, now=_NOW, stale_after_s=600) == "ok"
    assert liveness.latest_status(ok_first, now=_NOW, stale_after_s=600) == "ok"


def test_latest_status_and_record_returns_the_record_status_was_based_on() -> None:
    """A caller needing fields off the exact record (T5.8's health panel) must get the same
    one `latest_status` classified from, including on a same-second tie -- not re-derive
    'latest' a second, tie-unaware way (a code-review finding on an earlier dashboard.health)."""
    tied_at = "2026-01-08T11:00:00Z"
    runs = [
        {"status": "heartbeat", "recorded_at": tied_at, "step": "working"},
        {"status": "failed", "recorded_at": tied_at, "error": "boom"},
    ]

    status, record = liveness.latest_status_and_record(runs, now=_NOW, stale_after_s=600)

    assert status == "failed"
    assert record is not None
    assert record["error"] == "boom"


def test_latest_status_and_record_no_runs_returns_none() -> None:
    assert liveness.latest_status_and_record([], now=_NOW, stale_after_s=600) == ("never run", None)


def test_run_tick_and_liveness_integration_classifies_a_completed_tick_as_ok(
    store: JsonlStore,
) -> None:
    """End-to-end: run_tick's own written events, read back through liveness.latest_status."""
    stage = orchestrator.Stage("collect", lambda _s: 1, interval_hours=24)

    orchestrator.run_tick(store, [stage])

    runs = store.list_runs(stage="collect")
    classify_at = datetime.now(timezone.utc) + timedelta(seconds=1)
    assert liveness.latest_status(runs, now=classify_at, stale_after_s=600) == "ok"
