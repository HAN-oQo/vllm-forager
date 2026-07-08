"""Tests for the Orchestrator (T4.1) — offline & deterministic.

Per the DEVPLAN todo: a fake clock + fake agents (never the real `collector.main`/`analyze.main`
CLIs, no network/LLM anywhere here) exercise cadence routing (data plane daily / intel weekly)
and trigger-gated routing (contribution only when its own trigger says so).
"""

from datetime import datetime, timedelta, timezone

import pytest

from src import orchestrator, policy, taxonomy
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m4

_NOW = datetime(2026, 1, 8, tzinfo=timezone.utc)  # arbitrary fixed instant


@pytest.fixture
def store(tmp_path) -> JsonlStore:
    s = JsonlStore(tmp_path)
    taxonomy.create_taxonomy(s, ["rocm-build"])
    policy.create_policy(s, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1)
    return s


def _recording_stage(name: str, calls: list[str], **kwargs) -> orchestrator.Stage:
    return orchestrator.Stage(name, lambda _store: calls.append(name), **kwargs)


# --------------------------------------------------------------------- Stage validation


def test_stage_requires_interval_or_trigger() -> None:
    with pytest.raises(orchestrator.OrchestratorError, match="neither interval_hours nor trigger"):
        orchestrator.Stage("x", lambda _store: None)


# --------------------------------------------------------------------- cadence routing


def test_run_tick_runs_a_cadence_stage_that_has_never_run(store: JsonlStore) -> None:
    calls: list[str] = []
    stage = _recording_stage("collect", calls, interval_hours=24)

    result = orchestrator.run_tick(store, [stage], now=_NOW)

    assert calls == ["collect"]
    assert result.ran == ("collect",)
    assert result.skipped == {}


def test_run_tick_skips_a_cadence_stage_not_yet_due(store: JsonlStore) -> None:
    calls: list[str] = []
    stage = _recording_stage("collect", calls, interval_hours=24)
    orchestrator.run_tick(store, [stage], now=_NOW)  # first tick: runs, records last-run
    calls.clear()

    result = orchestrator.run_tick(store, [stage], now=_NOW + timedelta(hours=1))

    assert calls == []
    assert result.ran == ()
    assert "collect" in result.skipped


def test_run_tick_reruns_a_cadence_stage_once_its_interval_elapses(store: JsonlStore) -> None:
    calls: list[str] = []
    stage = _recording_stage("collect", calls, interval_hours=24)
    orchestrator.run_tick(store, [stage], now=_NOW)
    calls.clear()

    result = orchestrator.run_tick(store, [stage], now=_NOW + timedelta(hours=24))

    assert calls == ["collect"]
    assert result.ran == ("collect",)


def test_run_tick_data_plane_daily_and_intel_weekly_are_independent_cadences(
    store: JsonlStore,
) -> None:
    """DEVPLAN's own worked example: collect (daily) and classify+report (weekly) are on
    different clocks — a day passing runs collect again, but not yet intel."""
    calls: list[str] = []
    collect = _recording_stage("collect", calls, interval_hours=24)
    intel = _recording_stage("intel", calls, interval_hours=24 * 7)
    orchestrator.run_tick(store, [collect, intel], now=_NOW)
    calls.clear()

    result = orchestrator.run_tick(store, [collect, intel], now=_NOW + timedelta(days=1))

    assert calls == ["collect"]
    assert result.ran == ("collect",)
    assert "intel" in result.skipped

    result = orchestrator.run_tick(store, [collect, intel], now=_NOW + timedelta(days=8))
    assert set(result.ran) == {"collect", "intel"}


# --------------------------------------------------------------------- trigger routing


def test_run_tick_runs_a_triggered_stage_only_when_triggered(store: JsonlStore) -> None:
    calls: list[str] = []
    stage = orchestrator.Stage(
        "contribution", lambda _store: calls.append("contribution"), trigger=lambda _store: True
    )

    result = orchestrator.run_tick(store, [stage], now=_NOW)

    assert calls == ["contribution"]
    assert result.ran == ("contribution",)


def test_run_tick_skips_a_triggered_stage_with_no_trigger(store: JsonlStore) -> None:
    """DEVPLAN's own worked example: 'skips contribution unless a candidate triggers it' --
    the gate is respected, not just cadence."""
    calls: list[str] = []
    stage = orchestrator.Stage(
        "contribution", lambda _store: calls.append("contribution"), trigger=lambda _store: False
    )

    result = orchestrator.run_tick(store, [stage], now=_NOW)

    assert calls == []
    assert result.skipped == {"contribution": "no trigger"}


def test_run_tick_triggered_stage_has_no_cadence_cursor(store: JsonlStore) -> None:
    """A triggered stage running twice in a row (still triggered both times) must not be
    blocked by a cadence cursor it was never supposed to have."""
    calls: list[str] = []
    stage = orchestrator.Stage(
        "contribution", lambda _store: calls.append("contribution"), trigger=lambda _store: True
    )

    orchestrator.run_tick(store, [stage], now=_NOW)
    orchestrator.run_tick(store, [stage], now=_NOW)  # same instant, still triggered

    assert calls == ["contribution", "contribution"]


# --------------------------------------------------------------------- policy pinning


def test_run_tick_pins_and_reports_the_active_policy_version(store: JsonlStore) -> None:
    policy.update_policy(store, scoring_weights={"risk": 2.0})  # -> policy@2, now active

    result = orchestrator.run_tick(store, [], now=_NOW)

    assert result.policy_version == 2


def test_run_tick_raises_if_no_policy_exists(tmp_path) -> None:
    store = JsonlStore(tmp_path)  # no create_policy() call
    with pytest.raises(policy.PolicyError):
        orchestrator.run_tick(store, [], now=_NOW)


# --------------------------------------------------------------------- ordering / isolation


def test_run_tick_runs_stages_in_their_given_order(store: JsonlStore) -> None:
    calls: list[str] = []
    stages = [
        _recording_stage("collect", calls, interval_hours=24),
        orchestrator.Stage(
            "contribution", lambda _store: calls.append("contribution"), trigger=lambda _s: True
        ),
    ]

    orchestrator.run_tick(store, stages, now=_NOW)

    assert calls == ["collect", "contribution"]


def test_run_tick_one_stage_failing_does_not_block_earlier_stages_from_being_recorded(
    store: JsonlStore,
) -> None:
    """A later stage raising must not roll back an earlier stage's already-recorded run --
    each stage's cursor commits independently as it completes."""
    calls: list[str] = []
    ok_stage = _recording_stage("collect", calls, interval_hours=24)

    def _boom(_store) -> None:
        raise RuntimeError("boom")

    bad_stage = orchestrator.Stage("intel", _boom, interval_hours=24)

    with pytest.raises(RuntimeError, match="boom"):
        orchestrator.run_tick(store, [ok_stage, bad_stage], now=_NOW)

    assert calls == ["collect"]
    # collect's cursor was persisted even though the tick as a whole raised later
    assert store.get_state("orchestrator:last_run:collect") is not None
