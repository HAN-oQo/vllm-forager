"""Tests for the dashboard cost panel (T5.13) — offline & deterministic.

Per the DEVPLAN todo: seeded cost records → per-agent/day series + a budget-exceeded flag.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dashboard.cost import cost_panel
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5

_NOW = datetime(2026, 1, 8, 12, tzinfo=timezone.utc)


def _record(store: JsonlStore, **overrides) -> None:
    base = {
        "stage": "cost",
        "agent": "analyst",
        "run_id": "r1",
        "loop": "intel",
        "provider": "claude_api",
        "model": "claude-sonnet-5",
        "tokens_in": 100,
        "tokens_out": 50,
        "tokens_cache": 0,
        "cost_usd": 1.0,
        "recorded_at": "2026-01-08T00:00:00Z",
    }
    base.update(overrides)
    store.record_run(base)


def test_cost_panel_empty_store(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)

    panel = cost_panel(store, now=_NOW)

    assert panel["agents"] == []
    assert panel["today_total_usd"] == 0.0
    assert panel["budget_exceeded"] is False


def test_cost_panel_buckets_today_week_and_total_per_agent(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _record(store, agent="analyst", cost_usd=2.0, recorded_at="2026-01-08T00:00:00Z")  # today
    _record(store, agent="analyst", cost_usd=3.0, recorded_at="2026-01-05T00:00:00Z")  # in week
    _record(store, agent="analyst", cost_usd=4.0, recorded_at="2025-12-01T00:00:00Z")  # old

    panel = cost_panel(store, now=_NOW)

    assert len(panel["agents"]) == 1
    analyst = panel["agents"][0]
    assert analyst["agent"] == "analyst"
    assert analyst["today_usd"] == 2.0
    assert analyst["week_usd"] == 5.0  # today + the-Jan-5 record, both within the trailing 7d
    assert analyst["total_usd"] == 9.0  # every record, including the December one


def test_cost_panel_breaks_down_by_model_and_provider(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _record(store, agent="analyst", model="claude-sonnet-5", provider="claude_api", cost_usd=1.0)
    _record(store, agent="analyst", model="claude-opus-4-8", provider="claude_api", cost_usd=5.0)

    panel = cost_panel(store, now=_NOW)

    by_model = panel["agents"][0]["by_model"]
    assert [m["model"] for m in by_model] == ["claude-opus-4-8", "claude-sonnet-5"]  # cost desc
    assert by_model[0]["cost_usd"] == 5.0
    assert by_model[0]["calls"] == 1


def test_cost_panel_includes_the_claude_code_session_cost(tmp_path: Path) -> None:
    """T5.13's own DEVPLAN line: "includes the Claude Code session cost (T4.9)" -- T4.9's own
    ingest tags these rows `agent="claude_code_session"`; cost_panel must fold them in exactly
    like any other agent's spend, via the shared `src.cost.rollup`."""
    store = JsonlStore(tmp_path)
    _record(
        store,
        agent="claude_code_session",
        run_id="session-1",
        loop=None,
        provider="claude_cli",
        cost_usd=7.5,
    )

    panel = cost_panel(store, now=_NOW)

    assert panel["agents"][0]["agent"] == "claude_code_session"
    assert panel["agents"][0]["today_usd"] == 7.5
    assert panel["today_total_usd"] == 7.5


def test_cost_panel_agents_sorted_by_total_spend_descending(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _record(store, agent="cheap_agent", cost_usd=1.0)
    _record(store, agent="expensive_agent", cost_usd=10.0)

    panel = cost_panel(store, now=_NOW)

    assert [a["agent"] for a in panel["agents"]] == ["expensive_agent", "cheap_agent"]


def test_cost_panel_budget_exceeded_flag(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _record(store, cost_usd=25.0, recorded_at="2026-01-08T00:00:00Z")

    under = cost_panel(store, now=_NOW, budget_usd=30.0)
    over = cost_panel(store, now=_NOW, budget_usd=20.0)

    assert under["budget_exceeded"] is False
    assert over["budget_exceeded"] is True
    assert over["budget_usd"] == 20.0


def test_cost_panel_budget_exceeded_checks_today_not_the_running_total(tmp_path: Path) -> None:
    """Regression: budget_exceeded must compare against *today's* spend, not the ever-growing
    lifetime total -- otherwise the flag trips permanently the first day any budget is set."""
    store = JsonlStore(tmp_path)
    _record(store, cost_usd=100.0, recorded_at="2025-01-01T00:00:00Z")  # old, large
    _record(store, cost_usd=1.0, recorded_at="2026-01-08T00:00:00Z")  # today, small

    panel = cost_panel(store, now=_NOW, budget_usd=20.0)

    assert panel["today_total_usd"] == 1.0
    assert panel["budget_exceeded"] is False


def test_cost_panel_defaults_budget_to_config(tmp_path: Path) -> None:
    from src import config

    store = JsonlStore(tmp_path)

    panel = cost_panel(store, now=_NOW)

    assert panel["budget_usd"] == config.DAILY_COST_BUDGET_USD


def test_cost_panel_burn_rate_is_trailing_week_average(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    _record(store, cost_usd=7.0, recorded_at="2026-01-08T00:00:00Z")  # today, in week
    _record(store, cost_usd=100.0, recorded_at="2025-01-01T00:00:00Z")  # old, excluded

    panel = cost_panel(store, now=_NOW)

    assert panel["burn_rate_usd_per_day"] == pytest.approx(1.0)  # 7.0 / 7 days


def test_cost_panel_record_with_unparseable_date_still_counts_toward_total(
    tmp_path: Path,
) -> None:
    """`src.cost.rollup` buckets a missing/corrupt `recorded_at` under the literal date
    "unknown" rather than dropping it -- cost_panel must still count that spend in `total_usd`
    (it happened, it's billed), just never in `today_usd`/`week_usd` (there's no real date to
    compare against the window)."""
    store = JsonlStore(tmp_path)
    store.record_run(
        {
            "stage": "cost",
            "agent": "analyst",
            "run_id": "r1",
            "loop": "intel",
            "provider": "claude_api",
            "model": "claude-sonnet-5",
            "tokens_in": 1,
            "tokens_out": 1,
            "tokens_cache": 0,
            "cost_usd": 2.0,
            "recorded_at": "",
        }
    )

    panel = cost_panel(store, now=_NOW)

    assert panel["agents"][0]["total_usd"] == 2.0
    assert panel["agents"][0]["today_usd"] == 0.0
    assert panel["today_total_usd"] == 0.0
