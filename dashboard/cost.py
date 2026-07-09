"""Cost panel (T5.13): per-agent spend (today / 7d / total), broken down by model & provider,
a burn-rate, and a budget line — lives in the Agents/Ops tab next to T5.8's live-health panel,
not the report tabs (this todo's own "cost belongs with health, not with the reports" framing).

Data-shaping only, composing over :func:`~src.cost.rollup` (T4.8/T4.9's own per-day/agent/model
aggregation of every ``stage="cost"`` record) rather than a second aggregation over
``store.list_runs(stage="cost")`` — matches T5.7's identical "data layer first, drift-line
rendering is a further step" scope boundary for the same reason. `rollup` already folds in the
Claude Code session cost (T4.9's own `agent="claude_code_session"` rows, ingested by
:mod:`src.cost_ccusage`) alongside every `llm.complete` call T4.8 recorded — this module doesn't
need to special-case that source.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from src import config
from src.cost import rollup
from src.store.base import Store

_WEEK_DAYS = 7


def _parse_date(date_str: str) -> date | None:
    """`date_str` (a :func:`~src.cost.rollup` bucket's own ``"YYYY-MM-DD"`` or the literal
    ``"unknown"``) as a real `date`, or `None` if it isn't one — a corrupt/missing
    ``recorded_at`` still contributes to `total_usd` (via `rollup`'s own "unknown" bucket
    convention) but can never match the today/7d window, since there's no real date to compare."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def cost_panel(
    store: Store, *, now: datetime | None = None, budget_usd: float | None = None
) -> dict:
    """``{"agents": [...], "budget_usd", "today_total_usd", "budget_exceeded",
    "burn_rate_usd_per_day"}``.

    Each `agents` entry: ``{"agent", "today_usd", "week_usd", "total_usd", "by_model": [...]}``,
    sorted by `total_usd` descending (highest spender first — the DEVPLAN's own "a bar per
    agent" worked example implies a ranked view, not alphabetical). Each `by_model` entry:
    ``{"model", "provider", "cost_usd", "calls"}``, sorted by `cost_usd` descending.

    `budget_usd` defaults to :data:`~src.config.DAILY_COST_BUDGET_USD`; `budget_exceeded` is
    `today_total_usd > budget_usd` — the DEVPLAN's own "turns red when a budget line is
    crossed" (a *day's* spend against a *daily* budget, not the running `total_usd`, which
    only ever grows and would trip the flag permanently the first day it's set).

    `burn_rate_usd_per_day` is the trailing-7-day average (`week` total / 7), a steadier signal
    than `today_total_usd` alone for a badge that's checked at an arbitrary time of day (partway
    through "today" always understates a full day's eventual spend).
    """
    when = now or datetime.now(timezone.utc)
    today = when.date()
    week_start = today - timedelta(days=_WEEK_DAYS - 1)
    budget = config.DAILY_COST_BUDGET_USD if budget_usd is None else budget_usd

    by_agent: dict[str, dict] = {}
    today_total_usd = 0.0
    week_total_usd = 0.0
    for bucket in rollup(store):
        agent = bucket["agent"]
        entry = by_agent.setdefault(
            agent,
            {"agent": agent, "today_usd": 0.0, "week_usd": 0.0, "total_usd": 0.0, "by_model": {}},
        )
        cost_usd = bucket["cost_usd"]
        entry["total_usd"] += cost_usd
        bucket_date = _parse_date(bucket["date"])
        in_week = bucket_date is not None and week_start <= bucket_date <= today
        if in_week:
            entry["week_usd"] += cost_usd
            week_total_usd += cost_usd
        if bucket_date == today:
            entry["today_usd"] += cost_usd
            today_total_usd += cost_usd
        model_key = (bucket["model"], bucket["provider"])
        model_entry = entry["by_model"].setdefault(
            model_key,
            {"model": bucket["model"], "provider": bucket["provider"], "cost_usd": 0.0, "calls": 0},
        )
        model_entry["cost_usd"] += cost_usd
        model_entry["calls"] += bucket["calls"]

    agents = []
    for entry in by_agent.values():
        entry["by_model"] = sorted(
            entry["by_model"].values(), key=lambda m: m["cost_usd"], reverse=True
        )
        agents.append(entry)
    agents.sort(key=lambda a: a["total_usd"], reverse=True)

    return {
        "agents": agents,
        "budget_usd": budget,
        "today_total_usd": today_total_usd,
        "budget_exceeded": today_total_usd > budget,
        "burn_rate_usd_per_day": week_total_usd / _WEEK_DAYS,
    }
