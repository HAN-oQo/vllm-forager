"""Cost panel (T5.13): per-agent spend (today / 7d / total), broken down by model & provider,
a burn-rate, and a budget line — lives in the Agents/Ops tab next to T5.8's live-health panel,
not the report tabs (this todo's own "cost belongs with health, not with the reports" framing).

Data-shaping only, composing over :func:`~src.cost.rollup`/:func:`~src.cost.rollup_records`
(T4.8/T4.9's own per-day/agent/model aggregation of every ``stage="cost"`` record) rather than
a second aggregation over ``store.list_runs(stage="cost")`` — matches T5.7's identical "data
layer first, drift-line rendering is a further step" scope boundary for the same reason.
`rollup`/`rollup_records` already fold in the Claude Code session cost (T4.9's own
`agent="claude_code_session"` rows, ingested by :mod:`src.cost_ccusage`) alongside every
`llm.complete` call T4.8 recorded — this module doesn't need to special-case that source.

**Inherited, disclosed limitation (not fixed here):** :func:`~src.cost.rollup_records`'s own
docstring already discloses that a bucket's `provider` is "whichever record is encountered
first" for a given (date, agent, model) — if the same model were ever served through two
different providers for the same agent on the same day, this module's own `by_model` breakdown
would silently attribute the whole day's cost to just one of them. A real fix means changing
`rollup`/`rollup_records`'s own bucket key (or a `Store`-level change), out of this
dashboard-rendering todo's scope.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from src import config
from src.cost import rollup_records
from src.store.base import Store

from .api import normalize_now

_WEEK_DAYS = 7


def _parse_date(date_str: str) -> date | None:
    """`date_str` (a :func:`~src.cost.rollup_records` bucket's own ``"YYYY-MM-DD"`` or the
    literal ``"unknown"``) as a real `date`, or `None` if it isn't one — a corrupt/missing
    ``recorded_at`` still contributes to `total_usd` (via `rollup_records`'s own "unknown"
    bucket convention) but can never match the today/7d window, since there's no real date to
    compare."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def cost_panel(
    store: Store,
    *,
    now: datetime | None = None,
    budget_usd: float | None = None,
    runs: list[dict] | None = None,
) -> dict:
    """``{"agents": [...], "budget_usd", "today_total_usd", "budget_exceeded",
    "burn_rate_usd_per_day"}``.

    Each `agents` entry: ``{"agent", "today_usd", "week_usd", "total_usd", "by_model": [...]}``,
    sorted by `today_usd` descending — the DEVPLAN's own "e.g." worked example is specifically
    "a bar per agent (**$ today**)", so the ranked view it implies ranks by *today's* spend,
    not the ever-growing lifetime total (a code-review finding: an earlier version sorted by
    `total_usd`, which can rank a since-retired agent with a large historical total ahead of
    today's actual top spender). Each `by_model` entry: ``{"model", "provider", "cost_usd",
    "calls"}``, sorted by `cost_usd` descending.

    `budget_usd` defaults to :data:`~src.config.DAILY_COST_BUDGET_USD` — an unreviewed
    placeholder default (see that constant's own comment), not a researched number; a caller
    presenting `budget_exceeded` to a human should make that provenance visible rather than
    imply a real, reviewed budget was crossed (a code-review finding on an earlier version's
    rendering, which gave no such indication).

    `budget_exceeded` is `today_total_usd > budget_usd` — the DEVPLAN's own "turns red when a
    budget line is crossed" (a *day's* spend against a *daily* budget, not the running
    `total_usd`, which only ever grows and would trip the flag permanently the first day it's
    set).

    `burn_rate_usd_per_day` is the trailing-7-day average (`week` total / 7) — steadier than
    `today_total_usd` alone for a badge checked at an arbitrary time of day (partway through
    "today" always understates a full day's eventual spend). **Disclosed limitation:** the
    divisor is always 7 regardless of how much cost history actually exists — right after cost
    tracking starts (or after a data gap), this under-reports the true run rate rather than
    over-reporting it, since a short history is divided by the same fixed window a full one
    would be.

    `runs` — a code-review finding — lets a caller that's already fetched
    ``store.list_runs()`` for another purpose (:func:`~dashboard.render.render_ops_tab` also
    builds T5.8's health panel from the identical unfiltered fetch) pass that same list in via
    :func:`~src.cost.rollup_records`, rather than this function's own `rollup(store)` call
    triggering a second, redundant full scan of the same underlying collection. `None` (the
    default) fetches fresh.
    """
    when = normalize_now(now)
    today = when.date()
    week_start = today - timedelta(days=_WEEK_DAYS - 1)
    budget = config.DAILY_COST_BUDGET_USD if budget_usd is None else budget_usd

    fetched_runs = runs if runs is not None else store.list_runs(stage="cost")
    buckets = rollup_records(fetched_runs)

    models_by_agent: dict[str, dict[tuple[str, str], dict]] = {}
    totals_by_agent: dict[str, dict[str, float]] = {}
    for bucket in buckets:
        agent = bucket["agent"]
        totals = totals_by_agent.setdefault(
            agent, {"today_usd": 0.0, "week_usd": 0.0, "total_usd": 0.0}
        )
        cost_usd = bucket["cost_usd"]
        totals["total_usd"] += cost_usd
        bucket_date = _parse_date(bucket["date"])
        if bucket_date is not None and week_start <= bucket_date <= today:
            totals["week_usd"] += cost_usd
            if bucket_date == today:
                totals["today_usd"] += cost_usd
        models = models_by_agent.setdefault(agent, {})
        model_key = (bucket["model"], bucket["provider"])
        model_entry = models.setdefault(
            model_key,
            {"model": bucket["model"], "provider": bucket["provider"], "cost_usd": 0.0, "calls": 0},
        )
        model_entry["cost_usd"] += cost_usd
        model_entry["calls"] += bucket["calls"]

    agents: list[dict] = [
        {
            **totals,
            "agent": agent,
            "by_model": sorted(
                models_by_agent[agent].values(), key=lambda m: m["cost_usd"], reverse=True
            ),
        }
        for agent, totals in totals_by_agent.items()
    ]
    agents.sort(key=lambda a: a["today_usd"], reverse=True)

    today_total_usd = sum(a["today_usd"] for a in agents)
    week_total_usd = sum(a["week_usd"] for a in agents)

    return {
        "agents": agents,
        "budget_usd": budget,
        "today_total_usd": today_total_usd,
        "budget_exceeded": today_total_usd > budget,
        "burn_rate_usd_per_day": week_total_usd / _WEEK_DAYS,
    }
