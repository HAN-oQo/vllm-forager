"""Trend series (T1.7): per-category activity time series from the KB.

Turns classified items (T1.4's ``category`` field) into "how much activity per category, per
week" — the raw material for M5's trend charts (docs/PLAN.md: "technique/topic momentum over
time per taxonomy category ... sourced from classified issue/PR volume").

An item counts once, in the ISO week it was **created** (``created_at``) — not every week it
was touched — so activity reflects new issues/PRs appearing, not existing ones being edited.
Items with no ``category`` yet (T1.4 hasn't classified them) are excluded — a
technique/topic-momentum chart has nothing meaningful to say about "not yet classified".
Items explicitly categorized ``Other`` (the Analyst's own hallucination/no-match fallback,
:mod:`src.agents.reporter`'s ``OTHER``) ARE counted, as their own category: how much
"doesn't-fit-anywhere" activity there is over time is itself a signal T2.3's Curator can act
on (propose a new category once Other's volume trends up).

``week_stamp`` (the ISO-year/ISO-week formatter) lives here, not in :mod:`src.report`: it's a
pure, dependency-free ``datetime -> str`` function, and this module is the lowest-altitude
place both :mod:`src.report` and a future M5 dashboard (docs/PLAN.md names it a third
consumer) can depend on without pulling in the CLI/reporter/LLM stack just to format a date.
``src.report`` imports it back from here.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone

from .store.base import Store

# The same GitHub-API-style timestamp format every other stage in this codebase already uses
# (collector.py's cursor, forecaster.py's TS_FORMAT) — created_at is machine-written by the
# collector from GitHub's own API, never LLM-generated, so (unlike forecaster.py's due_date)
# there's no real-world formatting drift here to tolerate; strict parsing is the right choice.
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def week_stamp(when: datetime | None = None) -> str:
    """The ISO-year/ISO-week stamp used for report filenames and trend buckets, e.g. ``2026-W27``.

    ``%G``/``%V`` are the ISO-8601 year and week (not ``%Y``/``%U``): near a year boundary the
    ISO week's year can differ from the calendar year, and this keeps week numbers contiguous
    (…W52, W53?, W01…). Defaults to now (UTC) when `when` is omitted.
    """
    when = when or datetime.now(timezone.utc)
    return when.strftime("%G-W%V")


def _created_week(item: dict) -> str | None:
    """The ISO-year/ISO-week `item` was created, or None if `created_at` is missing/malformed.

    A malformed timestamp is skipped (and logged to stderr), not raised — one bad record
    shouldn't crash a trend computation meant to summarize thousands of items, matching the
    "skip and log" tolerance collector.py's own T0.10 robustness guardrail and
    analyst.py/forecaster.py's per-item error handling already established.
    """
    raw = item.get("created_at")
    if not raw:
        return None
    try:
        created = datetime.strptime(raw, _TS_FORMAT).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError) as exc:
        repo, number = item.get("repo"), item.get("number")
        print(f"trends: skipping {repo}#{number}: bad created_at {raw!r}: {exc}", file=sys.stderr)
        return None
    return week_stamp(created)


def category_trends(items: list[dict]) -> dict[str, dict[str, int]]:
    """Bucket `items` into ``{category: {week: count}}`` by each item's creation week.

    Only classified items (a truthy, string ``category``) are counted — a non-string
    ``category`` (e.g. a list, from an upstream classifier bug) is treated as malformed and
    skipped, the same as an unparseable ``created_at``, rather than crashing on an unhashable
    dict key. Items with a missing or malformed ``created_at`` are skipped too.
    """
    trends: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        category = item.get("category")
        if not category or not isinstance(category, str):
            continue
        week = _created_week(item)
        if week is None:
            continue
        trends[category][week] += 1
    return {category: dict(weeks) for category, weeks in trends.items()}


def trends_from_store(store: Store) -> dict[str, dict[str, int]]:
    """Read all items from `store` and compute their per-category trend series."""
    return category_trends(store.query())
