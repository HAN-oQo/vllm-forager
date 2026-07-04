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
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from .report import week_stamp
from .store.base import Store

# The same GitHub-API-style timestamp format every other stage in this codebase already uses
# (collector.py's cursor, forecaster.py's _TS_FORMAT).
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _created_week(item: dict) -> str | None:
    """The ISO-year/ISO-week `item` was created, or None if `created_at` is missing/malformed.

    A malformed timestamp is skipped, not raised — one bad record shouldn't crash a trend
    computation meant to summarize thousands of items (the same tolerance the collector's own
    robustness guardrail, T0.10, applies to malformed records at collection time).
    """
    raw = item.get("created_at")
    if not raw:
        return None
    try:
        created = datetime.strptime(raw, _TS_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return week_stamp(created)


def category_trends(items: list[dict]) -> dict[str, dict[str, int]]:
    """Bucket `items` into ``{category: {week: count}}`` by each item's creation week.

    Only classified items (a truthy ``category``) are counted; items with a missing or
    malformed ``created_at`` are skipped.
    """
    trends: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        category = item.get("category")
        if not category:
            continue
        week = _created_week(item)
        if week is None:
            continue
        trends[category][week] += 1
    return {category: dict(weeks) for category, weeks in trends.items()}


def trends_from_store(store: Store) -> dict[str, dict[str, int]]:
    """Read all items from `store` and compute their per-category trend series."""
    return category_trends(store.query())
