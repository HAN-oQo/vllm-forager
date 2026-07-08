"""Trend visualizations (T5.3): category momentum over time, windowed and gap-filled into
chart-ready points.

:func:`~dashboard.api.trends` (T5.1) returns each category's *sparse* ``{week: count}`` series
-- only weeks with recorded activity are present. A line chart needs every week in its window
to appear (a quiet week is a real, meaningful zero, not an absent point): :func:`category_series`
is the one place that windowing/gap-filling happens, so a future chart-rendering step (T5.9's
Reports/Trends archive tab, or T5.12's console shell) can read dense points straight off this
function instead of reimplementing the gap-fill per caller. Matches this todo's own worked
example -- "a line chart of 'spec decoding' vs 'disaggregated prefill' activity across the last
12 weeks" -- directly: ``category_series(store, ["spec decoding", "disaggregated prefill"])``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from src.store.base import Store
from src.trends import week_stamp

from . import api


def _last_n_weeks(weeks: int, *, now: datetime | None = None) -> list[str]:
    """The `weeks` ISO week-stamps ending at (and including) `now`'s own week, oldest first."""
    when = now or datetime.now(timezone.utc)
    return [week_stamp(when - timedelta(weeks=i)) for i in range(weeks - 1, -1, -1)]


def category_series(
    store: Store,
    categories: Sequence[str] | None = None,
    *,
    weeks: int = 12,
    now: datetime | None = None,
) -> dict[str, list[dict]]:
    """Each requested category's own weekly activity count over the last `weeks` ISO weeks, as
    a dense, gap-filled ``[{"week": ..., "count": ...}, ...]`` list -- every week in the window
    appears, with ``count=0`` for a quiet week, so a chart never has to reindex or gap-fill a
    sparse series itself.

    `categories=None` (the default) covers every category with any recorded activity, ever
    (:func:`~dashboard.api.trends`'s own keys) -- not just activity inside the window, so a
    category that went quiet exactly `weeks` weeks ago still shows up (correctly, as all
    zeros) rather than disappearing. Passing an explicit list only computes those categories,
    even ones with no activity at all yet (still returned, all zeros) -- this is what lets a
    caller compare two named categories side by side regardless of which one currently has
    more activity.
    """
    series = api.trends(store)
    selected = categories if categories is not None else sorted(series)
    window = _last_n_weeks(weeks, now=now)
    return {
        category: [{"week": w, "count": series.get(category, {}).get(w, 0)} for w in window]
        for category in selected
    }
