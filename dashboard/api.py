"""Dashboard read layer (T5.1): the one place every panel (T5.2–T5.14) reads the KB through —
plain, JSON-serializable shapes in, no HTML, no write path. Keeps the UI decoupled from the KB
backend (JSONL/Firestore, per :class:`~src.store.base.Store`'s own abstraction) and keeps the
dashboard strictly read-only, matching this todo's own DEVPLAN "Why".

Every function here takes `store` explicitly (no hidden global/singleton), matching the
convention :mod:`dashboard.render`/:mod:`dashboard.snapshot` already use. This module is
purely **additive**: :mod:`dashboard.render`/:mod:`dashboard.snapshot` (T5.16) already have
their own working, tested calls into ``src.agents.reporter_v1``/``src.trends``/
``src.agents.forecaster`` for the initial page render — rewiring them through this module would
be pure churn for no behavior change, so they're left as-is; this module is the surface a
*new* T5.2+ panel reaches for going forward.

Six data domains, per this todo's own text — what's cheap (a `Store.query()` scan, no LLM
call) vs. genuinely expensive is the load-bearing design decision here:

- **items** — :func:`items`/:func:`tree`, thin wrappers over :mod:`dashboard.snapshot`'s
  already-built, already-tested :func:`~dashboard.snapshot.items_at_path`/
  :func:`~dashboard.snapshot.build_snapshot` (no reason to duplicate either).
- **trends** — :func:`trends`/:func:`trends_for_category`, over
  :func:`~src.trends.trends_from_store`. Cheap (one scan).
- **predictions** — :func:`predictions`, over
  :func:`~src.agents.forecaster.list_predictions`. Cheap (reads the KB's small state map).
- **candidates** — :func:`candidates` — **not cheap**: every candidate costs one
  ``llm.complete`` call (:func:`~src.agents.scout.discover_from_store`'s own documented cost;
  nothing persists a scored queue anywhere in this codebase today — the orchestrator's
  ``_contribution`` stage only prints). A dashboard read endpoint that silently re-ran LLM
  scoring on every page view would be exactly the "tokens burn fast" hazard T4.8–T4.11 exist
  to measure/cap/cut — so :func:`candidates` defaults to **not** computing anything (returns
  ``[]``) unless a caller explicitly opts in with ``compute=True``, mirroring this codebase's
  own established "expensive work is opt-in, safe by default" convention (e.g.
  ``analyze_store``'s ``batch_size``, T4.10's own gating). Persisting a scored queue so a
  *cheap* read becomes possible is a real follow-up (T5.10's own "human-in-the-loop signal"
  todo is the natural place that lands), not this one.
- **parity** — :func:`parity`, over :func:`~src.parity.build_matrix`/
  :func:`~src.parity.find_gaps_for_all_targets`. Cheap (no LLM calls, just a scan + grouping) —
  unlike candidates, safe to compute on every read.
- **runs** — :func:`runs`/:func:`stage_status`, thin wrappers over
  :meth:`~src.store.base.Store.list_runs`/:func:`~src.liveness.latest_status` (already built
  for exactly this "what's running" purpose, T4.5).

Known limitation, not fixed here: every function that scans the store (items/tree/trends/
parity) still does its own independent full :meth:`~src.store.base.Store.query` — the same
"no `Store`-level `limit`/`offset`/path filter yet" gap :mod:`dashboard.snapshot`'s own
docstring already discloses. This module doesn't add a new one, but doesn't fix the underlying
one either; a real fix is a `Store` interface change, out of scope for a read-*layer* todo.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime

from src import parity as parity_module
from src.agents import scout
from src.agents.forecaster import list_predictions
from src.liveness import latest_status
from src.store.base import Store
from src.taxonomy import CategoryPath
from src.trends import trends_from_store

from .snapshot import DEFAULT_MAX_PRS_PER_NODE, build_snapshot, items_at_path


def items(
    store: Store, path: CategoryPath = (), *, offset: int = 0, limit: int | None = None
) -> tuple[list[dict], int]:
    """One page of items classified exactly at `path` (``()`` — the default — selects the
    unclassified/``Other`` bucket), plus the true total. Thin re-export of
    :func:`~dashboard.snapshot.items_at_path`."""
    return items_at_path(store, path, offset=offset, limit=limit)


def tree(store: Store, *, max_prs_per_node: int = DEFAULT_MAX_PRS_PER_NODE) -> list[dict]:
    """The full classified tree (counts + capped top-N leaf rows per node) — the ``"tree"``
    half of :func:`~dashboard.snapshot.build_snapshot`, for a panel that wants the tree without
    also paying for trends/forecasts it doesn't need."""
    return build_snapshot(store, max_prs_per_node=max_prs_per_node)["tree"]


def trends(store: Store) -> dict[str, dict[str, int]]:
    """Every category's per-week activity counts (``{category: {week: count}}``). Thin
    re-export of :func:`~src.trends.trends_from_store`."""
    return trends_from_store(store)


def trends_for_category(store: Store, category: str) -> dict[str, int]:
    """One category's own per-week series (``{week: count}``), or ``{}`` if it has no recorded
    activity — this todo's own worked example (``api.trends("quantization")``)."""
    return trends_from_store(store).get(category, {})


def predictions(store: Store, *, offset: int = 0, limit: int | None = None) -> list[dict]:
    """Every recorded prediction (oldest first, matching
    :func:`~src.agents.forecaster.list_predictions`'s own order), as plain dicts, sliced to one
    page. `limit=None` (the default) returns every remaining prediction from `offset` on."""
    all_predictions = [asdict(p) for p in list_predictions(store)]
    if limit is None:
        return all_predictions[offset:]
    return all_predictions[offset : offset + limit]


def _candidate_dict(candidate: scout.Candidate) -> dict:
    """`candidate` as a plain dict, `priority` included — it's a computed ``@property``, not a
    dataclass field, so ``dataclasses.asdict`` alone would silently drop the one field a
    ranked-queue panel needs to sort/display by."""
    return {**asdict(candidate), "priority": candidate.priority}


def candidates(
    store: Store, *, compute: bool = False, per_domain_limit: int | None = None
) -> list[dict]:
    """The risk-ranked contribution queue, highest :attr:`~src.agents.scout.Candidate.priority`
    first — or ``[]`` unless `compute=True` is passed explicitly.

    See this module's own docstring: nothing persists a scored candidate queue anywhere today,
    so producing a non-empty result here means calling
    :func:`~src.agents.scout.discover_from_store` — one real ``llm.complete`` call per
    candidate — live, on this call. Defaulting to `compute=False` means a dashboard page view
    can never *silently* trigger that spend; a caller that wants the real queue (e.g. T5.10's
    own selection UI) must opt in explicitly and accept the cost.
    """
    if not compute:
        return []
    found = scout.discover_from_store(store, per_domain_limit=per_domain_limit)
    return [_candidate_dict(c) for c in found]


def parity(store: Store) -> dict:
    """``{"cells": [...], "gaps": [...]}`` — the full engine × capability matrix
    (:func:`~src.parity.build_matrix`) plus every cross-engine gap
    (:func:`~src.parity.find_gaps_for_all_targets`), both as plain dicts. Cheap: no LLM calls,
    just a store scan plus grouping — safe to compute on every read, unlike :func:`candidates`.

    A :class:`~src.parity.ParityError` (no ``"primary"``-role engine configured) degrades to an
    empty ``gaps`` list rather than raising — the same graceful-degradation
    :func:`~src.agents.scout.discover_from_store` already applies, so a read endpoint doesn't
    500 over a config gap the matrix itself (which needs no target engine) doesn't have.
    """
    cells = parity_module.build_matrix(store.query())
    try:
        gaps = parity_module.find_gaps_for_all_targets(cells)
    except parity_module.ParityError as exc:
        print(f"dashboard.api: skipping parity gaps: {exc}", file=sys.stderr)
        gaps = []
    return {"cells": [asdict(c) for c in cells], "gaps": [asdict(g) for g in gaps]}


def runs(
    store: Store, *, stage: str | None = None, repo: str | None = None, number: int | None = None
) -> list[dict]:
    """Every recorded run matching the given filters (AND; omitted filters don't constrain) —
    thin re-export of :meth:`~src.store.base.Store.list_runs`."""
    return store.list_runs(stage=stage, repo=repo, number=number)


def stage_status(store: Store, stage: str, *, now: datetime, stale_after_s: float) -> str:
    """`stage`'s current liveness — ``"running"``/``"stalled"``/``"ok"``/``"failed"``/
    ``"never run"`` — the T5.8 health panel's own worked example, one stage at a time. Thin
    re-export of :func:`~src.liveness.latest_status` over that stage's own run events."""
    return latest_status(store.list_runs(stage=stage), now=now, stale_after_s=stale_after_s)
