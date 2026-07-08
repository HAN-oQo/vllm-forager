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
  :func:`~dashboard.snapshot.capped_tree` (no reason to duplicate either).
- **trends** — :func:`trends`, over :func:`~src.trends.trends_from_store`. Cheap (one scan).
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
  todo is the natural place that lands), not this one. **Known limitation, not fixed here**:
  ``[]`` means both "never computed" and "computed, genuinely empty" — indistinguishable to a
  caller. T5.10 (the human-in-the-loop selection UI) will very likely need to replace this
  plain `compute: bool` with a persisted-queue read plus a separate trigger-computation entry
  point (a stable per-candidate identity to write a `decision` record against, and a
  non-request-blocking way to trigger a scoring pass) — this signature is not guaranteed to
  survive that redesign unchanged, and callers shouldn't assume it will.
- **parity** — :func:`parity`, over :func:`~src.parity.build_matrix`/
  :func:`~src.parity.safe_find_gaps_for_all_targets`. Cheap (no LLM calls, just a scan +
  grouping) — unlike candidates, safe to compute on every read. **Known limitation, not fixed
  here**: :func:`candidates`\\ (``compute=True``) independently recomputes this exact same
  matrix/gap pair inside :func:`~src.agents.scout.discover_from_store` — a caller wanting both
  on one page pays for it twice; not fixed here since nothing in this codebase calls both
  together yet (no real caller to design the sharing against).
- **runs** — :func:`runs`/:func:`stage_status`, thin wrappers over
  :meth:`~src.store.base.Store.list_runs`/:func:`~src.liveness.latest_status` (already built
  for exactly this "what's running" purpose, T4.5).

**Known limitation, not fixed here — deferred for a second review cycle in a row**: every
function that scans the store (items/tree/trends/parity) still does its own independent full
:meth:`~src.store.base.Store.query` — the same "no `Store`-level `limit`/`offset`/path filter
yet" gap :mod:`dashboard.snapshot`'s own docstring already discloses, itself written pointing
at "T5.1's read layer" as "the natural place" to add it. T5.1 (this todo) still didn't add it —
no real multi-domain caller exists yet to design the filter against, so speculatively building
one now risked guessing wrong; but this gap has now survived two dedicated review cycles
(T5.16(c), T5.1) without a concrete next owner. Whoever hits this for real (most likely T5.2's
"monitoring panels," the first todo combining several of these domains on one page) should
either fix `Store.query()` itself or open a dedicated todo naming it explicitly, rather than
deferring a third time with the same paragraph.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from src import parity as parity_module
from src.agents import scout
from src.agents.forecaster import list_predictions
from src.liveness import latest_status
from src.store.base import Store
from src.taxonomy import CategoryPath
from src.trends import trends_from_store

from .snapshot import DEFAULT_MAX_PRS_PER_NODE, capped_tree, items_at_path, paginate


def items(
    store: Store, path: CategoryPath = (), *, offset: int = 0, limit: int | None = None
) -> tuple[list[dict], int]:
    """One page of items classified exactly at `path` (``()`` — the default — selects the
    unclassified/``Other`` bucket), plus the true total. Thin re-export of
    :func:`~dashboard.snapshot.items_at_path`."""
    return items_at_path(store, path, offset=offset, limit=limit)


def tree(store: Store, *, max_prs_per_node: int = DEFAULT_MAX_PRS_PER_NODE) -> list[dict]:
    """The full classified tree (counts + capped top-N leaf rows per node). Thin re-export of
    :func:`~dashboard.snapshot.capped_tree` — *not* ``build_snapshot(...)["tree"]``: an earlier
    version of this function went through :func:`~dashboard.snapshot.build_snapshot`, which
    also unconditionally computes trends (a second full store scan) and forecasts (a full
    prediction-log read) only to discard both — wasted work, and a tree-only read could crash
    on a corrupt *prediction* record that has nothing to do with the tree (a code-review
    finding on this PR)."""
    return capped_tree(store, max_prs_per_node=max_prs_per_node)


def trends(store: Store, category: str | None = None) -> dict:
    """Every category's per-week activity counts (``{category: {week: count}}``), or — when
    `category` is given — just that one category's own series (``{week: count}``, or ``{}`` if
    it has no recorded activity). One function, not two (an earlier version split this into
    `trends`/`trends_for_category` — a code-review finding that the split didn't earn its
    keep), matching this todo's own worked example (``api.trends("quantization")``) more
    directly. Thin re-export of :func:`~src.trends.trends_from_store`."""
    series = trends_from_store(store)
    return series if category is None else series.get(category, {})


def predictions(store: Store, *, offset: int = 0, limit: int | None = None) -> list[dict]:
    """One page of recorded predictions (oldest first, matching
    :func:`~src.agents.forecaster.list_predictions`'s own order), as plain dicts. `limit=None`
    (the default) returns every remaining prediction from `offset` on.

    The `(offset, limit)` slice is taken *before* the ``dataclasses.asdict`` conversion, not
    after — an earlier version of this function converted every recorded prediction first and
    only then sliced, paying to transform rows it was about to discard; the sibling
    :func:`~dashboard.snapshot.items_at_path` was fixed to avoid exactly this shape one todo
    earlier (T5.16's own code review), and this function repeated it until this fix.
    """
    page = paginate(list_predictions(store), offset=offset, limit=limit)
    return [asdict(p) for p in page]


def asdict_with(obj: Any, **computed: object) -> dict:
    """`obj` (a dataclass instance) as a plain dict via ``dataclasses.asdict``, with `computed`
    merged in afterward — the shared fix for a computed ``@property`` (which ``asdict`` alone
    silently drops, since it isn't a real dataclass field) or a field ``asdict`` gets wrong
    (e.g. a ``Path`` that needs to become a ``str`` for JSON). Extracted here after this exact
    one-line pattern was independently re-derived three times across the dashboard package
    (this module's own :func:`_candidate_dict`, :mod:`dashboard.review`'s
    ``_gate_result_dict``, :mod:`dashboard.guardrails`'s ``rag_eval_series``) — a code-review
    finding that the third occurrence crossed the point where a shared helper earns its keep.
    """
    return {**asdict(obj), **computed}


def _candidate_dict(candidate: scout.Candidate) -> dict:
    """`candidate` as a plain dict, `priority` included — it's a computed ``@property``, not a
    dataclass field, so ``dataclasses.asdict`` alone would silently drop the one field a
    ranked-queue panel needs to sort/display by."""
    return asdict_with(candidate, priority=candidate.priority)


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
    (:func:`~src.parity.safe_find_gaps_for_all_targets`), both as plain dicts. Cheap: no LLM
    calls, just a store scan plus grouping — safe to compute on every read, unlike
    :func:`candidates`.

    A :class:`~src.parity.ParityError` (no ``"primary"``-role engine configured) degrades to an
    empty ``gaps`` list rather than raising, via the same shared helper
    :func:`~src.agents.scout.discover_from_store` uses — so a read endpoint doesn't 500 over a
    config gap the matrix itself (which needs no target engine) doesn't have, and the two real
    call sites can't independently drift on how that degradation is handled.
    """
    cells = parity_module.build_matrix(store.query())
    gaps = parity_module.safe_find_gaps_for_all_targets(cells, caller="dashboard.api")
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
