"""Precomputed dashboard snapshot (T5.16): a bounded, JSON-serializable summary of the KB —
tree (counts + capped top-N leaf rows per node) + trends + forecasts — built with exactly one
full store scan, then rendered/paginated from that snapshot alone with no further store access.

Why: :mod:`dashboard.render`'s original M1 thin slice re-scanned the whole store and emitted
every single item's row into one HTML string on every request — already 14 MB / ~54.8k rows
(DEVPLAN T5.16's own "Why"), growing without bound. This module is the "aggregate-first"
foundation every other M5 panel (T5.1's read layer, T5.2–T5.14's panels) is meant to build on:
a page's byte size stays roughly constant as the KB grows, because only each node's own
``count`` scales — the *rendered* leaf rows are always capped at
:data:`DEFAULT_MAX_PRS_PER_NODE`; the rest is reachable via :func:`items_at_path`'s on-demand
pagination (:mod:`dashboard.server`'s ``/api/node-prs`` endpoint) instead of ever landing in
the page's own HTML.

Not done here (a further M5 todo, not this one): a periodic/scheduled job that writes this
snapshot to a ``dashboard.json`` file the server reads instead of rebuilding on every request —
:func:`build_snapshot`/:func:`dump_snapshot`/:func:`load_snapshot` below make that possible
(this module is what such a job would call), but :mod:`dashboard.server` still calls
:func:`build_snapshot` fresh per request today, matching the existing "never caches, so a
long-running process reflects new items without a restart" design that module's own docstring
already commits to.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from src.agents.forecaster import list_predictions
from src.agents.reporter import OTHER
from src.agents.reporter_v1 import TreeNode, pr_entry, tree_from_store
from src.store.base import Store
from src.taxonomy import CategoryPath
from src.trends import trends_from_store

# Rendered inline in the initial page load; the rest of a node's own items are reachable via
# the paginated /api/node-prs endpoint (dashboard/server.py) instead of ever landing in the
# page's own HTML -- the one change that makes page size stop scaling with KB size.
DEFAULT_MAX_PRS_PER_NODE = 50


def _capped_node(node: TreeNode, *, path: CategoryPath, max_prs: int) -> dict:
    """`node` (see :class:`~src.agents.reporter_v1.TreeNode`) as a JSON-serializable dict, its
    own ``prs`` capped to `max_prs` — ``prs_total``/``prs_truncated`` tell the renderer/UI
    there's more to page through via :func:`items_at_path`, without embedding it."""
    prs = list(node.prs)
    return {
        "name": node.name,
        "path": list(path),
        "summary": node.summary,
        "count": node.count,
        "gaps": node.gaps,
        "children": [
            _capped_node(child, path=(*path, child.name), max_prs=max_prs)
            for child in node.children
        ],
        "prs": prs[:max_prs],
        "prs_total": len(prs),
        "prs_truncated": len(prs) > max_prs,
    }


def build_snapshot(store: Store, *, max_prs_per_node: int = DEFAULT_MAX_PRS_PER_NODE) -> dict:
    """One full store scan (:func:`~src.agents.reporter_v1.tree_from_store`'s own
    ``store.query()`` plus one per-node summary read), producing a plain, JSON-serializable
    dict — everything :func:`~dashboard.render.render_snapshot_page` needs to render a page,
    and everything :func:`dump_snapshot`/:func:`load_snapshot` need to round-trip through a
    file, without ever touching `store` again afterward.
    """
    nodes = tree_from_store(store)
    tree = [_capped_node(node, path=(node.name,), max_prs=max_prs_per_node) for node in nodes]
    forecasts = [asdict(p) for p in list_predictions(store)]
    return {"tree": tree, "trends": trends_from_store(store), "forecasts": forecasts}


def dump_snapshot(snapshot: dict) -> str:
    """`snapshot` as a JSON string — e.g. for a future scheduled job to write to
    ``dashboard.json`` (see this module's own docstring)."""
    return json.dumps(snapshot)


def load_snapshot(raw: str) -> dict:
    """The inverse of :func:`dump_snapshot`."""
    return json.loads(raw)


def items_at_path(store: Store, path: CategoryPath) -> list[dict]:
    """Every item classified exactly at `path` (not a descendant) — the same population one
    tree node's own ``prs`` covers (see :class:`~src.agents.reporter_v1.TreeNode`'s docstring),
    computed directly rather than building the whole tree — what :mod:`dashboard.server`'s
    ``/api/node-prs`` pagination endpoint calls for an on-demand page past a node's capped
    inline rows. Each item is shaped via
    :func:`~src.agents.reporter_v1.pr_entry`, matching a `TreeNode`'s own ``prs`` shape exactly.

    `path == (OTHER,)` (or empty) matches the same "no path, or classified OTHER" population
    :func:`~src.agents.reporter_v1.build_tree` folds into its own flat ``Other`` root node.
    """
    if not path or path == (OTHER,):
        items = [
            item
            for item in store.query()
            if not isinstance(item.get("path"), list)
            or not item.get("path")
            or item.get("path") == [OTHER]
        ]
    else:
        path_list = list(path)
        items = [item for item in store.query() if item.get("path") == path_list]
    return [pr_entry(item) for item in items]
