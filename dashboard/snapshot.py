"""Precomputed dashboard snapshot (T5.16): a bounded, JSON-serializable summary of the KB —
tree (counts + capped top-N leaf rows per node) + trends + forecasts — then rendered/paginated
from that snapshot alone with no further store access for the *render* step.

Why: :mod:`dashboard.render`'s original M1 thin slice re-scanned the whole store and emitted
every single item's row into one HTML string on every request — already 14 MB / ~54.8k rows
(DEVPLAN T5.16's own "Why"), growing without bound. This module fixes that *rendering/transfer*
half: a page's byte size stays roughly constant as the KB grows, because only each node's own
``count`` scales — the *rendered* leaf rows are always capped at
:data:`DEFAULT_MAX_PRS_PER_NODE`; the rest is reachable via :func:`items_at_path`'s on-demand
pagination (:mod:`dashboard.server`'s ``/api/node-prs`` endpoint) instead of ever landing in
the page's own HTML.

**Known limitation, not fixed here — the *read* half of the problem is still open**
(code-review finding on this todo): :func:`build_snapshot` still does **at least two**
independent full store scans per call — one inside
:func:`~src.agents.reporter_v1.tree_from_store` (plus one ``get_state`` per taxonomy node for
its summary, that function's own pre-existing disclosed cost) and a second, separate one
inside :func:`~src.trends.trends_from_store` — not "exactly one" as an earlier version of this
docstring incorrectly claimed. Nothing in this module reduces that read cost; it only bounds
what gets rendered/transferred from whatever was read. A periodic/scheduled job that writes
this snapshot to a ``dashboard.json`` file the server reads instead of rebuilding on every
request — :func:`build_snapshot`/:func:`dump_snapshot`/:func:`load_snapshot` below make that
possible (this module is what such a job would call) — is a further M5 todo, not this one;
:mod:`dashboard.server` still calls :func:`build_snapshot` fresh per request today, matching
the existing "never caches, so a long-running process reflects new items without a restart"
design that module's own docstring already commits to. Likewise, :func:`items_at_path` (the
``/api/node-prs`` pagination endpoint's own read) does its own fresh full ``store.query()`` on
*every* page request, including every "show N more" click — pagination here bounds the
*response size* and the amount of per-row transform work (see its own docstring), not the
number of store reads; a real fix needs either a server-side cache of the already-fetched
per-node item lists, or (T5.16(c)'s own ask) a genuine ``path``-filtered, ``limit``/``offset``
-aware ``Store.query()`` — T5.1's read layer, not yet built, is the natural place for that.

**Known limitation, not fixed here**: pagination is offset-based against
:class:`~src.store.base.Store`'s own ``updated_at``-ascending ordering (see its own docstring),
re-sorted fresh on every call — if an item's ``updated_at`` changes (e.g. a reclassification,
or the collector's normal refresh touching it) in the window between an initial page load and
a later "show more" click for the same node, the sort order can shift enough that
:func:`items_at_path`'s next page duplicates a row already shown, or skips one — an inherent
gap of *offset*-based paging over a live-re-sorted collection, not fixable without a stable
pagination cursor (e.g. keyed on ``(repo, number)`` continuation rather than a raw integer
offset), which is a real design change, not a contained fix here.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from src.agents.forecaster import list_predictions
from src.agents.reporter import OTHER
from src.agents.reporter_v1 import TreeNode, is_unclassified_or_other, pr_entry, tree_from_store
from src.store.base import Store
from src.taxonomy import CategoryPath
from src.trends import trends_from_store

# Rendered inline in the initial page load; the rest of a node's own items are reachable via
# the paginated /api/node-prs endpoint (dashboard/server.py) instead of ever landing in the
# page's own HTML -- the one change that makes page size stop scaling with KB size.
DEFAULT_MAX_PRS_PER_NODE = 50


def paginate(seq: list, *, offset: int, limit: int | None) -> list:
    """`seq[offset:]` (unbounded) or `seq[offset:offset + limit]` — the one place this module's
    own :func:`items_at_path` and :func:`~dashboard.api.predictions` slice a page from, so the
    "what does `limit=None` mean" convention can't drift between two independent copies of the
    same ternary (a code-review finding)."""
    return seq[offset:] if limit is None else seq[offset : offset + limit]


def _capped_node(node: TreeNode, *, path: CategoryPath, max_prs: int) -> dict:
    """`node` (see :class:`~src.agents.reporter_v1.TreeNode`) as a JSON-serializable dict, its
    own ``prs`` capped to `max_prs` — ``prs_total``/``prs_truncated`` tell the renderer/UI
    there's more to page through via :func:`items_at_path`, without embedding it. Slices
    `node.prs` (a tuple) directly rather than copying it into a `list` first — cheap for every
    node, not just the ones near the cap (the "Other" bucket this todo's own "Why" names can
    hold tens of thousands of entries)."""
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
        "prs": list(node.prs[:max_prs]),
        "prs_total": len(node.prs),
        "prs_truncated": len(node.prs) > max_prs,
    }


def capped_tree(store: Store, *, max_prs_per_node: int = DEFAULT_MAX_PRS_PER_NODE) -> list[dict]:
    """The classified tree alone (counts + capped top-N leaf rows per node) — one
    :func:`~src.agents.reporter_v1.tree_from_store` scan, nothing else. Factored out of
    :func:`build_snapshot` (which also computes trends/forecasts) so a caller that only wants
    the tree — e.g. :func:`~dashboard.api.tree` — isn't forced to also pay for (and then
    discard) a second full store scan for trends plus a full prediction-log read, a code-review
    finding on an earlier version of :mod:`dashboard.api` that also risked a tree-only read
    crashing on an unrelated corrupt *prediction* record.
    """
    nodes = tree_from_store(store)
    return [_capped_node(node, path=(node.name,), max_prs=max_prs_per_node) for node in nodes]


def build_snapshot(store: Store, *, max_prs_per_node: int = DEFAULT_MAX_PRS_PER_NODE) -> dict:
    """Producing a plain, JSON-serializable dict — everything
    :func:`~dashboard.render.render_snapshot_page` needs to render a page, and everything
    :func:`dump_snapshot`/:func:`load_snapshot` need to round-trip through a file, without ever
    touching `store` again afterward.

    **At least two** independent full store scans, not one — see this module's own Known
    limitations: one inside :func:`capped_tree` (via
    :func:`~src.agents.reporter_v1.tree_from_store`, plus a per-node summary read), a second
    inside :func:`~src.trends.trends_from_store`.
    """
    tree = capped_tree(store, max_prs_per_node=max_prs_per_node)
    forecasts = [asdict(p) for p in list_predictions(store)]
    return {"tree": tree, "trends": trends_from_store(store), "forecasts": forecasts}


def dump_snapshot(snapshot: dict) -> str:
    """`snapshot` as a JSON string — e.g. for a future scheduled job to write to
    ``dashboard.json`` (see this module's own docstring)."""
    return json.dumps(snapshot)


def load_snapshot(raw: str) -> dict:
    """The inverse of :func:`dump_snapshot`."""
    return json.loads(raw)


def items_at_path(
    store: Store, path: CategoryPath, *, offset: int = 0, limit: int | None = None
) -> tuple[list[dict], int]:
    """One page of items classified exactly at `path` (not a descendant) — the same population
    one tree node's own ``prs`` covers (see :class:`~src.agents.reporter_v1.TreeNode`'s
    docstring), computed directly rather than building the whole tree. Returns
    ``(page, total)``: `page` is only the requested slice, each item shaped via
    :func:`~src.agents.reporter_v1.pr_entry` (matching a `TreeNode`'s own ``prs`` shape
    exactly) — filtering happens first and the (offset, limit) slice is taken *before*
    `pr_entry` runs, so a request for a small page never pays to transform every matching item,
    only the ones actually returned.

    `path == (OTHER,)` (or empty) matches the same "no path, or classified OTHER" population
    :func:`~src.agents.reporter_v1.build_tree` folds into its own flat ``Other`` root node —
    both call :func:`~src.agents.reporter_v1.is_unclassified_or_other` so the two populations
    can't silently drift apart.

    `limit=None` returns every remaining item from `offset` onward (unbounded) — the caller
    (:mod:`dashboard.server`'s ``/api/node-prs``) is what enforces a real page-size cap; this
    function's own contract is "give me a slice," not "give me a small slice."

    This signature (`offset`/`limit` accepted here, not applied by the caller after the fact)
    is deliberately where a future ``Store``-level ``limit``/``offset``/path filter (T5.16(c),
    T5.1's read layer) would slot in — today it still does a full, unfiltered `store.query()`
    and slices in Python (see this module's own Known limitations), but callers already only
    ever see a bounded page, so swapping the internals later needs no call-site change.
    """
    if not path or path == (OTHER,):
        items = [item for item in store.query() if is_unclassified_or_other(item.get("path"))]
    else:
        path_list = list(path)
        items = [item for item in store.query() if item.get("path") == path_list]
    total = len(items)
    page = paginate(items, offset=offset, limit=limit)
    return [pr_entry(item) for item in page], total
