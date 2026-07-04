"""Analyst agent (T1.4, path-classifying since T1.5.2): classify delta items into the active
taxonomy via ``llm.complete``.

The first LLM-backed stage of the intelligence plane: every collected issue/PR that hasn't
been classified yet ("delta" — no ``path`` field, so a rerun only touches new items and is
safe to repeat) is bucketed into the KB's current taxonomy, and the result is written back to
the item's own KB record — no separate classification table, so a later reporter/grader reads
one record per item, not a join.

Evidence principle: classification never drops or rewrites an item's own source ``url`` — a
category is an *addition* to the record, not a replacement of it. Each classified record also
carries ``taxonomy_version``, the version under which it was classified, so a later audit or
grading pass (T2.2) can ask "what taxonomy was active for this item" without guessing.

T1.5.2: the taxonomy is a tree of **paths** (T1.5.1), not a flat list, so classification now
walks the tree one level at a time — the model returns a whole path in one ``llm.complete``
call (cheaper than one call per level), and :func:`_canonical_path` then validates it level by
level against :meth:`~src.taxonomy.Taxonomy.children`: the first level whose value isn't one
of that step's controlled options is where drift stops the walk, keeping the valid prefix
classified so far (an item is never rejected outright for going one level too far off-script).
An item that doesn't validate at even level 0 (a hallucinated root, or the model went
completely off-script) falls back to a depth-1 :data:`OTHER` path — mirrors the T0.8 baseline
reporter's own fallback bucket, so an item is never silently unclassified.

Two fields are written: ``path`` (the list of levels — T1.5.4's tree report reads this) and
``category`` (``path`` flattened via :data:`~src.taxonomy.LEVEL_SEPARATOR` — the same join
:attr:`~src.taxonomy.Taxonomy.labels` uses — kept for T1.6/T1.7's reporter/trends modules,
which only know how to group by one flat string per item and haven't been taught to read
``path`` yet). Known limitation, not fixed here: once the taxonomy grows past depth 1,
:mod:`src.trends`'s ``category_trends`` will bucket by the *full* flattened path (e.g. an
opaque ``"ROCm/AMD > DeepSeek-V4 > performance"`` bucket) rather than rolling up under
``"ROCm/AMD"`` — a T1.6/T1.7 concern to fix once ``path`` is available to group by, not this
module's.

Known limitation, not fixed here: :func:`_path_prompt` lists every known path in one
unbounded, comma-joined hint string, and the model never sees :meth:`~src.taxonomy.
Taxonomy.children`'s exact per-level options before answering (:func:`_canonical_path`
validates its whole guess post-hoc instead) — both fine at today's taxonomy size (never
seeded in production yet), but worth revisiting once the taxonomy is large/deep enough that
prompt length or guess-then-validate accuracy actually matters (T1.5.4/T2.3).

Previously a known limitation, fixed in T1.10: ``Store.upsert_items`` now merges the given
fields onto an existing ``(repo, number)`` record rather than fully replacing it (both
backends). Before that fix, the collector's own ``_normalize()`` — which never carries
classification fields forward — would silently erase this module's classification the next
time it re-fetched an already-classified item (any new comment/label bumps ``updated_at`` back
into the incremental window). Now the collector's re-normalized record simply doesn't mention
``path``/``category``, and the store-level merge leaves the existing value alone — which also
means the "delta" check above no longer treats a re-clobbered item as unclassified again, so a
re-collected item doesn't get repeatedly and wastefully re-sent through ``llm.complete``.
"""

from __future__ import annotations

import sys

from .. import llm
from ..store.base import Store
from ..taxonomy import LEVEL_SEPARATOR, CategoryPath, Taxonomy, casefold_label
from ..taxonomy import get_active as get_active_taxonomy
from .reporter import OTHER

_PATH_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "array", "items": {"type": "string"}}},
    "required": ["path"],
}

# How much of an item's body to feed the model — enough for context without an unbounded
# prompt on the rare very-long issue.
_BODY_CHARS = 2000


def _path_prompt(item: dict, taxonomy: Taxonomy) -> str:
    """The classification prompt for one item: title + a body excerpt + the known paths."""
    title = item.get("title") or ""
    body = (item.get("body") or "")[:_BODY_CHARS]
    known_paths = ", ".join(taxonomy.labels)
    return (
        "Classify the following GitHub issue/PR into this taxonomy tree. Known category "
        f"paths (root > ... > leaf): {known_paths}.\n\n"
        "Respond with a path as a JSON array of level names, root first — e.g. "
        '["ROCm/AMD", "DeepSeek-V4", "performance"]. Use an existing path exactly where it '
        "fits; if you're only confident about the higher levels, return a shorter path "
        "(just the levels you're sure of) rather than guessing the rest.\n\n"
        f"Title: {title}\n\nBody: {body}"
    )


def _canonical_path(raw: object, taxonomy: Taxonomy) -> CategoryPath:
    """Validate `raw` level by level against `taxonomy`'s tree; else a depth-1 :data:`OTHER`.

    Walks `raw` from the root, matching each level case/whitespace-insensitively
    (:func:`~src.taxonomy.casefold_label`) against :meth:`~src.taxonomy.Taxonomy.children` of
    the path validated so far — exactly the "controlled per-level label set" that keeps a
    hallucinated or drifted level from ever entering the KB. The walk stops at the first level
    that doesn't match (keeping the valid prefix, not discarding the whole classification);
    an empty or entirely-invalid `raw` (not a list of strings, or level 0 doesn't match)
    becomes ``(OTHER,)`` — never an empty path.
    """
    if not isinstance(raw, list) or not raw or not all(isinstance(level, str) for level in raw):
        return (OTHER,)
    validated: list[str] = []
    for level in raw:
        options = taxonomy.children(tuple(validated))
        normalized = casefold_label(level)
        match = next((opt for opt in options if casefold_label(opt) == normalized), None)
        if match is None:
            break
        validated.append(match)
    return tuple(validated) if validated else (OTHER,)


def _classified_record(item: dict, path: CategoryPath, taxonomy_version: int) -> dict:
    """`item`'s fields (its ``url`` citation included, unchanged) plus ``path`` (list of
    levels), ``category`` (``path`` flattened via :data:`~src.taxonomy.LEVEL_SEPARATOR`, for
    callers that only read a flat label), and ``taxonomy_version``.

    The one place a classification result — LLM-produced or a fallback — is assembled, so
    :func:`classify_item` and :func:`analyze_store`'s empty-taxonomy short-circuit can never
    silently disagree on the shape of "no valid classification" (e.g. if :data:`OTHER`'s own
    representation ever changes).
    """
    return {
        **item,
        "path": list(path),
        "category": LEVEL_SEPARATOR.join(path),
        "taxonomy_version": taxonomy_version,
    }


def classify_item(item: dict, taxonomy: Taxonomy) -> dict:
    """Classify one item into a path in `taxonomy`'s tree.

    Returns a **new** dict — see :func:`_classified_record`.

    Raises:
        llm.LLMError: the completion call failed (transport error, timeout, non-JSON reply).
    """
    reply = llm.complete(_path_prompt(item, taxonomy), json_schema=_PATH_SCHEMA)
    raw_path = reply.get("path") if isinstance(reply, dict) else None
    path = _canonical_path(raw_path, taxonomy)
    return _classified_record(item, path, taxonomy.version)


def analyze_store(store: Store) -> list[dict]:
    """Classify every not-yet-classified item in `store` and write the results back.

    "Delta" = items with no ``path`` key yet, so a rerun only classifies what an earlier run
    (or the most recent collection) hasn't already labeled — this also means an item
    classified by the pre-T1.5.2 flat-``category``-only Analyst gets re-classified once, to
    backfill its ``path``. A failing item (an ``llm.LLMError``) is skipped — logged to stderr
    and left pending for the next run — rather than discarding every other item already
    classified in this batch.

    Returns the newly-classified items (``[]`` if there was nothing to do).

    Raises:
        TaxonomyError: there are pending items but no taxonomy has been created yet.
    """
    pending = [item for item in store.query() if "path" not in item]
    if not pending:
        return []

    active = get_active_taxonomy(store)
    if not active.categories:
        # Nothing to classify into — every item is trivially Other; skip the LLM entirely.
        classified = [_classified_record(item, (OTHER,), active.version) for item in pending]
    else:
        classified = []
        for item in pending:
            try:
                classified.append(classify_item(item, active))
            except llm.LLMError as exc:
                print(
                    f"analyst: skipping {item.get('repo')}#{item.get('number')}: {exc}",
                    file=sys.stderr,
                )

    if classified:
        store.upsert_items(classified)
    return classified
