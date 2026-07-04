"""Analyst agent (T1.4): classify delta items into the active taxonomy via ``llm.complete``.

The first LLM-backed stage of the intelligence plane: every collected issue/PR that hasn't
been classified yet ("delta" — no ``category`` field, so a rerun only touches new items and
is safe to repeat) is bucketed into one of the KB's current taxonomy categories, and the
result is written back to the item's own KB record — no separate classification table, so a
later reporter/grader reads one record per item, not a join.

Evidence principle: classification never drops or rewrites an item's own source ``url`` — a
category is an *addition* to the record, not a replacement of it. Each classified record also
carries ``taxonomy_version``, the version under which it was classified, so a later audit or
grading pass (T2.2) can ask "what taxonomy was active for this item" without guessing.

An item the model puts outside the active taxonomy (a hallucinated/unknown category name, or
one that only differs by case/whitespace — matched the same way :func:`~src.taxonomy.add_category`
already does) falls back to :data:`OTHER` — mirrors the T0.8 baseline reporter's own fallback
bucket, so an item is never silently unclassified because the model went off-script.

Previously a known limitation, fixed in T1.10: ``Store.upsert_items`` now merges the given
fields onto an existing ``(repo, number)`` record rather than fully replacing it (both
backends). Before that fix, the collector's own ``_normalize()`` — which never carries
``category``/``taxonomy_version`` forward — would silently erase this module's classification
the next time it re-fetched an already-classified item (any new comment/label bumps
``updated_at`` back into the incremental window). Now the collector's re-normalized record
simply doesn't mention ``category``, and the store-level merge leaves the existing value alone
— which also means the "delta" check above (``"category" not in item``) no longer treats a
re-clobbered item as unclassified again, so a re-collected item doesn't get repeatedly and
wastefully re-sent through ``llm.complete``.

T1.5.1 made a taxonomy category a **path** (e.g. ``("ROCm/AMD", "DeepSeek-V4")``), not a flat
name — this module still classifies into one flat label per item (via
:attr:`~src.taxonomy.Taxonomy.labels`, each path's levels joined with `` > ``), unchanged from
before T1.5.1. Teaching this module to classify into a *path* (writing each level from a
controlled per-level label set) is T1.5.2's job, not this one's.
"""

from __future__ import annotations

import sys

from .. import llm
from ..store.base import Store
from ..taxonomy import Taxonomy
from ..taxonomy import get_active as get_active_taxonomy
from .reporter import OTHER

_CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}},
    "required": ["category"],
}

# How much of an item's body to feed the model — enough for context without an unbounded
# prompt on the rare very-long issue.
_BODY_CHARS = 2000


def _prompt(item: dict, categories: tuple[str, ...]) -> str:
    """The classification prompt for one item: title + a body excerpt + the category list."""
    title = item.get("title") or ""
    body = (item.get("body") or "")[:_BODY_CHARS]
    options = ", ".join(categories)
    return (
        "Classify the following GitHub issue/PR into exactly one of these categories: "
        f"{options}.\n\nTitle: {title}\n\nBody: {body}"
    )


def _canonical_category(raw: object, categories: tuple[str, ...]) -> str:
    """Match `raw` against `categories` case/whitespace-insensitively; else :data:`OTHER`.

    Mirrors :func:`~src.taxonomy.add_category`'s own ``.strip().casefold()`` normalization,
    so a reply of ``"ROCm-Build"`` against a taxonomy category ``"rocm-build"`` is recognized
    as the same category instead of silently becoming :data:`OTHER` for what the model
    actually got right. Returns the taxonomy's own canonical spelling on a match, not the
    model's raw casing.
    """
    if not isinstance(raw, str):
        return OTHER
    normalized = raw.strip().casefold()
    for category in categories:
        if category.strip().casefold() == normalized:
            return category
    return OTHER


def classify_item(item: dict, taxonomy: Taxonomy) -> dict:
    """Classify one item into `taxonomy`'s categories.

    Returns a **new** dict — `item`'s fields (its ``url`` citation included, unchanged) plus
    ``category`` and ``taxonomy_version``.

    Raises:
        llm.LLMError: the completion call failed (transport error, timeout, non-JSON reply).
    """
    reply = llm.complete(_prompt(item, taxonomy.labels), json_schema=_CATEGORY_SCHEMA)
    raw_category = reply.get("category") if isinstance(reply, dict) else None
    category = _canonical_category(raw_category, taxonomy.labels)
    return {**item, "category": category, "taxonomy_version": taxonomy.version}


def analyze_store(store: Store) -> list[dict]:
    """Classify every not-yet-classified item in `store` and write the results back.

    "Delta" = items with no ``category`` key yet, so a rerun only classifies what an earlier
    run (or the most recent collection) hasn't already labeled. A failing item (an
    ``llm.LLMError``) is skipped — logged to stderr and left pending for the next run —
    rather than discarding every other item already classified in this batch.

    Returns the newly-classified items (``[]`` if there was nothing to do).

    Raises:
        TaxonomyError: there are pending items but no taxonomy has been created yet.
    """
    pending = [item for item in store.query() if "category" not in item]
    if not pending:
        return []

    active = get_active_taxonomy(store)
    if not active.categories:
        # Nothing to classify into — every item is trivially Other; skip the LLM entirely.
        classified = [
            {**item, "category": OTHER, "taxonomy_version": active.version} for item in pending
        ]
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
