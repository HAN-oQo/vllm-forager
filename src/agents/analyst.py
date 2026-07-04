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

An item the model puts outside the active taxonomy (a hallucinated/unknown category name)
falls back to :data:`OTHER` — mirrors the T0.8 baseline reporter's own fallback bucket, so an
item is never silently unclassified because the model went off-script.
"""

from __future__ import annotations

from .. import llm
from ..store.base import Store
from ..taxonomy import Taxonomy
from ..taxonomy import get_active as get_active_taxonomy

OTHER = "Other"

_CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}},
    "required": ["category"],
}

# How much of an item's body to feed the model — enough for context without an unbounded
# prompt on the rare very-long issue.
_BODY_CHARS = 2000


def _prompt(item: dict, categories: list[str]) -> str:
    """The classification prompt for one item: title + a body excerpt + the category list."""
    title = item.get("title") or ""
    body = (item.get("body") or "")[:_BODY_CHARS]
    options = ", ".join(categories)
    return (
        "Classify the following GitHub issue/PR into exactly one of these categories: "
        f"{options}.\n\nTitle: {title}\n\nBody: {body}"
    )


def classify_item(item: dict, taxonomy: Taxonomy) -> dict:
    """Classify one item into `taxonomy`'s categories.

    Returns a **new** dict — `item`'s fields (its ``url`` citation included, unchanged) plus
    ``category`` and ``taxonomy_version``. A category the model names outside
    `taxonomy.categories` becomes :data:`OTHER` rather than being trusted verbatim.
    """
    reply = llm.complete(_prompt(item, list(taxonomy.categories)), json_schema=_CATEGORY_SCHEMA)
    category = reply.get("category") if isinstance(reply, dict) else None
    if category not in taxonomy.categories:
        category = OTHER
    return {**item, "category": category, "taxonomy_version": taxonomy.version}


def analyze_store(store: Store) -> list[dict]:
    """Classify every not-yet-classified item in `store` and write the results back.

    "Delta" = items with no ``category`` key yet, so a rerun only classifies what an earlier
    run (or the most recent collection) hasn't already labeled.

    Returns the newly-classified items (``[]`` if there was nothing to do).

    Raises:
        TaxonomyError: no taxonomy has been created yet — there's nothing to classify into.
    """
    active = get_active_taxonomy(store)
    pending = [item for item in store.query() if "category" not in item]
    if not pending:
        return []
    classified = [classify_item(item, active) for item in pending]
    store.upsert_items(classified)
    return classified
