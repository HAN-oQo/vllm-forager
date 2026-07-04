"""Reporter v1 (T1.6): an LLM-written, per-category cited weekly report.

Builds on T1.4's classification (an item's own ``category`` field) instead of T0.8's
fixed-keyword bucketing: each category's items get ONE batched ``llm.complete`` call asking
for a concise, insightful one-sentence claim per item, and the report renders each claim
immediately followed by *that item's own* source URL. The citation is always attached by this
module's own code, never trusted from the model's text — so "every claim line has ≥1
evidence URL" (CLAUDE.md's evidence principle) holds regardless of what the model writes,
whether it fails outright, or returns fewer/invalid claims than items.

Uncategorized items (nothing T1.4 has classified yet) get a plain citation line under
:data:`~src.agents.reporter.OTHER` — there's nothing classified to summarize, so no LLM call.
"""

from __future__ import annotations

from .. import llm
from ..store.base import Store
from .reporter import OTHER, _cite, _num

# Per item, inside a batched per-category prompt — kept short since several items share one
# call (unlike a single-item agent's budget, e.g. the Analyst's 2000 chars).
_BODY_CHARS = 500

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "item_index": {"type": "integer"},
                },
                "required": ["text", "item_index"],
            },
        },
    },
    "required": ["claims"],
}


def _category_prompt(category: str, items: list[dict]) -> str:
    listing = "\n\n".join(
        f"{i}: {item.get('title') or ''}\n{(item.get('body') or '')[:_BODY_CHARS]}"
        for i, item in enumerate(items)
    )
    return (
        f"These are GitHub issues/PRs classified under the {category!r} category. For as "
        "many as you can, write ONE concise, insightful sentence about it (what it reports "
        "or proposes, and why it matters). Reply with `claims`: a list of "
        "{text, item_index} objects, item_index matching the 0-based number below.\n\n"
        f"{listing}"
    )


def _cited_claims(category: str, items: list[dict]) -> list[str]:
    """Render `items` as cited Markdown bullets, LLM text where available.

    Every returned line carries its item's own source URL — attached here, never trusted
    from the model's reply — so the evidence principle holds even if the model call fails,
    returns nothing, or names an out-of-range/duplicate ``item_index``: any item the model
    didn't (validly) cover falls back to :func:`~src.agents.reporter._cite`'s plain citation,
    matching the completeness guarantee the v0 reporter already had (every item is cited).
    """
    covered: dict[int, str] = {}
    try:
        reply = llm.complete(_category_prompt(category, items), json_schema=_CLAIMS_SCHEMA)
        claims = reply.get("claims") if isinstance(reply, dict) else None
        for claim in claims or []:
            if not isinstance(claim, dict):
                continue
            idx, text = claim.get("item_index"), claim.get("text")
            if isinstance(idx, int) and 0 <= idx < len(items) and isinstance(text, str) and text:
                covered.setdefault(idx, text)
    except llm.LLMError:
        pass  # every item still gets a plain citation line below

    return [
        f"- {covered[i]} — {item.get('url') or ''}" if i in covered else _cite(item)
        for i, item in enumerate(items)
    ]


def build_report_v1(items: list[dict], *, title: str = "vLLM (ROCm) weekly digest") -> str:
    """Render `items` into an LLM-written, per-category cited Markdown digest.

    Groups by each item's own ``category`` (T1.4's classification), not the fixed-keyword
    taxonomy — an item with no ``category`` yet falls under :data:`OTHER`. Deterministic
    given a deterministic ``llm.complete`` (tests mock it); categories appear in
    first-occurrence order in `items`, ``Other`` last.
    """
    buckets: dict[str, list[dict]] = {}
    other: list[dict] = []
    for item in items:
        category = item.get("category")
        (other if not category else buckets.setdefault(category, [])).append(item)

    repos = {item.get("repo") for item in items if item.get("repo")}
    lines = [f"# {title}", "", f"{len(items)} items across {len(repos)} repos.", ""]
    for category, bucket in [*buckets.items(), (OTHER, other)]:
        if not bucket:
            continue
        ordered = sorted(bucket, key=lambda item: (item.get("repo") or "", _num(item)))
        ordered.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        lines.append(f"## {category} ({len(bucket)})")
        if category == OTHER:
            lines.extend(_cite(item) for item in ordered)
        else:
            lines.extend(_cited_claims(category, ordered))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def report_from_store(store: Store, *, title: str = "vLLM (ROCm) weekly digest") -> str:
    """Read all items from `store` and render the v1 report."""
    return build_report_v1(store.query(), title=title)
