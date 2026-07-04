"""Reporter v1 (T1.6): an LLM-written, per-category cited weekly report.

Builds on T1.4's classification (an item's own ``category`` field) instead of T0.8's
fixed-keyword bucketing: each category's items get batched ``llm.complete`` calls asking for
a concise, insightful one-sentence claim per item, and the report renders each claim through
:func:`~src.agents.reporter._cite` — the same function the v0 reporter uses for its plain
citations — so the model's text gets the identical whitespace-collapsing (no structural
Markdown injection) and URL-or-synthesized-fallback treatment as every other bullet in this
codebase, instead of a second, divergent implementation of "cite a source."

Items are batched in chunks (:data:`_CHUNK_SIZE`) per ``llm.complete`` call, not one call per
whole category: an unbounded single call risks the model's context window, the output-token
cap (a truncated JSON reply discards every claim in the call), and — for the default
``claude_cli`` provider — the OS argv-length limit, which surfaces as a bare ``OSError``, not
``llm.LLMError``. Chunking bounds each call's size and confines a chunk's failure (of *any*
kind — caught broadly, not just ``llm.LLMError``) to that chunk's items, logged to stderr.

Uncategorized items (nothing T1.4 has classified yet) get the same :func:`_cite` treatment
under :data:`~src.agents.reporter.OTHER`, with no LLM call — there's nothing classified to
summarize. If there ARE such items, the report says so and points at ``python -m
src.analyze`` — silently degrading every section to plain citations with no explanation
would be a worse failure mode than a visible note.
"""

from __future__ import annotations

import sys

from .. import llm
from ..store.base import Store
from .reporter import OTHER, _cite, _num

# Per item, inside a batched per-category prompt — kept short since several items share one
# call (unlike a single-item agent's budget, e.g. the Analyst's 2000 chars).
_BODY_CHARS = 500

# Items per llm.complete call. Bounds prompt size (context window, argv length for claude_cli)
# and output size (a truncated JSON reply loses every claim in the call, not just the
# overflow items) regardless of how large a category grows over the project's life.
_CHUNK_SIZE = 25

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


def _chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


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


def _claim_chunk(category: str, chunk: list[dict]) -> dict[int, str]:
    """Ask the model for claims about one chunk; return ``{index-within-chunk: text}``.

    Any failure (LLM transport/timeout/parse error, or — for an oversized prompt on the
    ``claude_cli`` provider — a bare ``OSError`` from the OS argv-length limit, which is not
    an ``llm.LLMError``) is caught broadly and logged: this chunk's items fall back to plain
    citations rather than losing the rest of the category, or the whole report.
    """
    try:
        reply = llm.complete(_category_prompt(category, chunk), json_schema=_CLAIMS_SCHEMA)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"reporter_v1: skipping a {category!r} chunk: {exc}", file=sys.stderr)
        return {}
    claims = reply.get("claims") if isinstance(reply, dict) else None
    covered: dict[int, str] = {}
    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        idx, text = claim.get("item_index"), claim.get("text")
        if isinstance(idx, int) and 0 <= idx < len(chunk) and isinstance(text, str) and text:
            covered.setdefault(idx, text)
    return covered


def _cited_claims(category: str, items: list[dict]) -> list[str]:
    """Render `items` as cited Markdown bullets, LLM text where available.

    Every returned line goes through :func:`~src.agents.reporter._cite`, so the evidence
    principle (a real source URL, structure-safe text) holds whether the model covers an
    item or not: any item a chunk's call didn't (validly) cover falls back to `_cite`'s plain
    title, matching the completeness guarantee the v0 reporter already had.
    """
    lines: list[str] = []
    for chunk in _chunks(items, _CHUNK_SIZE):
        covered = _claim_chunk(category, chunk)
        lines.extend(
            _cite(item, text=covered[i]) if i in covered else _cite(item)
            for i, item in enumerate(chunk)
        )
    return lines


def build_report_v1(items: list[dict], *, title: str = "vLLM (ROCm) weekly digest") -> str:
    """Render `items` into an LLM-written, per-category cited Markdown digest.

    Groups by each item's own ``category`` (T1.4's classification) — an item with no
    ``category`` yet, or one the Analyst itself couldn't classify, both fall under
    :data:`OTHER` (a single merged section, not two). Deterministic given a deterministic
    ``llm.complete`` (tests mock it); categories appear in first-occurrence order in `items`,
    ``Other`` last.
    """
    buckets: dict[str, list[dict]] = {}
    never_classified = 0
    for item in items:
        category = item.get("category")
        if not category:
            never_classified += 1
        buckets.setdefault(category or OTHER, []).append(item)
    other = buckets.pop(OTHER, [])

    repos = {item.get("repo") for item in items if item.get("repo")}
    lines = [f"# {title}", "", f"{len(items)} items across {len(repos)} repos.", ""]
    if never_classified:
        lines.append(
            f"_{never_classified} item(s) have no category yet — run `python -m "
            "src.analyze` for a fully classified digest; they're listed under Other below._"
        )
        lines.append("")
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
