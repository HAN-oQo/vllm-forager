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

T1.5.4 adds a second, tree-shaped report alongside the original flat-per-category one above:
:func:`build_tree` groups items by their classified ``path`` (T1.5.2) instead of the flat
``category``, :func:`render_tree_markdown` renders it as an indented 大 → 소(summary) → 소소 →
PRs digest (the approved ``docs/design/report-tree-mockup.html``), and ``TreeNode.to_dict()``
is what ``report.py --tree`` writes to ``data/reports/<week>.tree.json`` for T1.5.5's
dashboard to consume. The two report styles are independent — building the tree never calls
``llm.complete`` itself; the per-node prose already lives in the KB from T1.5.3's summarizer.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from .. import llm
from ..store.base import Store
from ..taxonomy import LEVEL_SEPARATOR, CategoryPath
from .reporter import OTHER, _cite, _num, evidence_url
from .summarizer import get_node_summary

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


# --------------------------------------------------------------------- T1.5.4: tree report


@dataclass(frozen=True)
class TreeNode:
    """One node in the taxonomy tree: `count`/`prs` cover only items classified **exactly**
    at this node's path — a deeper item contributes to a `children` node instead, never both.

    ``count`` rolls up this node's own item count plus every descendant's (so a parent always
    shows its subtree total, per T1.5.4's own "Why"). ``gaps`` is a placeholder — every node
    reports 0 here regardless. T2.4's parity matrix (:mod:`src.parity`) is now built and used
    by :mod:`~src.agents.scout`, but nothing wires its gap data into a tree node yet — that's
    T5.4's own todo ("Parity diagram"), not this function's. A node with ``count == 0`` can
    never be constructed by :func:`build_tree`
    (it only ever creates a node because some item's path passes through it), so "empty
    branches pruned" is a structural guarantee, not a separate filtering step.
    """

    name: str
    summary: str | None
    count: int
    gaps: int
    children: tuple[TreeNode, ...]
    prs: tuple[dict, ...]

    def to_dict(self) -> dict:
        """The ``{name, summary, count, gaps, children[], prs[]}`` shape T1.5.4 specifies,
        recursively — what gets written to ``data/reports/<week>.tree.json``."""
        return {
            "name": self.name,
            "summary": self.summary,
            "count": self.count,
            "gaps": self.gaps,
            "children": [child.to_dict() for child in self.children],
            "prs": [dict(pr) for pr in self.prs],
        }


def pr_entry(item: dict) -> dict:
    """The minimal, JSON-serializable shape one leaf PR/issue carries in a tree node: enough
    for T1.5.5's dashboard to render a cited row + a merged/open/issue state chip, without
    bloating ``tree.json`` with every raw item field (body text, labels, etc.).

    ``url`` uses :func:`~src.agents.reporter.evidence_url` — the same URL-or-synthesized
    logic :func:`~src.agents.reporter._cite` uses — so a citation looks identical whether it
    came from the flat report or this one. ``repo``/``number`` are `None` when `item` lacks
    them (not omitted) — :func:`~src.agents.reporter._cite`, called on this dict by
    :func:`render_tree_markdown`, treats a `None` value the same as a missing key (renders
    ``?``), so this never surfaces a Python-literal ``None`` in report text.

    Public (not ``_pr_entry``) since T5.16: :mod:`dashboard.snapshot`'s ``items_at_path`` calls
    this directly too, for the same "one leaf row" shape a paginated dashboard endpoint returns.
    """
    return {
        "repo": item.get("repo"),
        "number": item.get("number"),
        "title": item.get("title") or "",
        "url": evidence_url(item),
        "state": item.get("state"),
        "type": item.get("type"),
    }


def is_unclassified_or_other(path: object) -> bool:
    """True if `path` (an item's raw ``"path"`` field) belongs in the flat ``Other`` bucket —
    missing/not-yet-classified, or explicitly classified :data:`~src.agents.reporter.OTHER`.

    Public (not inlined) since T5.16: :func:`build_tree` and
    :func:`~dashboard.snapshot.items_at_path` both need the *identical* rule (an item the tree
    counts under ``Other`` must be the same population a paginated ``Other`` page returns) —
    two independent copies previously risked silently drifting apart if one were ever updated
    without the other.
    """
    return not isinstance(path, list) or not path or path == [OTHER]


def build_tree(
    items: list[dict], summary_lookup: Callable[[CategoryPath], str | None] = lambda path: None
) -> list[TreeNode]:
    """Group `items` into a nested tree by their classified ``path`` (T1.5.2).

    An item with no ``path`` yet, or one classified :data:`~src.agents.reporter.OTHER`, is
    collected into one flat ``Other`` root node (no further nesting — there's nothing to nest,
    same convention :func:`build_report_v1` already uses). `summary_lookup` supplies each
    node's synthesis (T1.5.3's :func:`~src.agents.summarizer.get_node_summary`, injected so
    this function stays pure/offline-testable — the default returns ``None`` everywhere, i.e.
    "no summaries available").

    Root nodes are ordered by first appearance in `items`; each node's own children are
    ordered the same way. Deterministic given deterministic input order.

    This does its own "walk every prefix of every classified path" pass, independent of
    :func:`~src.agents.summarizer._nodes_from_items`'s own (similar-looking) walk — not
    shared, since the two need different aggregations: summarizer wants the cumulative union
    of items under a prefix (to feed an LLM synthesis), this wants exact-match-per-node plus
    structural child discovery (to roll up `count`). A shared low-level primitive is feasible
    but is a real refactor, deferred to avoid risking T1.5.3's already-shipped behavior here.
    """
    other_items: list[dict] = []
    exact: dict[CategoryPath, list[dict]] = {}
    # dict-as-ordered-set (mirrors Taxonomy.children()'s own `seen: dict[str, str]` pattern) —
    # one structure for both "what are prefix's children" and "in what order", not two.
    children_of: dict[CategoryPath, dict[str, None]] = defaultdict(dict)

    for item in items:
        path = item.get("path")
        if is_unclassified_or_other(path):
            other_items.append(item)
            continue
        assert isinstance(path, list)  # is_unclassified_or_other's own contract guarantees this
        path_t = tuple(path)
        exact.setdefault(path_t, []).append(item)
        for depth in range(len(path_t)):
            prefix, level = path_t[:depth], path_t[depth]
            children_of[prefix].setdefault(level, None)

    def make_node(prefix: CategoryPath) -> TreeNode:
        children = tuple(make_node((*prefix, name)) for name in children_of.get(prefix, {}))
        own = exact.get(prefix, [])
        return TreeNode(
            name=prefix[-1],
            summary=summary_lookup(prefix),
            count=len(own) + sum(child.count for child in children),
            gaps=0,
            children=children,
            prs=tuple(pr_entry(item) for item in own),
        )

    tree = [make_node((name,)) for name in children_of.get((), {})]
    if other_items:
        tree.append(
            TreeNode(
                name=OTHER,
                summary=None,
                count=len(other_items),
                gaps=0,
                children=(),
                prs=tuple(pr_entry(item) for item in other_items),
            )
        )
    return tree


def _node_heading(name: str, count: int, depth: int) -> str:
    """A Markdown heading for `depth` (0 = top level), or a bold line past H6 — Markdown
    doesn't render a 7th ``#`` as a heading at all, so piling on more would silently stop
    working instead of merely looking the same as H6 (which distinct depths >= 4 already do).
    """
    level = depth + 2
    if level <= 6:
        return f"{'#' * level} {name} ({count})"
    return f"**{name} ({count})**"


def render_tree_markdown(nodes: list[TreeNode], *, title: str = "vLLM (ROCm) weekly digest") -> str:
    """Render `nodes` as indented Markdown: 대 → 소(summary) → 소소 → PRs, per the approved
    report-tree mockup — one heading level per tree depth up to H6 (a bold line beyond that,
    see :func:`_node_heading`), each node's summary (if any) as an italic line, and its own
    PRs as :func:`~src.agents.reporter._cite` bullets before its children.
    """
    total = sum(node.count for node in nodes)
    lines = [f"# {title}", "", f"{total} items.", ""]

    def render(node: TreeNode, depth: int) -> None:
        lines.append(_node_heading(node.name, node.count, depth))
        if node.summary:
            lines.append(f"_{node.summary}_")
        lines.append("")
        lines.extend(_cite(pr) for pr in node.prs)
        if node.prs:
            lines.append("")
        for child in node.children:
            render(child, depth + 1)

    for node in nodes:
        render(node, 0)
    return "\n".join(lines).rstrip() + "\n"


def tree_from_store(store: Store) -> list[TreeNode]:
    """Read all items from `store`, build the tree, and attach each node's stored T1.5.3
    summary (:func:`~src.agents.summarizer.get_node_summary`) — the impure entry point CLI/
    dashboard callers use; :func:`build_tree` itself stays store-free and pure for testing.

    One node's per-request summary read is not batched against :class:`~src.store.base.Store`
    (no batch-read API exists) — for `N` tree nodes this is `N` sequential ``get_state`` calls
    (a full ``state.json`` re-read/re-parse per call on :class:`~src.store.jsonl_store.
    JsonlStore`, or `N` network round trips on Firestore), the same "one call per item, no
    cap" shape this codebase has bounded twice before (:mod:`~src.agents.reporter_v1`'s own
    ``_CHUNK_SIZE``, :mod:`~src.agents.summarizer`'s ``_MAX_ITEMS_PER_NODE``) — not fixed here
    since it would need a new ``Store``-level batch-read primitive, out of scope for adding
    tree building itself.
    """

    def lookup(path: CategoryPath) -> str | None:
        try:
            summary = get_node_summary(store, path)
        except Exception as exc:  # noqa: BLE001 - a corrupt KB record must not crash the run
            print(
                f"reporter_v1: skipping summary for {LEVEL_SEPARATOR.join(path)}: {exc}",
                file=sys.stderr,
            )
            return None
        return summary.text if summary is not None else None

    return build_tree(store.query(), summary_lookup=lookup)
