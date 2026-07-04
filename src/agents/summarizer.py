"""Node summarizer agent (T1.5.3): a 1-2 line cited synthesis per taxonomy tree node.

Turns a bucket of classified items under one taxonomy path node (e.g. ``("ROCm/AMD",
"DeepSeek-V4", "performance")``) into one scannable "what's happening here" line via
``llm.complete`` — the biggest readability lever after T1.5.1's nesting itself (a flat list of
50 PRs under one category name tells you nothing at a glance; one synthesized sentence does).
T1.5.4's tree report reads these summaries to render the 大 → 소(summary) → 소소 → PRs mockup.

Node discovery: every distinct **prefix** of every classified item's ``path`` (T1.5.2) is a
node — depth 1 through the item's own full depth — so a 4-level classified item contributes to
4 node summaries (its root, root+1, root+2, and its own full path), each a progressively
narrower slice of the same underlying items. An item with no ``path`` yet (T1.5.2 hasn't run)
or one classified :data:`~src.agents.reporter.OTHER` is excluded — ``OTHER`` is the leftover
bucket by definition, not a meaningful node to synthesize. Known limitation, not fixed here:
this exclusion checks ``path == [OTHER]``, which can't distinguish the fallback sentinel from a
genuine, human-registered taxonomy category literally named "Other" (:mod:`~src.taxonomy`
places no reserved-word restriction on category names) — inherited from
:mod:`~src.agents.analyst`'s existing sentinel design, not introduced here.

Evidence principle, structural not inline: unlike a per-item citation (T0.8/T1.6's ``_cite``,
one URL trailing one claim), a node summary synthesizes *many* items into one sentence, so its
citation is a separate ``evidence`` field (the URLs of every item behind the synthesis) —
mirrors :class:`~src.agents.forecaster.Prediction`'s own ``evidence`` tuple, not raw links
woven into prose (which the approved report-tree mockup's summary lines never do either). The
hallucination guard mirrors T1.8's ``answer_query``: an empty node (no items) returns ``None``
**without calling the LLM at all** — there is nothing to synthesize, so "no summary" is a
code-level guarantee, not something hoped for from the model. A summary whose own text is
missing/empty after the call also becomes ``None``, never an empty string written to the KB.

Storage: one summary per node, keyed by its flattened path (the same `` > `` join
:attr:`~src.taxonomy.Taxonomy.labels`/:mod:`~src.agents.analyst` use) in the KB's generic state
map (``node_summary@<flattened path>``) — a plain overwrite, not an append-only log (a node's
summary reflects its *current* item set, not a history of past syntheses, unlike T1.5's
prediction log) — so T1.5.4's tree report can look a node's summary up via
:func:`get_node_summary` without re-running this module on every render. Because each node's
write is a full recompute-from-scratch (never a read-modify-write of the same key), it can't
suffer the "lost update" race :mod:`~src.taxonomy`/:mod:`~src.agents.forecaster` document for
their own read-then-write patterns — but two overlapping :func:`summarize_store` runs can still
race on which run's write lands last per node (whichever finishes last wins, even if it saw a
staler item snapshot); not fixed here, matching those same siblings' "known limitation, real
fix belongs at the Store layer" stance.

Known limitations, not fixed here (both low-stakes: production has never created a taxonomy
against the real KB, so none of T1.5.1-T1.5.3 has run for real yet): (1) every node — including
an ancestor whose item set is just the union of children already summarized separately — gets
its own full ``llm.complete`` call re-synthesizing from raw items, rather than a cheaper
hierarchical roll-up of child summaries; call count scales roughly as items × average taxonomy
depth. (2) :func:`summarize_store` always recomputes every node every run with no
delta/skip-if-unchanged cache — a deliberate freshness-over-cost tradeoff (a node's item set
only grows, so a naive cache would leave earlier summaries permanently stale), worth revisiting
once real run volume exists.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from .. import llm
from ..store.base import Store
from ..taxonomy import LEVEL_SEPARATOR, CategoryPath
from .reporter import OTHER

_SUMMARY_KEY_PREFIX = "node_summary@"

_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

# Per item, inside a batched per-node prompt — several items share one call, so kept short
# (mirrors reporter_v1's own per-item budget inside its batched prompt).
_BODY_CHARS = 500

# Items per node's llm.complete call. Bounds prompt size: a root node's item set is the union
# of every descendant's items, so without a cap it grows unbounded as the KB grows. For the
# claude_cli provider, llm.complete passes the whole prompt as an argv element — an oversized
# prompt raises a bare OSError (not llm.LLMError), which would otherwise escape
# summarize_store's per-node isolation and abort the entire run. Mirrors reporter_v1's
# _CHUNK_SIZE, added for the identical reason.
_MAX_ITEMS_PER_NODE = 25


@dataclass(frozen=True)
class NodeSummary:
    """One taxonomy node's synthesis: its path, the 1-2 line text, and its evidence URLs.

    Raises:
        ValueError: `text` is blank, or `evidence` is empty — the invariants
            :func:`summarize_node` already enforces at the call site, re-enforced here so a
            corrupted/hand-edited KB record (read back via :func:`get_node_summary`) can never
            silently masquerade as a validated synthesis.
    """

    path: CategoryPath
    text: str
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a node summary's text must not be blank")
        if not self.evidence:
            raise ValueError("a node summary must cite at least one item URL")

    def to_json(self) -> str:
        """Serialize for storage in the KB's state map."""
        return json.dumps({"text": self.text, "evidence": list(self.evidence)})

    @staticmethod
    def from_json(path: CategoryPath, raw: str) -> NodeSummary:
        """Deserialize a value previously produced by :meth:`to_json`, paired with the `path`
        it was stored under (the state-map key, not part of the JSON body itself)."""
        data = json.loads(raw)
        return NodeSummary(path=path, text=data["text"], evidence=tuple(data["evidence"]))


def _node_key(path: CategoryPath) -> str:
    return f"{_SUMMARY_KEY_PREFIX}{LEVEL_SEPARATOR.join(path)}"


def _node_prompt(path: CategoryPath, items: list[dict]) -> str:
    heading = LEVEL_SEPARATOR.join(path)
    listing = "\n\n".join(
        f"{item.get('title') or ''}\n{(item.get('body') or '')[:_BODY_CHARS]}" for item in items
    )
    return (
        f"These {len(items)} GitHub issues/PRs are classified under the taxonomy category "
        f"{heading!r}. Write ONE-to-TWO concise, insightful sentences synthesizing what's "
        "happening in this category overall (the dominant theme(s), not a list of each item). "
        f"Reply with `summary`: the sentence(s).\n\nItems:\n{listing}"
    )


def summarize_node(path: CategoryPath, items: list[dict]) -> NodeSummary | None:
    """A 1-2 line cited synthesis of `items` (all classified under `path`) — or ``None``.

    The hallucination guard: an empty `items` refuses to answer (never calling the LLM at
    all), so "an empty node yields no summary" is a code-level guarantee. A reply with a
    missing/blank ``summary`` also becomes ``None`` rather than an empty string written to
    the KB. `items` is capped at :data:`_MAX_ITEMS_PER_NODE` — both the prompt and `evidence`
    only ever reflect the items actually shown to the model, never a claim about items it
    never saw. `evidence` is every (capped) item's URL (deduped, order preserved) — the
    structural citation for whatever the model synthesized, since the sentence itself doesn't
    (and shouldn't, per the approved report-tree mockup) embed raw links.

    Raises:
        llm.LLMError: the completion call failed (transport error, timeout, non-JSON reply).
    """
    if not items:
        return None
    items = items[:_MAX_ITEMS_PER_NODE]
    reply = llm.complete(_node_prompt(path, items), json_schema=_SUMMARY_SCHEMA)
    text = reply.get("summary") if isinstance(reply, dict) else None
    if not isinstance(text, str) or not text.strip():
        return None
    evidence = tuple(dict.fromkeys(item["url"] for item in items if item.get("url")))
    if not evidence:
        return None
    return NodeSummary(path=path, text=text.strip(), evidence=evidence)


def _nodes_from_items(items: list[dict]) -> dict[CategoryPath, list[dict]]:
    """Group `items` by every distinct prefix of their classified ``path`` (T1.5.2), excluding
    unclassified items and the :data:`OTHER` fallback bucket."""
    nodes: dict[CategoryPath, list[dict]] = {}
    for item in items:
        path = item.get("path")
        if not isinstance(path, list) or not path or path == [OTHER]:
            continue
        for depth in range(1, len(path) + 1):
            nodes.setdefault(tuple(path[:depth]), []).append(item)
    return nodes


def summarize_store(store: Store) -> dict[CategoryPath, NodeSummary]:
    """Summarize every taxonomy node exercised by `store`'s classified items.

    Every node's summary is recomputed from its current item set and written to the KB,
    overwriting whatever was stored for that node before — a node's item set only grows as
    more items get classified under it, so a "delta" (skip-if-already-summarized) cache would
    leave earlier summaries permanently stale as new items arrive.

    A failing node (any exception from :func:`summarize_node` — caught broadly, not just
    ``llm.LLMError``, since an oversized prompt can raise a bare ``OSError`` for the
    ``claude_cli`` provider; see :data:`_MAX_ITEMS_PER_NODE`), or a node that came back with
    nothing worth storing, is skipped — logged to stderr for the failure case — rather than
    discarding every other node's summary in this run.

    Returns the newly written ``{path: NodeSummary}`` map (``{}`` if nothing to summarize).
    """
    nodes = _nodes_from_items(store.query())
    summaries: dict[CategoryPath, NodeSummary] = {}
    for path, node_items in nodes.items():
        try:
            summary = summarize_node(path, node_items)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            print(f"summarizer: skipping {LEVEL_SEPARATOR.join(path)}: {exc}", file=sys.stderr)
            continue
        if summary is not None:
            summaries[path] = summary
            store.set_state(_node_key(path), summary.to_json())
    return summaries


def get_node_summary(store: Store, path: CategoryPath) -> NodeSummary | None:
    """The stored summary for `path`, or ``None`` if none has been computed yet."""
    raw = store.get_state(_node_key(path))
    return NodeSummary.from_json(path, raw) if raw is not None else None
