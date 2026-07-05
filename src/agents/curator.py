"""Curator agent (T2.3): propose new / retire dead taxonomy categories from activity.

The taxonomy tracks a moving field — new techniques appear, old ones die — so a vocabulary
fixed at T1.2's ``create_taxonomy()`` (or any single ``add_category()`` since) goes stale
without something watching activity and proposing changes. This module watches two signals:

- **Retirement** (:func:`propose_retirements`): a category in the *active* taxonomy with no
  recorded activity (:mod:`src.trends`) in the last `inactive_weeks` weeks — including a
  category that's *never* had any — is proposed for retirement.
- **New categories** (:func:`propose_new_categories`): items classified :data:`~src.agents.
  reporter.OTHER` (the Analyst's own "doesn't fit anywhere" fallback) are exactly the ones a
  new category would help — a taxonomy that fit everything wouldn't have any. They're
  clustered by embedding similarity (:mod:`src.embed`), and any cluster at or above
  `min_cluster_size` is named by ``llm.complete`` and proposed as a new category.

Both are **proposals only** — a queue of recommendations for a human (or a later,
not-yet-built apply step) to review, mirroring M2's other outer-loop stages (T2.5's candidate
queue is the same shape: discovery/ranking, not automatic action). Nothing here calls
:func:`~src.taxonomy.add_category` or mutates the taxonomy — "retire" has no existing removal
primitive in :mod:`src.taxonomy` at all (append-only by design), and minting that primitive is
a taxonomy-versioning design decision this module shouldn't make as a side effect of proposing
one candidate retirement.

Clustering is deliberately simple: single-pass greedy assignment (each item joins the first
existing cluster whose *first* item it's similar enough to, by cosine similarity over
:func:`~src.embed.embed_texts`; otherwise it starts a new cluster) — not k-means or anything
requiring a predetermined cluster count, since the whole point is discovering how many novel
themes exist, not fitting a known number. Order-dependent (a different item order can produce
different clusters) and O(items × clusters) rather than O(items²), which is fine at this
project's scale (M1's own report/dashboard modules make the same brute-force-is-fine call);
revisit if the ``Other`` bucket ever grows large enough for that to matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .. import embed, llm
from .. import taxonomy as taxonomy_module
from ..store.base import Store
from ..trends import trends_from_store
from .reporter import OTHER

# A category with its most recent activity this many weeks (or more) in the past — or with no
# recorded activity at all — is proposed for retirement. 8 weeks (~2 months) is long enough
# that a real slowdown, not just an unlucky quiet fortnight, is the more likely explanation.
_DEFAULT_INACTIVE_WEEKS = 8

# A cluster of `Other` items smaller than this is more likely sampling noise (a handful of
# genuinely unrelated stragglers that happen to be somewhat similar) than a real emerging
# theme worth a whole new taxonomy category.
_DEFAULT_MIN_CLUSTER_SIZE = 5

# How similar two items' embeddings must be (cosine) to join the same cluster — high enough
# that the "hash" embedding provider's coarse lexical-overlap signal (the offline-safe
# default; see src/embed.py) still produces meaningfully coherent clusters, not everything
# lumped into one bucket.
_DEFAULT_SIMILARITY_THRESHOLD = 0.5

_NAME_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}


@dataclass(frozen=True)
class RetireProposal:
    """A category with no recent activity, proposed for retirement."""

    category: str
    weeks_inactive: float  # float so "never active" can be represented as math.inf


@dataclass(frozen=True)
class NewCategoryProposal:
    """A cluster of `Other` items coherent enough to warrant a new taxonomy category."""

    name: str
    evidence: tuple[str, ...]
    size: int


def _week_start(week_stamp: str) -> datetime:
    """The Monday (00:00 UTC) of the ISO week `week_stamp` (e.g. ``"2026-W23"``) names."""
    return datetime.strptime(f"{week_stamp}-1", "%G-W%V-%u").replace(tzinfo=timezone.utc)


def weeks_since_active(category: str, series: dict[str, dict[str, int]], *, now: datetime) -> float:
    """Weeks between `now` and `category`'s most recent activity in `series`.

    ``math.inf`` if `category` has no recorded activity at all — "never active" is more
    inactive than any finite week count, not a zero/undefined value to special-case at every
    call site.
    """
    weeks = series.get(category)
    if not weeks:
        return float("inf")
    last_active = max(weeks)  # ISO "YYYY-Www" strings sort chronologically as plain strings
    return (now - _week_start(last_active)).days / 7


def propose_retirements(
    store: Store, *, inactive_weeks: int = _DEFAULT_INACTIVE_WEEKS, now: datetime | None = None
) -> list[RetireProposal]:
    """Every active-taxonomy category with no activity in the last `inactive_weeks` weeks.

    Reads the *flat* label per category (:attr:`~src.taxonomy.Taxonomy.labels`) since that's
    what :mod:`src.trends` currently buckets by (T1.5.2's known limitation: trends doesn't yet
    roll a multi-level path up by prefix) — a category several levels deep is checked by its
    own full flattened label, not aggregated with its siblings/parent.
    """
    when = now or datetime.now(timezone.utc)
    series = trends_from_store(store)
    active = taxonomy_module.get_active(store)
    proposals = []
    for category in active.labels:
        inactive_for = weeks_since_active(category, series, now=when)
        if inactive_for >= inactive_weeks:
            proposals.append(RetireProposal(category=category, weeks_inactive=inactive_for))
    return proposals


def _cluster_items(
    items: list[dict], *, similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
) -> list[list[dict]]:
    """Greedy single-pass clustering by embedding similarity — see module docstring."""
    if not items:
        return []
    texts = [f"{item.get('title') or ''} {(item.get('body') or '')[:500]}" for item in items]
    vectors = embed.embed_texts(texts)
    clusters: list[list[dict]] = []
    representatives: list[list[float]] = []
    for item, vector in zip(items, vectors, strict=True):
        for cluster, rep in zip(clusters, representatives, strict=True):
            if embed.cosine_similarity(vector, rep) >= similarity_threshold:
                cluster.append(item)
                break
        else:
            clusters.append([item])
            representatives.append(vector)
    return clusters


def _name_cluster(items: list[dict]) -> str | None:
    """Ask the model for a short taxonomy-style category name covering `items`' common theme.

    `None` if the model's reply has no usable ``name`` — a cluster that can't be named isn't
    proposed (see module docstring: a proposal a human can't act on isn't worth surfacing).
    """
    listing = "\n".join(f"- {item.get('title') or ''}" for item in items)
    prompt = (
        "These GitHub issues/PRs didn't fit any existing taxonomy category. Propose ONE short "
        "category name (2-4 words, matching the style of labels like 'quantization' or "
        "'attention > flash-attention') that captures their common theme. Reply with `name`.\n\n"
        f"Items:\n{listing}"
    )
    reply = llm.complete(prompt, json_schema=_NAME_SCHEMA)
    name = reply.get("name") if isinstance(reply, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else None


def propose_new_categories(
    store: Store,
    *,
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
) -> list[NewCategoryProposal]:
    """Cluster every :data:`~src.agents.reporter.OTHER`-classified item; propose a new category
    for each cluster at or above `min_cluster_size`, named by ``llm.complete``.

    Raises:
        llm.LLMError: a naming call failed (transport error, timeout, non-JSON reply) — not
            caught here, since (unlike T1.4/T1.5's per-item classification loops) a handful of
            clusters failing to name shouldn't silently produce a partial, hard-to-audit
            proposal list; a caller wanting per-cluster failure isolation can catch this
            around individual :func:`_name_cluster` calls itself.
    """
    other_items = [item for item in store.query() if item.get("category") == OTHER]
    proposals = []
    for cluster in _cluster_items(other_items, similarity_threshold=similarity_threshold):
        if len(cluster) < min_cluster_size:
            continue
        name = _name_cluster(cluster)
        if name is None:
            continue
        evidence = tuple(item["url"] for item in cluster if item.get("url"))
        proposals.append(NewCategoryProposal(name=name, evidence=evidence, size=len(cluster)))
    return proposals
