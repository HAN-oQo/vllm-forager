"""Curator agent (T2.3): propose new / retire dead taxonomy categories from activity.

The taxonomy tracks a moving field — new techniques appear, old ones die — so a vocabulary
fixed at T1.2's ``create_taxonomy()`` (or any single ``add_category()`` since) goes stale
without something watching activity and proposing changes. This module watches two signals:

- **Retirement** (:func:`propose_retirements`): a category in the *active* taxonomy with no
  recorded activity (:mod:`src.trends`) in the last `inactive_weeks` weeks — including a
  category that's *never* had any — is proposed for retirement, evidenced by whichever item
  was that category's last activity (evidence principle — see :class:`RetireProposal`).
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
themes exist, not fitting a known number. Order-dependent on :meth:`Store.query`'s own
iteration order specifically (which item a caller's backend happens to yield first becomes a
cluster's representative, and everything else's membership follows from similarity to that
one item) — not just an abstract "item order," so a store migration or re-collection can
change which items cluster together with no underlying signal having changed. O(items ×
clusters) rather than O(items²), which is fine at this project's scale (M1's own report/
dashboard modules make the same brute-force-is-fine call); revisit if the ``Other`` bucket
ever grows large enough for either cost to matter.

Known limitations, not fixed here:
- Neither proposal function tracks what it already surfaced — every call recomputes from
  scratch, so an unchanged KB re-proposes the exact same retirements/clusters every run, and a
  human's "reviewed, rejected" decision has nowhere to be recorded. Deliberate for this first
  M2 cut (see "proposals only" above); persistence (mirroring :mod:`~src.agents.grader`'s
  append-only grade log) is a reasonable follow-up once something actually consumes these
  proposals on a schedule.
- :func:`propose_retirements` (via :mod:`src.trends`) and :func:`propose_new_categories` each
  independently call :meth:`Store.query` — two full store scans per "run the curator" cycle
  instead of one shared read. No caller invokes both today to justify threading a pre-fetched
  item list through (unlike T2.2's `src/grade.py`, which genuinely needed the same grade list
  twice in one process) — worth revisiting once a real orchestrating caller exists.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import embed, llm
from .. import taxonomy as taxonomy_module
from ..store.base import Store
from ..trends import category_trends
from .reporter import OTHER, evidence_url

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

# Items shown in one naming prompt. Mirrors summarizer.py's _MAX_ITEMS_PER_NODE / reporter_v1
# .py's _CHUNK_SIZE (same value, same reason): an unbounded prompt risks the claude_cli
# provider's argv-length OSError that summarizer.py's own history already hit once. Bounds
# only what the model *sees* when naming — evidence/size below still reflect the full cluster.
_MAX_ITEMS_PER_NAMING_PROMPT = 25

_NAME_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}


@dataclass(frozen=True)
class RetireProposal:
    """A category with no recent activity, proposed for retirement.

    ``evidence`` cites the category's own last-active item (the evidence principle: a claim
    of inactivity should point at *something* concrete a human can check) — `None` for a
    category that's never had any activity at all, since there's nothing to cite.
    """

    category: str
    weeks_inactive: float  # float so "never active" can be represented as math.inf
    evidence: str | None


@dataclass(frozen=True)
class NewCategoryProposal:
    """A cluster of `Other` items coherent enough to warrant a new taxonomy category.

    ``size`` is the full cluster's size; ``evidence`` may be shorter than `size` if some
    cluster items had nothing :func:`~src.agents.reporter.evidence_url` could resolve — the
    two aren't guaranteed equal.
    """

    name: str
    evidence: tuple[str, ...]
    size: int


def _week_start(week_stamp: str) -> datetime:
    """The Monday (00:00 UTC) of the ISO week `week_stamp` (e.g. ``"2026-W23"``) names."""
    return datetime.strptime(f"{week_stamp}-1", "%G-W%V-%u").replace(tzinfo=timezone.utc)


def weeks_since_active(category: str, series: dict[str, dict[str, int]], *, now: datetime) -> float:
    """Weeks between `now` and `category`'s most recent activity in `series`.

    ``math.inf`` if `category` has no recorded activity at all, or if its most recent week
    isn't a parseable ISO week-stamp (a corrupt/hand-edited state record) — treated as "never
    active" rather than raising, so one bad category's data doesn't abort every other
    category's retirement check (mirrors :mod:`src.trends`'s own "skip malformed, don't crash"
    handling of the same kind of data).

    Measured from the *start* (Monday 00:00 UTC) of the most recent active week, not the
    precise timestamp of the last item in it — up to ~6 days of systematic overstatement for
    an item created later in its week. `series` is week-bucketed by design (:mod:`src.trends`),
    so this is the granularity available; immaterial at the default 8-week threshold.
    """
    weeks = series.get(category)
    if not weeks:
        return float("inf")
    last_active = max(weeks)  # ISO "YYYY-Www" strings sort chronologically as plain strings
    try:
        start = _week_start(last_active)
    except ValueError:
        return float("inf")
    return (now - start).days / 7


def _last_active_item(category: str, items: list[dict]) -> dict | None:
    """The most recently created item classified `category`, or `None` if there are none."""
    candidates = [item for item in items if item.get("category") == category]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.get("created_at") or "")


def propose_retirements(
    store: Store, *, inactive_weeks: int = _DEFAULT_INACTIVE_WEEKS, now: datetime | None = None
) -> list[RetireProposal]:
    """Every active-taxonomy category with no activity in the last `inactive_weeks` weeks.

    Reads the *flat* label per category (:attr:`~src.taxonomy.Taxonomy.labels`) since that's
    what :mod:`src.trends` currently buckets by (T1.5.2's known limitation: trends doesn't yet
    roll a multi-level path up by prefix) — a category several levels deep is checked by its
    own full flattened label, not aggregated with its siblings/parent.

    Raises:
        taxonomy.TaxonomyError: no taxonomy has ever been created — mirrors
            :func:`~src.agents.analyst.analyze_store`'s own behavior when there's real work to
            check and no taxonomy to check it against (that function only short-circuits when
            there's *nothing pending*, not when a taxonomy is missing) — a bootstrapping gap
            to notice and fix, not a "nothing to propose" state to silently return past.
    """
    when = now or datetime.now(timezone.utc)
    active = taxonomy_module.get_active(store)  # fail fast before the full-store scan below
    items = store.query()
    series = category_trends(items)
    proposals = []
    for category in active.labels:
        inactive_for = weeks_since_active(category, series, now=when)
        if inactive_for >= inactive_weeks:
            last_item = _last_active_item(category, items)
            evidence = evidence_url(last_item) or None if last_item else None
            proposals.append(
                RetireProposal(category=category, weeks_inactive=inactive_for, evidence=evidence)
            )
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
    Only the first :data:`_MAX_ITEMS_PER_NAMING_PROMPT` items are shown to the model (bounds
    prompt size); the caller's own `evidence`/`size` still reflect the full cluster.
    """
    shown = items[:_MAX_ITEMS_PER_NAMING_PROMPT]
    # Collapsed whitespace (not raw titles) — a stray embedded newline could otherwise make
    # one item look like two lines/two items, matching reporter._cite's own established
    # "internal whitespace collapsed so it can't inject structure" treatment of the same risk.
    listing = "\n".join(f"- {' '.join((item.get('title') or '').split())}" for item in shown)
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

    A cluster whose naming call fails (``llm.LLMError``) is skipped and logged to stderr —
    left out of the returned list — rather than losing every other cluster's already-computed
    proposal in the same call, matching :func:`~src.agents.analyst.analyze_store`'s,
    :func:`~src.agents.forecaster.forecast_store`'s, and :func:`~src.agents.grader.
    grade_store`'s own per-item failure isolation.

    Raises:
        embed.EmbedError: the embedding call failed (e.g. ``EMBED_PROVIDER=local`` pointed at
            an unreachable/misconfigured server) — not caught, unlike the per-cluster naming
            failure above: embedding is one batch call for every `Other` item at once, a
            foundational precondition for clustering at all, not a per-cluster failure there's
            anything meaningful to isolate.
    """
    other_items = [item for item in store.query() if item.get("category") == OTHER]
    proposals = []
    for cluster in _cluster_items(other_items, similarity_threshold=similarity_threshold):
        if len(cluster) < min_cluster_size:
            continue
        try:
            name = _name_cluster(cluster)
        except llm.LLMError as exc:
            print(f"curator: skipping a cluster of {len(cluster)} item(s): {exc}", file=sys.stderr)
            continue
        if name is None:
            continue
        evidence = tuple(url for item in cluster if (url := evidence_url(item)))
        proposals.append(NewCategoryProposal(name=name, evidence=evidence, size=len(cluster)))
    return proposals
