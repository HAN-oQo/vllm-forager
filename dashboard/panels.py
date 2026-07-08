"""Dashboard monitoring panels (T5.2): the per-stage "is it working + what did it decide"
summaries -- collection stats, taxonomy timeline, prediction scoreboard, candidate queue.

Every function here takes `store` explicitly and returns a plain, JSON-serializable shape,
matching :mod:`dashboard.api`'s own convention -- this module sits on top of it rather than
duplicating it: :func:`candidate_queue` is a thin re-export of
:func:`~dashboard.api.candidates`, not a second implementation, so the two can't
independently drift on that function's safety-critical `compute=False` default.

All four panels here are cheap (no LLM calls): :func:`collection_stats` and
:func:`taxonomy_timeline` are store scans / small state-map reads, :func:`prediction_scoreboard`
reads only already-recorded grades (never triggers new grading), and :func:`candidate_queue`
inherits :func:`~dashboard.api.candidates`'s own opt-in `compute=True` gate for the one
genuinely expensive domain.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from src import config, taxonomy
from src.agents import curator
from src.agents.forecaster import parse_ts
from src.agents.grader import GradeMetrics, compute_metrics, list_grades
from src.store.base import Store
from src.trends import week_stamp

from . import api


def collection_stats(store: Store) -> dict:
    """Per-repo item counts (``total``/``issue``/``pr``/``rocm``) plus a grand ``total`` --
    mirrors :func:`src.stats.summarize`'s own shape, but built from ``store.query()`` rather
    than a raw ``data/*.jsonl`` read, so it works against either backend
    (:class:`~src.store.jsonl_store.JsonlStore` or a future Firestore store) the same way
    every other panel in this module does, instead of assuming the on-disk default the way
    :mod:`src.stats` does.

    ``rocm`` is the same case-insensitive label-substring heuristic
    :func:`~src.stats.summarize` already uses (:data:`~src.config.ROCM_HINTS`) -- a heuristic,
    not a real classification; see that module's own docstring for the same caveat. Unlike
    :func:`~src.stats.summarize`'s ``Counter`` (whose keys vary with whatever ``type`` values
    happen to be present), each repo here always has the same four fixed keys, so a caller
    doesn't need to guard against a missing key for a repo that happens to have no PRs yet.
    """
    repos: dict[str, dict] = {}
    total = 0
    for item in store.query():
        repo = item.get("repo")
        if not repo:
            continue
        counts = repos.setdefault(repo, {"total": 0, "issue": 0, "pr": 0, "rocm": 0})
        counts["total"] += 1
        item_type = item.get("type")
        if item_type in ("issue", "pr"):
            counts[item_type] += 1
        labels = " ".join(str(x).lower() for x in (item.get("labels") or []))
        if any(hint in labels for hint in config.ROCM_HINTS):
            counts["rocm"] += 1
        total += 1
    return {"repos": repos, "total": total}


def taxonomy_timeline(store: Store, *, now: datetime | None = None) -> list[dict]:
    """One entry per category in the active taxonomy: ``{"category": label,
    "added_in_version": N, "retirement": {"weeks_inactive": float | None, "evidence": str |
    None} | None}``.

    Returns ``[]`` if no taxonomy has been created yet, rather than raising -- a fresh KB with
    no taxonomy yet is a normal state for a read endpoint to degrade past, matching
    :func:`~dashboard.api.parity`'s own established convention for a config/data gap.

    Known limitations, not fixed here:

    - ``added_in_version`` is the closest thing to "when" a category was added a taxonomy
      version can report -- :class:`~src.taxonomy.Taxonomy` carries no timestamp field (see
      its own module docstring), so no real added-at date exists yet to surface. A real fix
      needs :func:`~src.taxonomy.add_category` itself to start recording one; out of scope for
      this read-only panel.
    - ``retirement`` reflects only a currently *proposed* retirement
      (:func:`~src.agents.curator.propose_retirements`), since nothing in this codebase
      actually removes a category from the taxonomy yet (``curator.py``'s own documented gap)
      -- a category can be flagged inactive here without ever truly leaving the active
      taxonomy. ``weeks_inactive`` is ``None`` for a category with no recorded activity at all
      (:func:`~src.agents.curator.weeks_since_active` returns ``math.inf`` for that case,
      which isn't valid JSON -- ``None`` is the honest, serializable equivalent of "never").
    """
    try:
        active = taxonomy.get_active(store)
    except taxonomy.TaxonomyError:
        return []
    added_in_version: dict[str, int] = {}
    for version in range(1, active.version + 1):
        for label in taxonomy.get_taxonomy(store, version).labels:
            added_in_version.setdefault(label, version)
    retirements = {p.category: p for p in curator.propose_retirements(store, now=now)}
    timeline = []
    for label in active.labels:
        proposal = retirements.get(label)
        retirement = None
        if proposal is not None:
            weeks_inactive = proposal.weeks_inactive
            retirement = {
                "weeks_inactive": weeks_inactive if weeks_inactive != float("inf") else None,
                "evidence": proposal.evidence,
            }
        timeline.append(
            {
                "category": label,
                "added_in_version": added_in_version[label],
                "retirement": retirement,
            }
        )
    return timeline


def _metrics_dict(metrics: GradeMetrics) -> dict:
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "brier": metrics.brier,
        "n": metrics.n,
    }


def prediction_scoreboard(store: Store) -> dict:
    """Precision/recall/Brier per ISO week the grade was recorded (``graded_at``) -- the
    calibration-over-time view this todo's own worked example ("a scoreboard shows prediction
    precision over time") asks for: ``{week: {"precision":.., "recall":.., "brier":..,
    "n":..}}``.

    Cheap: only reads already-recorded grades (:func:`~src.agents.grader.list_grades`) -- no
    LLM call, unlike grading a *new* prediction
    (:func:`~src.agents.grader.grade_store`), which this function never triggers.

    Each week's metrics are computed from *only that week's own* grades (mirrors
    :func:`~src.trends.category_trends`'s own per-week, not cumulative, bucketing) -- a week
    with zero grades is simply absent from the result, not present with all-zero metrics, so a
    consumer can distinguish "nothing matured this week" from "predictions matured and scored
    badly".
    """
    by_week: dict[str, list] = defaultdict(list)
    for grade in list_grades(store):
        by_week[week_stamp(parse_ts(grade.graded_at))].append(grade)
    return {week: _metrics_dict(compute_metrics(grades)) for week, grades in by_week.items()}


def candidate_queue(
    store: Store, *, compute: bool = False, per_domain_limit: int | None = None
) -> list[dict]:
    """Thin re-export of :func:`~dashboard.api.candidates` -- see that function's own
    docstring for the `compute=False` safety default and the ``[]`` "never computed" vs.
    "genuinely empty" ambiguity it discloses. Not reimplemented here so the two can't
    independently drift on that same safety-critical default.
    """
    return api.candidates(store, compute=compute, per_domain_limit=per_domain_limit)
