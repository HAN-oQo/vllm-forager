"""Policy update from grades (T2.2): propose ``policy@vN+1`` from T2.1's grading results.

Closes the self-evolution loop — grading (T2.1) is only a diagnostic unless its scores
actually reweight the policy (T1.3, :mod:`~src.policy`) that drives the next round of
classification/scoring. This module is that link: for every category with at least one
*precision-defined* graded prediction (see below), report that category's observed precision
(:func:`~src.agents.grader.compute_metrics`) as its proposed ``scoring_weights`` entry, then
apply it only where it genuinely differs from the current weight.

Design note — no smoothing layer here, on purpose: precision is computed over *every*
recorded grade for a category, not just this run's newly-matured ones, so it's already a
stable statistic (more history narrows it, by the law of large numbers) — it doesn't need a
second exponential-moving-average blended on top of the *previous* weight. An earlier version
of this module did blend against the prior weight, and that was a real bug, not just
unnecessary caution: since precision is recomputed from the *same growing history* every run,
blending it against a weight that already reflects a prior run's blend of that same history
double-counts old evidence — a category's weight kept drifting long after its last genuinely
new grade, and never stabilized. Reporting raw precision directly is simpler and correct.

A grade doesn't carry its own category — a :class:`~src.agents.forecaster.Prediction` is
about a GitHub item, not a taxonomy node. Attribution follows the exact same evidence-URL
-to-item path :mod:`~src.agents.grader` already uses
(:func:`~src.agents.grader.resolve_evidence_items`): a grade's category is whichever cited
item's ``category`` field resolves first (and is actually a string — an item with a malformed
``category``, e.g. a list from an upstream classifier bug, is treated the same as having none,
mirroring :mod:`src.trends`'s own ``isinstance`` guard on the same field). A grade whose
evidence doesn't resolve to any item, or resolves to an item with no (valid) ``category`` yet,
is silently excluded — the same "only classified items count" rule :mod:`src.trends` already
applies to its own per-category aggregation, for the same reason.

A category with graded predictions but **none** predicted "true" (every ``prob`` below
:data:`~src.agents.grader.DEFAULT_THRESHOLD`) has *undefined* precision, not zero:
:func:`~src.agents.grader.compute_metrics` reports ``0.0`` by convention for that case (a
reasonable default for a print statement — see its own docstring), but ``0.0`` here would
wrongly punish a category whose predictions were simply always low-confidence, even if every
one of them correctly resolved false. Such a category is excluded from the proposal entirely,
leaving its weight unchanged, rather than reported as (a misleading) ``0.0``.

Known limitations, not fixed here:
- A `Prediction` with evidence spanning multiple items in *different* categories
  (unreachable today — :func:`~src.agents.forecaster.forecast_item` only ever cites the one
  item it was made about) would be attributed to whichever cited item's category resolves
  first, not split or double-counted — a design question for whenever a multi-item forecast
  actually exists, not before.
- A category name here is whatever an item's own ``category`` field says, with no
  cross-check against the *currently active* taxonomy version — an item classified under an
  older taxonomy version, or analyst.py's own ``Other`` fallback, can mint a ``scoring_weights``
  entry with no corresponding live taxonomy node. Reconciling category lifecycles against
  taxonomy evolution is T2.3's (Curator) job, not this module's.
- Like :mod:`~src.agents.grader`'s own already-documented cost profile, this module adds one
  more full walk of the grade log (:func:`update_policy_from_grades` — mitigated by accepting
  an already-computed `grades` list from a caller that has one, e.g. `src/grade.py`) plus one
  `Store.get_item` per grade for category attribution (mitigated *within* one call via
  `resolve_evidence_items`'s optional cache, but not shared across separate CLI-level calls
  such as `grader.grade_store`'s own item lookups moments earlier in the same process).
"""

from __future__ import annotations

import math
from collections import defaultdict

from .. import policy
from ..store.base import Store
from . import grader


def grade_category(
    grade: grader.Grade,
    store: Store,
    *,
    item_cache: dict[tuple[str, int], dict | None] | None = None,
) -> str | None:
    """The taxonomy category a graded prediction's evidence belongs to, or `None` if it can't
    be resolved (no evidence URL parses, the cited item is gone, or its `category` is missing
    or not a string).
    """
    for item in grader.resolve_evidence_items(grade.prediction, store, item_cache=item_cache):
        category = item.get("category")
        if category and isinstance(category, str):
            return category
    return None


def group_by_category(grades: list[grader.Grade], store: Store) -> dict[str, list[grader.Grade]]:
    """Bucket `grades` by :func:`grade_category`; grades that can't be attributed are dropped.

    Shares one `item_cache` across every grade in `grades`, so a repo whose items are cited by
    more than one grade (common as the grade log grows) is only read once per distinct item.
    """
    item_cache: dict[tuple[str, int], dict | None] = {}
    grouped: dict[str, list[grader.Grade]] = defaultdict(list)
    for grade in grades:
        category = grade_category(grade, store, item_cache=item_cache)
        if category is not None:
            grouped[category].append(grade)
    return dict(grouped)


def propose_scoring_weights(grades: list[grader.Grade], store: Store) -> dict[str, float]:
    """Each attributable category's raw observed precision — see the module docstring for why
    this isn't blended against the category's current weight.

    A category with grades but none predicted "true" (see module docstring) is excluded, not
    reported as ``0.0``. A category absent from `grades` entirely is also absent here —
    :func:`update_policy_from_grades` only applies the entries this function actually reports.
    """
    proposed = {}
    for category, category_grades in group_by_category(grades, store).items():
        if not any(g.prediction.prob >= grader.DEFAULT_THRESHOLD for g in category_grades):
            continue
        proposed[category] = grader.compute_metrics(category_grades).precision
    return proposed


def update_policy_from_grades(
    store: Store, grades: list[grader.Grade] | None = None
) -> policy.Policy | None:
    """Propose a new policy version from every recorded grade in `store` and apply it, but
    only if at least one category's proposed weight genuinely differs from its current one.

    `grades` lets a caller that already has the full grade list (e.g. `src/grade.py`, which
    needs it for its own metrics print) pass it straight through instead of this function
    re-walking the grade log a second time; defaults to :func:`~src.agents.grader.list_grades`.

    Returns the newly-active :class:`~src.policy.Policy`, or `None` if nothing would actually
    change (no attributable/precision-defined category, or every proposed weight already
    matches — a no-op re-run never bumps the policy version).

    Raises:
        policy.PolicyError: no policy exists yet (``create_policy`` hasn't been called) — a
            loud failure, matching :func:`~src.agents.analyst.analyze_store`'s own "no
            taxonomy yet" behavior, since scoring a category against a policy that doesn't
            exist is a bootstrapping gap to fix, not a "nothing to do" state to silently skip.
            Checked unconditionally (even when there'd otherwise be nothing to update), so a
            missing policy is caught on the very first run rather than only once some category
            happens to have gradable data.
    """
    current = policy.get_active(store)
    if grades is None:
        grades = grader.list_grades(store)
    proposed = propose_scoring_weights(grades, store)
    changed = {
        category: weight
        for category, weight in proposed.items()
        if category not in current.scoring_weights
        or not math.isclose(current.scoring_weights[category], weight, abs_tol=1e-9)
    }
    if not changed:
        return None
    return policy.update_policy(store, scoring_weights=changed)
