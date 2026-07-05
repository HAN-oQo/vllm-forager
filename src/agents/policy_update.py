"""Policy update from grades (T2.2): propose ``policy@vN+1`` from T2.1's grading results.

Closes the self-evolution loop — grading (T2.1) is only a diagnostic unless its scores
actually reweight the policy (T1.3, :mod:`~src.policy`) that drives the next round of
classification/scoring. This module is that link: for every category with at least one
graded prediction, compute that category's precision (:func:`~src.agents.grader.
compute_metrics`), then blend it into the category's current ``scoring_weights`` entry via an
exponential moving average — a low-precision category's weight moves down toward that
precision, a high-precision one moves up — rather than replacing the weight outright, so one
noisy batch of grades can't swing a category's trust wildly in a single update.

A grade doesn't carry its own category — a :class:`~src.agents.forecaster.Prediction` is
about a GitHub item, not a taxonomy node. Attribution here follows the exact same evidence-URL
-to-item path :mod:`~src.agents.grader` already uses (:func:`~src.agents.grader.
parse_repo_number` + :meth:`Store.get_item`): a grade's category is whichever cited item's
``category`` field resolves first. A grade whose evidence doesn't resolve to any item, or
resolves to an item with no ``category`` yet, contributes to no category's precision and is
silently excluded — the same "only classified items count" rule :mod:`src.trends` already
applies to its own per-category aggregation, for the same reason (nothing meaningful to
attribute an uncategorized signal to).

Known limitation, not fixed here: a `Prediction` with evidence spanning multiple items in
*different* categories (unreachable today — :func:`~src.agents.forecaster.forecast_item` only
ever cites the one item it was made about) would be attributed to whichever cited item's
category resolves first, not split or double-counted — a design question for whenever a
multi-item forecast actually exists, not before.
"""

from __future__ import annotations

from collections import defaultdict

from .. import policy
from ..store.base import Store
from . import grader

# A category with no prior weight starts here — a neutral "as trusted as any other" prior,
# rather than 0.0 (which would make a brand-new category's very first weight equal its
# observed precision with no smoothing at all, defeating the point of the EMA blend below).
_DEFAULT_WEIGHT = 1.0

# How much a single update run moves a category's weight toward its freshly observed
# precision, vs. keeping its prior value: 0.5 = split evenly, so one batch of grades shifts
# the weight halfway to the new evidence rather than replacing it outright (a noisy small
# sample shouldn't be able to swing a category's trust to an extreme in one step).
_DEFAULT_LEARNING_RATE = 0.5


def grade_category(grade: grader.Grade, store: Store) -> str | None:
    """The taxonomy category a graded prediction's evidence belongs to, or `None` if it can't
    be resolved (no evidence URL parses, the cited item is gone, or it has no `category` yet).
    """
    for url in grade.prediction.evidence:
        parsed = grader.parse_repo_number(url)
        if parsed is None:
            continue
        item = store.get_item(*parsed)
        category = item.get("category") if item else None
        if category:
            return str(category)
    return None


def group_by_category(grades: list[grader.Grade], store: Store) -> dict[str, list[grader.Grade]]:
    """Bucket `grades` by :func:`grade_category`; grades that can't be attributed are dropped."""
    grouped: dict[str, list[grader.Grade]] = defaultdict(list)
    for grade in grades:
        category = grade_category(grade, store)
        if category is not None:
            grouped[category].append(grade)
    return dict(grouped)


def propose_scoring_weights(
    grades: list[grader.Grade],
    store: Store,
    current_weights: dict[str, float],
    *,
    learning_rate: float = _DEFAULT_LEARNING_RATE,
) -> dict[str, float]:
    """New ``scoring_weights`` entries for every category with ≥1 attributable grade in
    `grades` — an EMA blend of `current_weights` (or :data:`_DEFAULT_WEIGHT` for a category
    with no prior entry) toward that category's freshly observed precision.

    A category absent from `grades` entirely gets no entry here — :func:`~src.policy.
    update_policy` only overwrites the keys it's given, so an un-graded category's weight is
    simply left as-is by the caller, not reset to the default.
    """
    updated = {}
    for category, category_grades in group_by_category(grades, store).items():
        precision = grader.compute_metrics(category_grades).precision
        current = current_weights.get(category, _DEFAULT_WEIGHT)
        updated[category] = (1 - learning_rate) * current + learning_rate * precision
    return updated


def update_policy_from_grades(
    store: Store, *, learning_rate: float = _DEFAULT_LEARNING_RATE
) -> policy.Policy | None:
    """Propose and apply a new policy version from every recorded grade in `store`.

    Returns the newly-active :class:`~src.policy.Policy`, or `None` if no grade could be
    attributed to any category (nothing to update — the active policy is left untouched, no
    new version created).

    Raises:
        policy.PolicyError: no policy exists yet (``create_policy`` hasn't been called) — a
            loud failure, matching :func:`~src.agents.analyst.analyze_store`'s own "no
            taxonomy yet" behavior, since scoring a category against a policy that doesn't
            exist is a bootstrapping gap to fix, not a "nothing to do" state to silently skip.
    """
    current = policy.get_active(store)
    grades = grader.list_grades(store)
    new_weights = propose_scoring_weights(
        grades, store, dict(current.scoring_weights), learning_rate=learning_rate
    )
    if not new_weights:
        return None
    return policy.update_policy(store, scoring_weights=new_weights)
