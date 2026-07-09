"""Attempts tab (T5.11): browse every worked candidate's T3.12 attempt report -- outcome
badge, issue overview, approach, and reproduce steps -- newest first, plus a rendered
title/body/diff/evidence view (and the T5.6 review bundle needed to approve it) for a
verified, gate-ready candidate.

A thin wrapper composing over :mod:`src.attempt_report` (the report itself, and -- per T5.9's
own "scanning belongs in `src`, the dashboard module is thin glue" correction -- the worked-
candidate listing) and :mod:`dashboard.review` (T5.6) -- not a second implementation. Reports
are computed live from the KB via :func:`~src.attempt_report.render_attempt_report`, the same
"the store is the source of truth, not a pre-written file on disk" pattern
:mod:`dashboard.review`'s own `review_bundle` already established, rather than requiring
``python -m src.attempt_report`` to have already been run and its `.md`/`.html` files to exist
on disk for a candidate to show up here.

Per T5.11's own DEVPLAN note ("with the T5.6 patch-review approve action wired to it, so a
human can go straight from 'browsing attempts' to 'approving this one' without leaving the
console"): :func:`open_attempt` folds :func:`~dashboard.review.review_bundle` straight into
its own return value for a verified, gate-ready candidate (`rendered_html is not None`), so a
caller already has everything :func:`~dashboard.review.review_decision` needs to approve/hold
without a second lookup into a different tab's module. No `dashboard/server.py` HTTP route
calls this module yet -- wiring an actual approve *button* to it is T5.12's job (the tabbed
console shell), matching T5.1-T5.10's own "data layer first" precedent.
"""

from __future__ import annotations

from dataclasses import asdict

from src import attempt_report
from src.store.base import Store

from . import review


def list_attempts(store: Store) -> list[dict]:
    """Every worked candidate, newest first -- see :func:`~src.attempt_report.list_worked`
    for the exact shape (`repo`/`number`/`verified`/`title`/`recorded_at`)."""
    return attempt_report.list_worked(store)


def open_attempt(store: Store, repo: str, number: int) -> dict | None:
    """The full attempt report for (`repo`, `number`) -- issue overview, approach, reproduce
    steps, and outcome -- or `None` if it hasn't been worked yet (no `stage="verify"` run;
    mirrors :func:`~src.attempt_report.render_attempt_report`'s own contract).

    A verified, gate-ready candidate's result also carries a `"review_bundle"` key (see
    module docstring) -- `None` for every other candidate, including a verified-but-not-yet-
    gate-ready one.
    """
    report = attempt_report.render_attempt_report(store, repo, number)
    if report is None:
        return None
    result = asdict(report)
    result["review_bundle"] = (
        review.review_bundle(store, repo, number) if report.rendered_html is not None else None
    )
    return result
