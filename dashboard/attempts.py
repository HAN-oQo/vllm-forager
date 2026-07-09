"""Attempts tab (T5.11): browse every worked candidate's T3.12 attempt report -- outcome
badge, issue overview, approach, and reproduce steps -- newest first, plus (opt-in) a rendered
title/body/diff/evidence view and the T5.6 review bundle needed to approve it, for a verified,
gate-ready candidate.

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
console"): :func:`open_attempt` can fold :func:`~dashboard.review.review_bundle` straight into
its own return value for a verified, gate-ready candidate, so a caller already has everything
:func:`~dashboard.review.review_decision` needs to approve/hold without a second lookup into a
different tab's module.

**`include_review_bundle` defaults to `False`** -- a code-review finding: `review_bundle` ->
`gate.assemble_bundle` -> `gate._risk_badge` pays one real `llm.complete` call, the exact cost
`render_attempt_report`'s own module docstring says it deliberately avoids ("Built from
`gate.verified_diff`, not `gate.assemble_bundle` -- the latter's risk-badge LLM call is pure
overhead here"). An earlier version of this function called `review_bundle` unconditionally
for every gate-ready candidate, so merely *browsing* the Attempts tab silently paid for an LLM
call on every open -- exactly the "scored on every page view" hazard T4.8-T4.11 exist to guard
against, and exactly the cost `dashboard.review`'s own docstring says it only pays because a
human has "explicitly opened to review right now." Mirrors `dashboard.api.candidates`'s own
`compute=False` convention: cheap by default, an explicit opt-in for the caller that actually
wants to review/approve. No `dashboard/server.py` HTTP route calls this module yet -- wiring an
actual approve *button* (and deciding when it passes `include_review_bundle=True`) is T5.12's
job (the tabbed console shell), matching T5.1-T5.10's own "data layer first" precedent.

**Known, disclosed limitation, not fixed here:** when `include_review_bundle=True`,
`render_attempt_report` (via `gate.verified_diff`) and `review_bundle` (via
`gate.assemble_bundle`) each independently re-derive overlapping gate-readiness evidence (diff,
self-review outcome, `verify_recorded_at`) for the same candidate from the same underlying
`Store.list_runs` data -- two separate assemblies of substantially the same evidence, doubling
the store-read cost of that one opt-in call. Left as-is: this only happens on an explicit,
human-initiated "show me the review bundle" action (bounded, not a page-view hazard), and a
real fix would mean changing `gate.assemble_bundle`'s own signature to accept already-computed
readiness data -- a `src.gate` change with a larger blast radius than this dashboard-layer
todo's own scope.
"""

from __future__ import annotations

from src.attempt_report import list_worked, render_attempt_report
from src.store.base import Store

from .api import asdict_with
from .review import review_bundle as _review_bundle


def list_attempts(store: Store) -> list[dict]:
    """Every worked candidate, newest first -- see :func:`~src.attempt_report.list_worked`
    for the exact shape (`repo`/`number`/`verified`/`title`/`recorded_at`)."""
    return list_worked(store)


def open_attempt(
    store: Store, repo: str, number: int, *, include_review_bundle: bool = False
) -> dict | None:
    """The full attempt report for (`repo`, `number`) -- issue overview, approach, reproduce
    steps, and outcome -- or `None` if it hasn't been worked yet (no `stage="verify"` run;
    mirrors :func:`~src.attempt_report.render_attempt_report`'s own contract).

    `include_review_bundle=True` also carries a `"review_bundle"` key (see module docstring)
    -- `None` if the candidate isn't verified/gate-ready. Left `False` (the default), the
    result never carries that key and pays no extra LLM cost -- see module docstring.
    """
    report = render_attempt_report(store, repo, number)
    if report is None:
        return None
    if not include_review_bundle:
        return asdict_with(report)
    return asdict_with(report, review_bundle=_review_bundle(store, repo, number))
