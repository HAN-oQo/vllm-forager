"""Liveness classification (T4.5): tell "still alive" apart from "silently stalled" using a
stage's own run-event history — the thing T4.4's plain ``"ok"``/``"failed"`` terminal events
alone can't do, since a crash before either one is written looks identical, from the KB's own
run history, to a stage that's simply still working.

Read-only over :meth:`~src.store.base.Store.list_runs`'s own records — this module writes
nothing itself. The write side (recording ``"started"``/``"heartbeat"``/terminal events) lives
in :mod:`src.orchestrator` (:func:`~src.orchestrator.run_tick`, :func:`~src.orchestrator.
emit_heartbeat`); this module only classifies what's already there. Powers the T5.8 health
panel's "running / idle / stalled / failed" badge.

Status vocabulary, in a stage's own lifecycle order: ``"started"`` -> zero or more
``"heartbeat"``\\ s (current step + a rolling tail of intermediate output) -> a terminal
``"ok"``/``"failed"`` event (T4.4's own vocabulary; T4.5 only adds the two lifecycle states
*before* it, not new terminal ones). :func:`latest_status` classifies the most recent event for
a stage:

- ``"ok"`` / ``"failed"`` — the latest event already is one of T4.4's terminal states.
- ``"running"`` — the latest event is ``"started"``/``"heartbeat"`` and recent (within
  `stale_after_s` of `now`).
- ``"stalled"`` — the latest event is ``"started"``/``"heartbeat"`` but *older* than
  `stale_after_s` — DEVPLAN's own words for this todo: "a crash must never look like 'still
  running'."
- ``"never run"`` — no run events at all for this stage.
"""

from __future__ import annotations

from datetime import datetime

from .orchestrator import parse_last_run
from .stages import latest_run

_LIFECYCLE_STATUSES = {"started", "heartbeat"}
_TERMINAL_STATUSES = {"ok", "failed"}

__all__ = ["latest_status", "latest_status_and_record"]


def latest_status_and_record(
    runs: list[dict], *, now: datetime, stale_after_s: float
) -> tuple[str, dict | None]:
    """:func:`latest_status`'s own classification, plus the exact run record it was based on --
    for a caller (T5.8's health panel) that also needs fields off that same record (its
    `step`/`output_tail`/`error`), not just the status string. Extracted here (a code-review
    finding) after an earlier version of the dashboard health panel independently re-derived
    "the latest run" a second way (`stages.latest_run` alone, with no tie-break) to read those
    extra fields — a same-second tie between a terminal and a non-terminal record could then
    make the reported status and the reported step/error come from two *different* records,
    since `stages.latest_run`'s own tie-break (first-encountered wins) doesn't know this
    module's own "ties break toward the terminal state" rule. Returning the record this
    function actually picked closes that gap for any caller that needs more than the status.

    Reuses `stages.latest_run`'s own "find the most-recently-recorded run, by `recorded_at`"
    convention (already shared by `engineer.py`/`self_review.py`/`gate.py`/etc.) rather than
    re-deriving the same `max(..., key=recorded_at)` logic here, then layers T4.5's one extra
    rule on top: ties among same-`recorded_at` records break toward the terminal state, since
    `stages.latest_run`'s own `max()` (Python semantics: first-encountered element wins a tie)
    would otherwise pick whichever of a same-second terminal/lifecycle pair happens to sort
    first in `runs`' own order -- non-deterministic from this function's point of view. Real
    same-second ties are possible: every run-event timestamp in this codebase has only
    second-resolution (`"%Y-%m-%dT%H:%M:%SZ"`), so a fast stage's own "started" and terminal
    events can legitimately land in the same rendered second.
    """
    if not runs:
        return "never run", None

    latest = latest_run(runs)
    assert latest is not None  # `runs` is non-empty, so `latest_run` can't return None here
    tied = [r for r in runs if (r.get("recorded_at") or "") == (latest.get("recorded_at") or "")]
    terminal_tied = [r for r in tied if r.get("status") in _TERMINAL_STATUSES]
    if terminal_tied:
        latest = terminal_tied[0]

    status = latest.get("status")
    if status in _TERMINAL_STATUSES:
        return status, latest
    if status not in _LIFECYCLE_STATUSES:
        return "stalled", latest  # unrecognized status -- fail toward "needs attention"

    recorded = parse_last_run(latest.get("recorded_at") or "")
    if recorded is None:
        return "stalled", latest  # unparseable timestamp -- can't confirm freshness
    age_s = (now - recorded).total_seconds()
    return ("stalled" if age_s > stale_after_s else "running"), latest


def latest_status(runs: list[dict], *, now: datetime, stale_after_s: float) -> str:
    """Classify a stage's liveness from its own `runs` (already filtered to one stage, e.g. via
    `store.list_runs(stage=...)` — this function doesn't filter by stage itself, so passing an
    unfiltered list mixes stages' histories into one classification). Thin wrapper over
    :func:`latest_status_and_record` for a caller that only wants the status string.
    """
    return latest_status_and_record(runs, now=now, stale_after_s=stale_after_s)[0]
