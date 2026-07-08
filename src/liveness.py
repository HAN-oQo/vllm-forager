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

from .orchestrator import _parse_last_run

_LIFECYCLE_STATUSES = {"started", "heartbeat"}
_TERMINAL_STATUSES = {"ok", "failed"}

__all__ = ["latest_status"]


def latest_status(runs: list[dict], *, now: datetime, stale_after_s: float) -> str:
    """Classify a stage's liveness from its own `runs` (already filtered to one stage, e.g. via
    `store.list_runs(stage=...)` — this function doesn't filter by stage itself, so passing an
    unfiltered list mixes stages' histories into one classification).

    Ties among same-`recorded_at` records break toward the terminal state -- `run_tick` records
    a stage's terminal event strictly after its own last heartbeat (if any), but this handles
    equal-resolution timestamps landing on the same second defensively rather than by luck of
    dict/list ordering.
    """
    if not runs:
        return "never run"

    def _rank(run: dict) -> tuple[str, int]:
        # (recorded_at, terminal-wins-a-tie) -- max() picks the last-sorting tuple, and a
        # terminal status should win over a lifecycle one at an identical timestamp.
        is_terminal = run.get("status") in _TERMINAL_STATUSES
        return (run.get("recorded_at") or "", 1 if is_terminal else 0)

    latest = max(runs, key=_rank)
    status = latest.get("status")
    if status in _TERMINAL_STATUSES:
        return status
    if status not in _LIFECYCLE_STATUSES:
        return "stalled"  # an unrecognized status -- fail toward "needs attention", not "fine"

    recorded = _parse_last_run(latest.get("recorded_at") or "")
    if recorded is None:
        return "stalled"  # unparseable timestamp -- can't confirm freshness, so don't assume it
    age_s = (now - recorded).total_seconds()
    return "stalled" if age_s > stale_after_s else "running"
