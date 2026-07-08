"""Live health panel (T5.8): reads T4.5's heartbeat events into a "what's running now" view --
per stage, its status (running / stalled / ok / failed / never run), current step, a tail of
intermediate output, elapsed time since its last event, and a top-level green/red badge.

Scoped to the same 3 real, liveness-tracked orchestrator stages :mod:`dashboard.pipeline_diagram`
(T5.5) already names -- imports :data:`~dashboard.pipeline_diagram.PLANES` rather than defining
a third, independent copy of "collect"/"intel"/"contribution" (T5.5's own review already found
two hand-maintained copies of this exact list a drift risk; a third would repeat it). DEVPLAN's
own "e.g." for this todo names finer-grained per-agent nodes ("classifier", "engineer") that
aren't independently liveness-tracked yet -- the same disclosed gap T5.5's own review found;
illustrative, not a literal contract, matching how T5.5's own "e.g." named individual agents its
actual scope didn't cover either.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.liveness import latest_status
from src.orchestrator import _parse_last_run
from src.stages import latest_run
from src.store.base import Store

from .pipeline_diagram import PLANES

_UNHEALTHY_STATUSES = {"stalled", "failed"}


def stage_health(
    store: Store, stage: str, *, now: datetime | None = None, stale_after_s: float
) -> dict:
    """``{"stage", "status", "step", "output_tail", "elapsed_s", "error"}`` for `stage`'s own
    run history. `status` is :func:`~src.liveness.latest_status`'s own classification;
    `step`/`output_tail` are only ever present on a ``"heartbeat"`` record (`None` otherwise --
    "current step" has no meaning once a stage is terminal); `error` is only ever present on a
    ``"failed"`` record. `elapsed_s` is real seconds between `now` and the latest event's own
    `recorded_at`, or `None` if `stage` has never run.

    **Known limitation, not fixed here:** `step`/`output_tail`/`elapsed_s`/`error` are read off
    the single most-recently-recorded event (:func:`~src.stages.latest_run`), independently of
    which exact record `status`'s own :func:`~src.liveness.latest_status` picked -- the two can
    disagree only in the rare case of two events for the same stage sharing an identical
    second-resolution timestamp where one is terminal and the other isn't (`latest_status`
    breaks that tie toward the terminal record via its own private rule, not re-derived here).
    Untested by this todo's own Test line, which doesn't exercise a same-second tie.
    """
    when = now or datetime.now(timezone.utc)
    runs = store.list_runs(stage=stage)
    status = latest_status(runs, now=when, stale_after_s=stale_after_s)
    latest = latest_run(runs)
    if latest is None:
        return {
            "stage": stage,
            "status": status,
            "step": None,
            "output_tail": None,
            "elapsed_s": None,
            "error": None,
        }
    recorded = _parse_last_run(latest.get("recorded_at") or "")
    return {
        "stage": stage,
        "status": status,
        "step": latest.get("step"),
        "output_tail": latest.get("output_tail"),
        "elapsed_s": (when - recorded).total_seconds() if recorded else None,
        "error": latest.get("error"),
    }


def health_panel(store: Store, *, now: datetime | None = None, stale_after_s: float) -> dict:
    """``{"stages": [stage_health(...) per real orchestrator stage], "overall": "green" |
    "red"}`` -- the top-level badge this todo's own "Why" names ("a stall must show red,
    never a frozen 'running'"): ``"red"`` if any stage is ``stalled``/``failed``, ``"green"``
    otherwise (``running``/``ok``/``"never run"`` all read as "nothing wrong right now").
    """
    stages = [
        stage_health(store, plane["id"], now=now, stale_after_s=stale_after_s) for plane in PLANES
    ]
    overall = "red" if any(s["status"] in _UNHEALTHY_STATUSES for s in stages) else "green"
    return {"stages": stages, "overall": overall}
