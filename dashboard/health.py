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

from collections import defaultdict
from datetime import datetime

from src.liveness import latest_status_and_record
from src.orchestrator import parse_last_run
from src.store.base import Store

from .api import normalize_now
from .pipeline_diagram import PLANES

_UNHEALTHY_STATUSES = {"stalled", "failed"}


def _health_from_runs(runs: list[dict], stage: str, *, now: datetime, stale_after_s: float) -> dict:
    """``{"stage", "status", "step", "output_tail", "elapsed_s", "error"}`` computed from
    `stage`'s own already-fetched `runs` -- the pure computation shared by :func:`stage_health`
    (fetches its own single-stage list) and :func:`health_panel` (fetches once, unfiltered, and
    groups locally, so N stages cost one store read instead of N -- a code-review finding: an
    earlier version had :func:`health_panel` call :func:`stage_health` once per stage, each
    doing its own `store.list_runs(stage=...)` call, reintroducing the exact N-separate-reads
    anti-pattern :mod:`dashboard.pipeline_diagram`'s own review already found and fixed for this
    identical case one PR earlier -- costly on :class:`~src.store.firestore_store.FirestoreStore`,
    which has no server-side `stage` filter and streams the whole collection per call).

    Uses :func:`~src.liveness.latest_status_and_record` (not `latest_status` plus a second,
    independent `stages.latest_run` call) so `step`/`output_tail`/`error` always come off the
    exact same record `status` was classified from -- an earlier version derived "the latest
    record" two different ways, which could disagree on a same-second tie between a terminal
    and a non-terminal event (a code-review finding, confirmed by three independent angles).
    """
    status, latest = latest_status_and_record(runs, now=now, stale_after_s=stale_after_s)
    step = output_tail = error = None
    elapsed_s = None
    if latest is not None:
        step = latest.get("step")
        output_tail = latest.get("output_tail")
        error = latest.get("error")
        recorded = parse_last_run(latest.get("recorded_at") or "")
        elapsed_s = (now - recorded).total_seconds() if recorded else None
    return {
        "stage": stage,
        "status": status,
        "step": step,
        "output_tail": output_tail,
        "elapsed_s": elapsed_s,
        "error": error,
    }


def stage_health(
    store: Store, stage: str, *, now: datetime | None = None, stale_after_s: float
) -> dict:
    """``{"stage", "status", "step", "output_tail", "elapsed_s", "error"}`` for `stage`'s own
    run history -- see :func:`_health_from_runs` for field semantics. A standalone,
    single-stage convenience matching :func:`~dashboard.api.stage_status`'s own `store`+`stage`
    signature; a caller computing health for every stage at once should use
    :func:`health_panel` instead, which fetches `store.list_runs()` only once.
    """
    when = normalize_now(now)
    runs = store.list_runs(stage=stage)
    return _health_from_runs(runs, stage, now=when, stale_after_s=stale_after_s)


def health_panel(store: Store, *, now: datetime | None = None, stale_after_s: float) -> dict:
    """``{"stages": [_health_from_runs(...) per real orchestrator stage], "overall": "green" |
    "red"}`` -- the top-level badge this todo's own "Why" names ("a stall must show red,
    never a frozen 'running'"): ``"red"`` if any stage is ``stalled``/``failed``, ``"green"``
    otherwise (``running``/``ok``/``"never run"`` all read as "nothing wrong right now").

    Fetches ``store.list_runs()`` exactly **once** (unfiltered) and groups the result by
    ``stage`` locally, mirroring :func:`~dashboard.pipeline_diagram.pipeline_diagram`'s own
    identical fix for the identical cost concern (see :func:`_health_from_runs`'s own
    docstring).
    """
    when = normalize_now(now)
    runs_by_stage: dict[str, list[dict]] = defaultdict(list)
    for run in store.list_runs():
        stage = run.get("stage")
        if isinstance(stage, str):
            runs_by_stage[stage].append(run)

    stages = [
        _health_from_runs(
            runs_by_stage.get(plane["id"], []), plane["id"], now=when, stale_after_s=stale_after_s
        )
        for plane in PLANES
    ]
    overall = "red" if any(s["status"] in _UNHEALTHY_STATUSES for s in stages) else "green"
    return {"stages": stages, "overall": overall}
