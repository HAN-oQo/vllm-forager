"""Pipeline data-flow diagram (T5.5): the architecture view of the running system, rendered
from live stage metadata rather than the hand-drawn mermaid diagram in `docs/PLAN.md`.

Scope matches this todo's own worked example ("the 3-plane flow rendered with each stage's
current status") and what live metadata can actually report today: `src/orchestrator.py` runs
exactly three real, liveness-tracked stages (`"collect"`, `"intel"`, `"contribution"`) --
`docs/PLAN.md`'s own richer diagram also shows an "outer loop" (grader/curator/scout) and
individual per-agent nodes inside each plane, but those aren't independently tracked
`Store.list_runs(stage=...)` events (see `src/orchestrator.py`'s own "Known limitations":
grader/curator/scout run as CLI-only or folded into the "contribution" tick, with no per-agent
heartbeat wiring yet) -- this diagram shows exactly the granularity that has a real status to
show, rather than inventing per-agent status this codebase can't back up yet.

Modeled as a hub-and-spoke graph, not a linear chain: CLAUDE.md's own architecture paragraph
frames this as "three planes **over one knowledge base**," matching `docs/PLAN.md`'s own
diagram (every plane's arrows run through the KB, not directly to the next plane) -- a fake
linear "collect -> intel -> contribution" chain would misrepresent that every plane reads and
writes the same shared store independently, not hands off directly to the next one.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from src.liveness import latest_status
from src.store.base import Store

from .api import normalize_now

# One entry per real, liveness-tracked orchestrator stage (`src/orchestrator.py::_real_stages`)
# -- labels/descriptions drawn from CLAUDE.md's own "Architecture (one paragraph)" section, so
# this diagram can't independently drift from that prose description of the same three planes.
#
# These `id`s are a second, hand-maintained copy of `_real_stages()`'s own `Stage.name` values
# rather than importing that function directly (a code-review finding) -- `_real_stages()`
# locally imports and wires up `analyze`/`candidates`/`collector`/`forecast`/`report` (the real
# CLIs each stage invokes), and pulling that whole production call graph into a read-only
# dashboard module just to read three name strings is a worse cost than the drift risk it would
# close. `test_pipeline_diagram_plane_ids_match_the_real_orchestrator_stages` (this todo's own
# test file) guards against drift at test time instead, mirroring `tests/test_orchestrator.py`'s
# own established `{stage.name: stage for stage in orchestrator._real_stages()}` pattern.
#
# Public (not `_PLANES`) since T5.8's `dashboard.health` also needs exactly this stage list
# and imports it from here rather than defining a third independent copy.
PLANES = (
    {
        "id": "collect",
        "label": "Data plane",
        "description": "collect -> normalize -> embed (no LLM)",
    },
    {
        "id": "intel",
        "label": "Intelligence plane",
        "description": "classify / forecast / cited report (scheduled LLM)",
    },
    {
        "id": "contribution",
        "label": "Contribution plane",
        "description": (
            "reproduce -> patch -> verify on MI250 -> ensemble self-review -> "
            "human gate -> draft PR"
        ),
    },
)


def pipeline_diagram(store: Store, *, now: datetime | None = None, stale_after_s: float) -> dict:
    """``{"nodes": [{"id", "label", "description", "status"}, ...], "edges": [(plane_id, "kb"),
    ...]}`` -- one node per real orchestrator stage plus one central ``"kb"`` node every plane
    connects to, and nothing else.

    Fetches ``store.list_runs()`` exactly **once** (unfiltered) and groups the result by
    ``stage`` locally, rather than calling :func:`~dashboard.api.stage_status` once per plane
    (a code-review finding: that would issue 3 separate full-collection reads on
    :class:`~src.store.firestore_store.FirestoreStore`, which has no server-side ``stage``
    filter -- billed per document, and only getting worse once a live health panel starts
    polling this repeatedly). Computes each plane's status directly via
    :func:`~src.liveness.latest_status` over its own already-fetched run list instead.

    `stale_after_s` has no default here, deliberately: `src/orchestrator.py`'s own docstring
    already flags that the real threshold is an open question ("`stale_after_s` for the real
    pipeline should stay generous until [real per-substep heartbeats land]"), so picking a
    number here would guess at a design question this todo doesn't own.
    """
    when = normalize_now(now)
    runs_by_stage: dict[str, list[dict]] = defaultdict(list)
    for run in store.list_runs():
        stage = run.get("stage")
        if isinstance(stage, str):
            runs_by_stage[stage].append(run)

    nodes = [
        {"id": "kb", "label": "Knowledge Base", "description": "versioned KB state", "status": None}
    ]
    edges = []
    for plane in PLANES:
        runs = runs_by_stage.get(plane["id"], [])
        status = latest_status(runs, now=when, stale_after_s=stale_after_s)
        nodes.append({**plane, "status": status})
        edges.append((plane["id"], "kb"))
    return {"nodes": nodes, "edges": edges}
