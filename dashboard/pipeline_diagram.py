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

from datetime import datetime, timezone

from src.store.base import Store

from . import api

# One entry per real, liveness-tracked orchestrator stage (`src/orchestrator.py::_real_stages`)
# -- labels/descriptions drawn from CLAUDE.md's own "Architecture (one paragraph)" section, so
# this diagram can't independently drift from that prose description of the same three planes.
_PLANES = (
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
    ...]}`` -- one node per real orchestrator stage (live status via
    :func:`~dashboard.api.stage_status`) plus one central ``"kb"`` node every plane connects
    to, and nothing else.

    `stale_after_s` has no default here, deliberately: it's forwarded as-is to
    :func:`~dashboard.api.stage_status`, which itself has none -- `src/orchestrator.py`'s own
    docstring already flags that the real threshold is an open question ("`stale_after_s` for
    the real pipeline should stay generous until [real per-substep heartbeats land]"), so
    picking a number here would guess at a design question this todo doesn't own.
    """
    when = now or datetime.now(timezone.utc)
    nodes = [
        {"id": "kb", "label": "Knowledge Base", "description": "versioned KB state", "status": None}
    ]
    for plane in _PLANES:
        status = api.stage_status(store, plane["id"], now=when, stale_after_s=stale_after_s)
        nodes.append({**plane, "status": status})
    edges = [(plane["id"], "kb") for plane in _PLANES]
    return {"nodes": nodes, "edges": edges}
