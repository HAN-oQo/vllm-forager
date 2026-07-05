"""Parity matrix (T2.4): engine × capability, with evidence + gap flags.

PLAN.md's contribution strategy is "find what exists elsewhere but is missing upstream" — this
module makes that comparison concrete. A **capability** is a taxonomy category (T1.5's
classified ``category`` field — the same controlled vocabulary already applied across every
tracked repo); an **engine** is a tracked repo (:data:`~src.config.REPOS`). A cell is
``present`` for (engine, capability) only if that engine has at least one *shipped* item under
it — a merged PR, not just an open issue or a PR that never landed — since a capability that's
only been *discussed* on an engine isn't evidence it actually exists there. "Shipped" uses this
codebase's established ``state == "closed"`` approximation for "merged" (the collector doesn't
capture GitHub's own ``merged``/``merged_at`` flags — see :mod:`dashboard.render`'s own
documented version of the same gap): a closed-without-merging PR would be misread as shipped,
but every item this codebase actually collects for a real capability is far more often merged
than abandoned, so this is directionally right until the collector captures the real flag.

:func:`find_gaps` is the sharp end: every capability shipped on the **fork**
(:data:`~src.config.REPOS`'s ``role: "fork"`` entry, i.e. ROCm/vllm) but not shipped upstream
(``role: "primary"``, i.e. vllm-project/vllm) — exactly the port-candidate signal T2.5's Scout
will rank. "Not shipped upstream" covers both "no evidence at all" and "discussed upstream but
never merged" — either way there's nothing to point at upstream yet.

Known limitation, not fixed here: like :mod:`~src.agents.curator`'s own retirement/new-category
proposals, this is computed fresh from a full :meth:`Store.query` scan on every call, with no
delta tracking of which gaps were already surfaced to a human.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import config
from .agents.reporter import evidence_url


def _is_shipped(item: dict) -> bool:
    """A PR, closed — see module docstring for the "closed == merged" approximation."""
    return item.get("type") == "pr" and item.get("state") == "closed"


@dataclass(frozen=True)
class ParityCell:
    """Whether `engine` has shipped `capability`, with evidence if so."""

    engine: str
    capability: str
    present: bool
    evidence: str | None


@dataclass(frozen=True)
class Gap:
    """`capability` is shipped on the fork but not upstream — a port candidate."""

    capability: str
    evidence: str | None


def build_matrix(items: list[dict]) -> list[ParityCell]:
    """One :class:`ParityCell` per (engine, capability) combination with at least one
    classified item — `present=True` if any of that combination's items shipped (see
    :func:`_is_shipped`), else `False` (there was activity, just nothing that landed yet).

    Items with no ``repo`` or no (string) ``category`` are excluded — the same "only
    classified items count" rule :mod:`src.trends`/:mod:`~src.agents.curator` already apply.
    """
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        repo = item.get("repo")
        category = item.get("category")
        if not repo or not category or not isinstance(category, str):
            continue
        by_cell[(repo, category)].append(item)

    cells = []
    for (engine, capability), cell_items in by_cell.items():
        shipped = [item for item in cell_items if _is_shipped(item)]
        evidence = evidence_url(shipped[0]) or None if shipped else None
        cells.append(
            ParityCell(
                engine=engine, capability=capability, present=bool(shipped), evidence=evidence
            )
        )
    return cells


def _default_engine(role: str) -> str:
    return next(repo["slug"] for repo in config.REPOS if repo["role"] == role)


def find_gaps(
    cells: list[ParityCell], *, fork_engine: str | None = None, upstream_engine: str | None = None
) -> list[Gap]:
    """Every capability shipped on `fork_engine` but not on `upstream_engine`.

    `fork_engine`/`upstream_engine` default to :data:`~src.config.REPOS`'s ``"fork"``/
    ``"primary"``-role entries (ROCm/vllm, vllm-project/vllm) — this project's real fork and
    contribution target — overridable for tests or a future multi-fork setup.
    """
    fork_engine = fork_engine or _default_engine("fork")
    upstream_engine = upstream_engine or _default_engine("primary")
    by_engine_capability = {(cell.engine, cell.capability): cell for cell in cells}
    gaps = []
    for cell in cells:
        if cell.engine != fork_engine or not cell.present:
            continue
        upstream_cell = by_engine_capability.get((upstream_engine, cell.capability))
        if upstream_cell is None or not upstream_cell.present:
            gaps.append(Gap(capability=cell.capability, evidence=cell.evidence))
    return gaps
