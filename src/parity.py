"""Parity matrix (T2.4): engine × capability, with evidence + gap flags.

PLAN.md's contribution strategy is "find what exists elsewhere but is missing upstream" — this
module makes that comparison concrete. A **capability** is a taxonomy category (T1.5's
classified ``category`` field — the same controlled vocabulary already applied across every
tracked repo); an **engine** is a tracked repo (:data:`~src.config.REPOS`). A cell is
``present`` for (engine, capability) only if that engine has at least one *shipped* item under
it — a merged PR (:func:`~src.agents.reporter.is_merged`), not just an open issue or a PR that
never landed — since a capability that's only been *discussed* on an engine isn't evidence it
actually exists there. "Shipped" uses this codebase's established ``state == "closed"``
approximation for "merged" (the collector doesn't capture GitHub's own ``merged``/
``merged_at`` flags — see :func:`~src.agents.reporter.is_merged`'s own docstring, also used by
:mod:`dashboard.render`'s PR-state chip): a closed-without-merging PR would be misread as
shipped, and this risk is sharpest for a *sparse* cell (a capability with only one or two
items total) where a single such PR can flip the whole cell — not just a diffuse aggregate
risk, directionally right only until the collector captures the real flag.

:func:`find_gaps` is the sharp end: every capability shipped on the **fork**
(:data:`~src.config.REPOS`'s ``role: "fork"`` entry, i.e. ROCm/vllm) but not shipped upstream
(``role: "primary"``, i.e. vllm-project/vllm) — exactly the port-candidate signal T2.5's Scout
will rank. "Not shipped upstream" covers both "no evidence at all" and "discussed upstream but
never merged" — either way there's nothing to point at upstream yet.

Known limitations, not fixed here:
- A capability's key is its *exact* flattened ``category`` string, not its hierarchical
  ``path``. Two items describing the same real capability can land at different depths
  (:func:`~src.agents.analyst._canonical_path` returns a shorter path when the model is only
  confident about the higher levels) — e.g. the fork's PR classified all the way to
  ``"quantization > FP8 > kv-cache"`` while upstream's equivalent only validated to
  ``"quantization > FP8"``. These become two different, non-matching capability keys, which
  can produce a false-positive gap (fork's deeper key looks port-worthy when the shallower
  upstream key actually covers it) or a false negative (the reverse). Same root cause as
  :mod:`src.trends`'s own documented "buckets by the full flattened path, not rolled up by
  prefix" limitation — a real fix means teaching capability comparison to roll up by path
  prefix, not just this module's problem to solve alone.
- Like :mod:`~src.agents.curator`'s own retirement/new-category proposals, this is computed
  fresh from a full :meth:`Store.query` scan on every call, with no delta tracking of which
  gaps were already surfaced to a human.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import config
from .agents.reporter import evidence_url, is_merged


class ParityError(RuntimeError):
    """No :data:`~src.config.REPOS` entry has the requested ``role``."""


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


def _best_evidence(shipped: list[dict]) -> str | None:
    """The most-recently-created shipped item's evidence URL — falling through to an older
    item if the newest one's evidence can't be resolved (:func:`~src.agents.reporter.
    evidence_url`'s own documented "" case for a record missing both ``url`` and ``repo``/
    ``number``), rather than reporting no evidence when a usable citation exists elsewhere in
    the same cell.
    """
    for item in sorted(shipped, key=lambda item: item.get("created_at") or "", reverse=True):
        url = evidence_url(item)
        if url:
            return url
    return None


def build_matrix(items: list[dict]) -> list[ParityCell]:
    """One :class:`ParityCell` per (engine, capability) combination with at least one
    classified item — `present=True` if any of that combination's items shipped (see
    :func:`~src.agents.reporter.is_merged`), else `False` (there was activity, just nothing
    that landed yet).

    Items with no ``repo`` or no (string) ``category`` are excluded — the same "only
    classified items count" rule :mod:`src.trends`'s ``category_trends`` established (also
    applied by :mod:`~src.agents.policy_update`'s ``grade_category``).
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
        shipped = [item for item in cell_items if is_merged(item)]
        cells.append(
            ParityCell(
                engine=engine,
                capability=capability,
                present=bool(shipped),
                evidence=_best_evidence(shipped) if shipped else None,
            )
        )
    return cells


def _default_engine(role: str) -> str:
    for repo in config.REPOS:
        if repo["role"] == role:
            return repo["slug"]
    raise ParityError(f"no config.REPOS entry has role {role!r}")


def find_gaps(
    cells: list[ParityCell], *, fork_engine: str | None = None, upstream_engine: str | None = None
) -> list[Gap]:
    """Every capability shipped on `fork_engine` but not on `upstream_engine`.

    `fork_engine`/`upstream_engine` default to :data:`~src.config.REPOS`'s ``"fork"``/
    ``"primary"``-role entries (ROCm/vllm, vllm-project/vllm) — this project's real fork and
    contribution target — overridable for tests or a future multi-fork setup.

    Raises:
        ParityError: a default was needed and no :data:`~src.config.REPOS` entry has that role.
    """
    fork_engine = fork_engine or _default_engine("fork")
    upstream_engine = upstream_engine or _default_engine("primary")
    upstream_cells = {cell.capability: cell for cell in cells if cell.engine == upstream_engine}
    gaps = []
    for cell in cells:
        if cell.engine != fork_engine or not cell.present:
            continue
        upstream_cell = upstream_cells.get(cell.capability)
        if upstream_cell is None or not upstream_cell.present:
            gaps.append(Gap(capability=cell.capability, evidence=cell.evidence))
    return gaps
