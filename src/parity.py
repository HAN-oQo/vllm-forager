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

:func:`find_gaps` is the sharp end: every capability shipped on **any** tracked engine but not
shipped on a given ``target_engine`` (one of :data:`~src.config.REPOS`'s ``role: "primary"``
entries — a real contribution target) — exactly the port-candidate signal T2.5's Scout will
rank. "Not shipped on the target" covers both "no evidence at all" and "discussed there but
never merged" — either way there's nothing to point at that target yet.

**Generalized off fork-vs-upstream (T3.14):** this used to compare exactly two hardcoded
engines — the retired ``ROCm/vllm`` fork (``role: "fork"``) against ``vllm-project/vllm``
(``role: "primary"``). Now that the fork is retired and the target set is multi-domain (three
``"primary"`` contribution targets — vllm, vllm-omni, vime — each surrounded by its own
ecosystem of ``"parity"``/``"source"``/``"radar"`` repos), a gap can come from *any* tracked
engine, not one designated fork, and there's more than one target to check it against. Use
:func:`find_gaps` for one specific target, or :func:`find_gaps_for_all_targets` to check every
``"primary"`` target at once — the latter is what :mod:`~src.agents.scout` actually uses, since
a capability upstream ships but vllm-omni or vime is missing is just as real a port candidate
as one from an external engine.

Known limitations, not fixed here:
- A capability's key is its *exact* flattened ``category`` string, not its hierarchical
  ``path``. Two items describing the same real capability can land at different depths
  (:func:`~src.agents.analyst._validated_subpath` returns a shorter path when the model is only
  confident about the higher levels) — e.g. a source engine's PR classified all the way to
  ``"quantization > FP8 > kv-cache"`` while the target's equivalent only validated to
  ``"quantization > FP8"``. These become two different, non-matching capability keys, which
  can produce a false-positive gap (the source's deeper key looks port-worthy when the
  shallower target key actually covers it) or a false negative (the reverse). Same root cause
  as :mod:`src.trends`'s own documented "buckets by the full flattened path, not rolled up by
  prefix" limitation — a real fix means teaching capability comparison to roll up by path
  prefix, not just this module's problem to solve alone. **Growing surface (T3.14):**
  generalizing from one fork/one upstream to N tracked engines × 3 primary targets across four
  taxonomy domains multiplies how often two engines can independently classify the same real
  capability at different depths — this limitation was always here, but the multi-domain
  target set makes it bite more often, not something this rework fixes.
- Like :mod:`~src.agents.curator`'s own retirement/new-category proposals, this is computed
  fresh from a full :meth:`Store.query` scan on every call, with no delta tracking of which
  gaps were already surfaced to a human.
- :func:`find_gaps`'s "first source engine (in `source_engines` order) to ship it wins" pick
  is a list-order artifact, not a judgment about which source is the best evidence — if two
  independent engines both ship the same capability, only one's evidence URL is cited and the
  corroborating second source is silently dropped. Was moot with a single fork; now that
  `source_engines` can be several engines, a human asking "why does it cite X and not Y" has no
  better answer than "REPOS list order."
"""

from __future__ import annotations

import sys
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
    """`capability` is shipped on `source_engine` but not on `target_engine` — a port
    candidate into `target_engine`."""

    capability: str
    target_engine: str
    source_engine: str
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


def _engines_with_role(role: str) -> list[str]:
    """Every :data:`~src.config.REPOS` slug with `role`, in file order."""
    return [repo["slug"] for repo in config.REPOS if repo["role"] == role]


def _default_engine(role: str) -> str:
    engines = _engines_with_role(role)
    if not engines:
        raise ParityError(f"no config.REPOS entry has role {role!r}")
    return engines[0]


def _domain_of(slug: str) -> str | None:
    """`slug`'s :data:`~src.config.REPOS` ``domain``, or `None` if `slug` isn't tracked."""
    for repo in config.REPOS:
        if repo["slug"] == slug:
            return repo.get("domain")
    return None


def _index_by_engine(cells: list[ParityCell]) -> dict[str, dict[str, ParityCell]]:
    """`cells` grouped `{engine: {capability: cell}}` — built once by
    :func:`find_gaps_for_all_targets` and threaded through :func:`find_gaps`'s private
    `_by_engine` so checking every ``"primary"`` target doesn't re-derive the identical index
    from the same `cells` list once per target (a code-review finding on this rework)."""
    by_engine: dict[str, dict[str, ParityCell]] = defaultdict(dict)
    for cell in cells:
        by_engine[cell.engine][cell.capability] = cell
    return by_engine


def find_gaps(
    cells: list[ParityCell],
    *,
    target_engine: str | None = None,
    source_engines: list[str] | None = None,
    _by_engine: dict[str, dict[str, ParityCell]] | None = None,
) -> list[Gap]:
    """Every capability shipped on a source engine but not on `target_engine` — a port
    candidate into `target_engine`, citing the first source engine (in `source_engines` order)
    that shipped it.

    `target_engine` defaults to :data:`~src.config.REPOS`'s first ``"primary"``-role entry (a
    real contribution target). `source_engines` defaults to every OTHER tracked engine in
    :data:`~src.config.REPOS` **whose `domain` is either `"engine"` (a generic cross-domain
    comparison baseline -- SGLang/Dynamo/llm-d) or matches `target_engine`'s own `domain`**
    (its surrounding ecosystem, e.g. `verl`/`OpenRLHF` for the `"rl"`-domain `vime` target) --
    generalized off the old hardcoded ``"fork"``-role default (T3.14, ROCm/vllm retired), but
    scoped by `config.REPOS`'s own `domain` field (its stated purpose: "groups a target with
    the ecosystem watched around it") so a capability from one domain's ecosystem doesn't get
    flagged as "missing" on an unrelated-domain target -- e.g. an RL post-training capability
    shipped only on `vime` (domain `"rl"`) must never be flagged as a gap against the
    speech-focused `vllm` or the omni-focused `vllm-omni`, now that there's more than one
    `"primary"` target and cross-domain contamination is actually reachable (a review finding
    on this very rework: before T3.14 there was exactly one fork and one upstream, so no domain
    could ever cross). A source or target engine with no cell at all (nothing shipped or even
    classified yet) is treated like any other "not present" case, not an error — overridable
    for tests or to scope a check to specific engines (an explicit `source_engines` bypasses
    the domain scoping entirely; it's the caller's own choice).

    Raises:
        ParityError: `target_engine` was defaulted and no :data:`~src.config.REPOS` entry has
            role ``"primary"``.
    """
    target_engine = target_engine or _default_engine("primary")
    if source_engines is None:
        target_domain = _domain_of(target_engine)
        source_engines = [
            repo["slug"]
            for repo in config.REPOS
            if repo["slug"] != target_engine
            and (repo.get("domain") == "engine" or repo.get("domain") == target_domain)
        ]

    by_engine = _by_engine if _by_engine is not None else _index_by_engine(cells)
    target_cells = by_engine.get(target_engine, {})

    gaps: dict[str, Gap] = {}
    for source_engine in source_engines:
        for capability, cell in by_engine.get(source_engine, {}).items():
            if not cell.present or capability in gaps:
                continue
            target_cell = target_cells.get(capability)
            if target_cell is not None and target_cell.present:
                continue
            gaps[capability] = Gap(
                capability=capability,
                target_engine=target_engine,
                source_engine=source_engine,
                evidence=cell.evidence,
            )
    return list(gaps.values())


def find_gaps_for_all_targets(cells: list[ParityCell]) -> list[Gap]:
    """:func:`find_gaps` for every :data:`~src.config.REPOS` ``"primary"``-role engine — this
    project now has multiple contribution targets (vllm, vllm-omni, vime), not one, so a gap
    check against a single default target would silently miss capabilities the other targets
    are missing.

    Raises:
        ParityError: no :data:`~src.config.REPOS` entry has role ``"primary"`` (the same error
            :func:`_default_engine` raises -- reused via a first defaulted call rather than a
            second hand-copied message, so the two can't drift out of sync).
    """
    targets = _engines_with_role("primary") or [_default_engine("primary")]
    by_engine = _index_by_engine(cells)
    gaps: list[Gap] = []
    for target in targets:
        gaps.extend(find_gaps(cells, target_engine=target, _by_engine=by_engine))
    return gaps


def safe_find_gaps_for_all_targets(cells: list[ParityCell], *, caller: str) -> list[Gap]:
    """:func:`find_gaps_for_all_targets`, degrading a :class:`ParityError` (no
    ``"primary"``-role engine configured) to an empty list instead of raising — a config gap
    that has nothing to do with candidate-discovery or dashboard-reading shouldn't crash
    either. `caller` (a short label, e.g. ``"scout"``/``"dashboard.api"``) tags the stderr
    message so the two real call sites (:func:`~src.agents.scout.discover_from_store`,
    :func:`~dashboard.api.parity`) can share one implementation instead of independently
    hand-copying the identical try/except (a code-review finding on the latter's own PR).
    """
    try:
        return find_gaps_for_all_targets(cells)
    except ParityError as exc:
        print(f"{caller}: skipping parity gaps: {exc}", file=sys.stderr)
        return []
