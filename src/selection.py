"""Candidate selection (T5.10): the human-in-the-loop signal that gates which candidates the
contribution plane is allowed to work, *before* any MI250 time is spent on one — distinct from
:mod:`src.gate` (T3.5/T3.7), which reviews the *result* of already-completed work. This is the
earlier gate: whether work should even start.

Storage mirrors :mod:`src.gate`'s own approach for the identical kind of "durable human
decision" record: appended to the KB's run-event history via :meth:`~src.store.base.Store.
record_run` as ``{"repo", "number", "stage": "selection", "decision": "selected" | "skip",
"by", "recorded_at"}``, read back via :meth:`~src.store.base.Store.list_runs` — not a new KB
collection, the same "a run record's shape beyond repo/number is caller-defined" contract
every other stage (`gate`, `self_review`, `pr_quality`) already relies on. A human can change
their mind: :func:`latest_decision` always reads the most recently recorded decision for a
candidate, never the first.

**Deliberately does not use** :func:`~src.stages.record_run_best_effort` **for the write
itself**, unlike every other human-decision-recording stage (`gate.py`, `self_review.py`,
`repro.py`, `engineer.py`, `pr_quality.py`) — a code-review finding that this deviation, while
correct, needed to be stated and justified explicitly, the same way
:func:`~src.orchestrator._best_effort`'s own docstring explains its analogous divergence.
Every one of those precedents records an outcome *after* real work (verification, a patch, an
LLM judgment) already completed — the KB write is a secondary bookkeeping step, and
best-effort exists so a bookkeeping hiccup can't erase real, already-done work.
:func:`record_decision`'s write **is** the entire action; there is no prior "real work" for a
swallowed failure to protect. Using best-effort semantics here would mean a human's "Work
this"/"Skip" click could silently have zero effect (the write raises internally, gets logged
and swallowed, the dashboard shows no error) with nothing to tell the human their instruction
was never recorded — the opposite of what a human-in-the-loop gate is for. Letting it raise
means a write failure surfaces as a visible error instead of a silent no-op.

:class:`~src.agents.scout.Candidate` carries no ``(repo, number)`` identity field of its own
(only a title, a scoring triad, and an ``evidence`` URL) — :func:`filter_selected` resolves
that identity via :func:`~src.agents.grader.parse_repo_number`, the same GitHub-URL-to-
``(repo, number)`` parser :mod:`~src.agents.grader`/:mod:`~src.agents.policy_update` already
use for the identical problem (a candidate/prediction that only carries an evidence URL, not a
first-class identity).

**Known, disclosed limitation, not fixed here:** neither :mod:`src.orchestrator`'s real
``_contribution`` stage nor :mod:`src.engineer` actually calls :func:`filter_selected`/
:func:`is_selected` yet — this module's write path (a human clicking "Work this"/"Skip") and
its own read functions are fully built and tested, but nothing in the real pipeline consults
them, so a recorded decision currently has no effect on what the orchestrator/engineer
actually works. Wiring `filter_selected` into `orchestrator.py`'s `_contribution` stage is a
further M4/M5 integration step, not this dashboard-facing todo's own scope (matching this
project's own established "infrastructure built, real-stage wiring deferred" shape, e.g.
T4.4's heartbeats, T5.1's `candidates()`).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from .agents.forecaster import TS_FORMAT
from .agents.grader import parse_repo_number
from .agents.scout import Candidate
from .stages import get_item_or_skip
from .store.base import Store

_VALID_DECISIONS = {"selected", "skip"}


class SelectionError(RuntimeError):
    """An invalid `decision` value was given to :func:`record_decision`."""


def _latest_by_timestamp(runs: list[dict]) -> dict | None:
    """The most recently recorded of `runs`, or `None` if empty — breaking a same-
    `recorded_at` tie toward the *last* entry in `runs`' own order, not
    :func:`~src.stages.latest_run`'s own first-wins tie-break.

    A code-review finding, empirically reproduced: two decisions recorded in the same
    wall-clock second (every timestamp here has only second resolution) are not a rare edge
    case for a "Work this"/"Skip" toggle — a human noticing and correcting a misclick within
    the same second is an ordinary interaction, and `stages.latest_run`'s own `max()` (Python
    semantics: first-encountered maximum wins a tie) would return the human's *first* decision
    instead of their corrected one. This function is reliable on
    :class:`~src.store.jsonl_store.JsonlStore`, whose own `list_runs` docstring promises exact
    append order; :class:`~src.store.firestore_store.FirestoreStore`'s own `list_runs`
    discloses no ordering guarantee at all, so a same-second tie there has no reliable
    resolution regardless of tie-break rule — a pre-existing backend limitation this function
    doesn't newly introduce. Kept local to this module rather than changing
    `stages.latest_run` itself, which `gate.py`/`self_review.py`/`engineer.py`/etc. all also
    depend on — fixing this tie-break globally is a larger, separate change.
    """
    if not runs:
        return None
    max_ts = max(run.get("recorded_at") or "" for run in runs)
    for run in reversed(runs):
        if (run.get("recorded_at") or "") == max_ts:
            return run
    return None  # unreachable: `runs` is non-empty, so some run must match `max_ts`


def record_decision(
    store: Store,
    repo: str,
    number: int,
    *,
    decision: str,
    by: str | None = None,
    now: datetime | None = None,
) -> dict | None:
    """Record a human's select/skip decision for candidate (`repo`, `number`); returns the
    recorded record, or `None` if (`repo`, `number`) isn't a real KB item — mirrors
    :mod:`src.gate`'s own :func:`~src.stages.get_item_or_skip` precondition (used by its
    `_readiness` check), so a stale dashboard row or a typo'd identifier can't silently create
    a permanent orphan decision record for an item that was never collected.

    Raises:
        SelectionError: `decision` isn't the literal string ``"selected"`` or ``"skip"`` —
            fails loudly on a typo rather than silently persisting a value
            :func:`latest_decision`/:func:`is_selected` would never recognize.
    """
    if decision not in _VALID_DECISIONS:
        raise SelectionError(
            f"decision must be one of {sorted(_VALID_DECISIONS)!r}, got {decision!r}"
        )
    if get_item_or_skip(store, repo, number, stage="selection") is None:
        return None
    when = now or datetime.now(timezone.utc)
    record = {
        "repo": repo,
        "number": number,
        "stage": "selection",
        "decision": decision,
        "by": by,
        "recorded_at": when.strftime(TS_FORMAT),
    }
    store.record_run(record)
    return record


def latest_decision(store: Store, repo: str, number: int) -> str | None:
    """The most recently recorded decision for (`repo`, `number`), or `None` if it's never
    been decided — a human can change their mind, so this is never the *first* recorded
    decision (see :func:`_latest_by_timestamp` for the same-second tie-break rule)."""
    latest = _latest_by_timestamp(store.list_runs(repo=repo, number=number, stage="selection"))
    return latest.get("decision") if latest is not None else None


def is_selected(store: Store, repo: str, number: int) -> bool:
    """Whether (`repo`, `number`)'s most recent decision is ``"selected"`` — the single-
    candidate check the (not-yet-wired, see module docstring) orchestrator/engineer queue
    filter would use before starting MI250 work on one specific candidate."""
    return latest_decision(store, repo, number) == "selected"


def _decisions_by_key(store: Store) -> dict[tuple[str, int], str]:
    """Every (`repo`, `number`) with any recorded selection history, mapped to its most
    recent decision — built from **one** ``store.list_runs(stage="selection")`` fetch, so
    :func:`filter_selected` costs one store read total, not one per candidate (a code-review
    finding, confirmed by two independent angles: an earlier version called :func:`is_selected`
    per candidate, each issuing its own full store query — the exact N-separate-reads
    anti-pattern already found and fixed twice in this milestone, for
    :mod:`dashboard.pipeline_diagram` and :mod:`dashboard.health`).
    """
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in store.list_runs(stage="selection"):
        repo, number = run.get("repo"), run.get("number")
        if isinstance(repo, str) and isinstance(number, int):
            by_key[(repo, number)].append(run)
    decisions: dict[tuple[str, int], str] = {}
    for key, runs in by_key.items():
        latest = _latest_by_timestamp(runs)
        decision = latest.get("decision") if latest is not None else None
        if isinstance(decision, str):
            decisions[key] = decision
    return decisions


def filter_selected(store: Store, candidates: list[Candidate]) -> list[Candidate]:
    """Only `candidates` that resolve (via their own ``evidence`` URL) to a ``(repo, number)``
    with a ``"selected"`` decision — the engineer/orchestrator's own queue: human-selected
    only, per this todo's own worked example (`"the engineer's queue = selected-only"`). See
    this module's own docstring for the disclosed "not yet actually called by the real
    orchestrator/engineer" gap.

    A candidate whose `evidence` doesn't parse as a GitHub issue/PR URL (mirrors
    :func:`~src.agents.grader.parse_repo_number`'s own documented "not a recognized URL" case)
    can never have been selected — there's no ``(repo, number)`` to have recorded a decision
    against — so it's excluded, not raised on.
    """
    decisions = _decisions_by_key(store)
    selected = []
    for candidate in candidates:
        parsed = parse_repo_number(candidate.evidence)
        if parsed is not None and decisions.get(parsed) == "selected":
            selected.append(candidate)
    return selected
