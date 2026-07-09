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

:class:`~src.agents.scout.Candidate` carries no ``(repo, number)`` identity field of its own
(only a title, a scoring triad, and an ``evidence`` URL) — :func:`filter_selected` resolves
that identity via :func:`~src.agents.grader.parse_repo_number`, the same GitHub-URL-to-
``(repo, number)`` parser :mod:`~src.agents.grader`/:mod:`~src.agents.policy_update` already
use for the identical problem (a candidate/prediction that only carries an evidence URL, not a
first-class identity).
"""

from __future__ import annotations

from datetime import datetime, timezone

from .agents.grader import parse_repo_number
from .agents.scout import Candidate
from .stages import latest_run
from .store.base import Store

_VALID_DECISIONS = {"selected", "skip"}
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class SelectionError(RuntimeError):
    """An invalid `decision` value was given to :func:`record_decision`."""


def record_decision(
    store: Store,
    repo: str,
    number: int,
    *,
    decision: str,
    by: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Record a human's select/skip decision for candidate (`repo`, `number`); returns the
    recorded record.

    Raises:
        SelectionError: `decision` isn't the literal string ``"selected"`` or ``"skip"`` —
            fails loudly on a typo rather than silently persisting a value
            :func:`latest_decision`/:func:`is_selected` would never recognize.
    """
    if decision not in _VALID_DECISIONS:
        raise SelectionError(
            f"decision must be one of {sorted(_VALID_DECISIONS)!r}, got {decision!r}"
        )
    when = now or datetime.now(timezone.utc)
    record = {
        "repo": repo,
        "number": number,
        "stage": "selection",
        "decision": decision,
        "by": by,
        "recorded_at": when.strftime(_TS_FORMAT),
    }
    store.record_run(record)
    return record


def latest_decision(store: Store, repo: str, number: int) -> str | None:
    """The most recently recorded decision for (`repo`, `number`), or `None` if it's never
    been decided — a human can change their mind, so this is never the *first* recorded
    decision.

    Known limitation, inherited from :func:`~src.stages.latest_run`, not introduced here: two
    decisions recorded in the same wall-clock second (every timestamp in this codebase has
    only second-resolution) tie on `recorded_at`, and `latest_run`'s own `max()` picks the
    *first*-encountered tied record, not the one actually written last — the reverse of what
    "a human changed their mind moments later" needs. Not realistically triggerable by a human
    clicking a UI control twice in one second in practice; not fixed here since it's a
    pre-existing property of the shared `latest_run` helper (also used by `gate.py`/
    `self_review.py`), not something specific to this module.
    """
    latest = latest_run(store.list_runs(repo=repo, number=number, stage="selection"))
    return latest.get("decision") if latest is not None else None


def is_selected(store: Store, repo: str, number: int) -> bool:
    """Whether (`repo`, `number`)'s most recent decision is ``"selected"`` — what the
    orchestrator/engineer's own queue filter checks before starting MI250 work on a
    candidate."""
    return latest_decision(store, repo, number) == "selected"


def filter_selected(store: Store, candidates: list[Candidate]) -> list[Candidate]:
    """Only `candidates` that resolve (via their own ``evidence`` URL) to a ``(repo, number)``
    with a ``"selected"`` decision — the engineer/orchestrator's own queue: human-selected
    only, per this todo's own worked example (`"the engineer's queue = selected-only"`).

    A candidate whose `evidence` doesn't parse as a GitHub issue/PR URL (mirrors
    :func:`~src.agents.grader.parse_repo_number`'s own documented "not a recognized URL" case)
    can never have been selected — there's no ``(repo, number)`` to have recorded a decision
    against — so it's excluded, not raised on.
    """
    selected = []
    for candidate in candidates:
        parsed = parse_repo_number(candidate.evidence)
        if parsed is None:
            continue
        repo, number = parsed
        if is_selected(store, repo, number):
            selected.append(candidate)
    return selected
