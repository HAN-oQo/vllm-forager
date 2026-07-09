"""Candidate selection surface (T5.10): the dashboard's second -- and, per the M5 milestone's
own "Expected output" framing, last -- write path. "Work this" / "Skip" writes a decision
record via :mod:`src.selection`, so the orchestrator/engineer only spends MI250 time on
human-selected candidates. Distinct from T5.6 (:mod:`dashboard.review`), which approves the
*result* of already-completed work; this is the earlier gate, deciding whether work should
even start.

A thin wrapper composing over :mod:`src.selection` -- not a second implementation, matching
the "dashboard composes over an existing src.* function" shape T5.6/T5.7/T5.9 already
established (T5.9's own review specifically caught and fixed a version of this module that
implemented its logic directly in the dashboard layer instead of composing).
"""

from __future__ import annotations

from datetime import datetime

from src.selection import latest_decision, record_decision
from src.store.base import Store


def select_candidate(
    store: Store, repo: str, number: int, *, by: str | None = None, now: datetime | None = None
) -> dict | None:
    """ "Work this" -- records a ``"selected"`` decision for candidate (`repo`, `number`), or
    `None` if it isn't a real KB item (see :func:`~src.selection.record_decision`)."""
    return record_decision(store, repo, number, decision="selected", by=by, now=now)


def skip_candidate(
    store: Store, repo: str, number: int, *, by: str | None = None, now: datetime | None = None
) -> dict | None:
    """ "Skip" -- records a ``"skip"`` decision for candidate (`repo`, `number`), or `None`
    if it isn't a real KB item (see :func:`~src.selection.record_decision`)."""
    return record_decision(store, repo, number, decision="skip", by=by, now=now)


def candidate_decision_status(store: Store, repo: str, number: int) -> str | None:
    """The most recently recorded decision for candidate (`repo`, `number`) -- ``"selected"``,
    ``"skip"``, or `None` if never decided -- so the Candidates tab can show each row's
    current decision alongside its "Work this"/"Skip" controls."""
    return latest_decision(store, repo, number)
