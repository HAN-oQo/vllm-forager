"""Shared helpers for M3 pipeline stages (T3.2+): repro, verify, and future review/gate stages
each look up one KB item before acting and persist a run record afterward — this module is the
one place those two mechanical, schema-independent steps live, so a third and fourth stage
don't each retype them from memory of an earlier one's version.

Deliberately NOT shared here: how a stage builds its own LLM prompt or shapes its own run
record — those differ in schema per stage (a repro command vs. a patch, a `reproduced` flag vs.
a `verified` one), and forcing them through one generic helper now, with only two call sites,
would be exactly the premature abstraction CLAUDE.md warns against ("three similar lines is
better than a premature abstraction").
"""

from __future__ import annotations

import sys

from .store.base import Store


def get_item_or_skip(store: Store, repo: str, number: int, *, stage: str) -> dict | None:
    """`store.get_item(repo, number)`, or `None` (logged to stderr as `stage`) if absent — the
    "no candidate to act on" skip every M3 stage applies the same way, not raised."""
    item = store.get_item(repo, number)
    if item is None:
        print(f"{stage}: no KB record for {repo}#{number}", file=sys.stderr)
    return item


def record_run_best_effort(
    store: Store, record: dict, *, stage: str, repo: str, number: int
) -> None:
    """`store.record_run(record)`, logging (not raising) on failure — an already-completed
    MI250 result must never be discarded over a mere KB-write hiccup."""
    try:
        store.record_run(record)
    except Exception as exc:
        print(f"{stage}: failed to record run for {repo}#{number}: {exc}", file=sys.stderr)
