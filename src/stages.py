"""Shared helpers for M3 pipeline stages (T3.2+): repro, verify, and self-review each look up
one KB item before acting, call an LLM for a schema-shaped reply, and persist a run record
afterward — this module is the one place those three mechanical, schema-independent steps live,
so a fourth stage doesn't retype them from memory of an earlier one's version.

`complete_or_none` was added once a *third* stage (T3.4's self-review) needed the identical
"call the LLM, catch `llm.LLMError`, require a dict reply" shell around a schema call — with
two call sites (T3.2, T3.3) this stayed inline per-module, matching CLAUDE.md's "three similar
lines is better than a premature abstraction"; a third *whole copy of the same function* crossed
that line. `latest_run` was added the same way once "find the most-recently-recorded run in a
list, by `recorded_at`" appeared identically in `engineer.py`, `self_review.py`, and three
places in `gate.py` (T3.5) — five copies of one `max(...)` expression. What's still deliberately
NOT shared: extracting/validating a reply's own fields — that differs in shape per stage (a
repro command vs. a patch vs. a bool+reason vote) and stays in each stage's own module.
"""

from __future__ import annotations

import sys

from . import llm
from .store.base import Store


def get_item_or_skip(store: Store, repo: str, number: int, *, stage: str) -> dict | None:
    """`store.get_item(repo, number)`, or `None` (logged to stderr as `stage`) if absent — the
    "no candidate to act on" skip every M3 stage applies the same way, not raised."""
    item = store.get_item(repo, number)
    if item is None:
        print(f"{stage}: no KB record for {repo}#{number}", file=sys.stderr)
    return item


def latest_run(runs: list[dict]) -> dict | None:
    """The most recently recorded run in `runs` (by `recorded_at`, missing treated as oldest),
    or `None` if `runs` is empty — one consistent "no run" sentinel, unlike each call site
    picking its own (a stray `{}` fallback in one spot behaves differently from `None` in
    another when a caller forgets to check first)."""
    if not runs:
        return None
    return max(runs, key=lambda r: r.get("recorded_at") or "")


def complete_or_none(prompt: str, schema: dict, *, stage: str, subject: str) -> dict | None:
    """`llm.complete(prompt, json_schema=schema)`, or `None` if the call failed or didn't
    return a dict (logged to stderr as `stage`/`subject`, not raised) — field-level validation
    of the reply is still each caller's own job (see module docstring)."""
    try:
        reply = llm.complete(prompt, json_schema=schema)
    except llm.LLMError as exc:
        print(f"{stage}: LLM call failed for {subject!r}: {exc}", file=sys.stderr)
        return None
    return reply if isinstance(reply, dict) else None


def record_run_best_effort(
    store: Store, record: dict, *, stage: str, repo: str, number: int
) -> None:
    """`store.record_run(record)`, logging (not raising) on failure — an already-completed
    MI250 result must never be discarded over a mere KB-write hiccup."""
    try:
        store.record_run(record)
    except Exception as exc:
        print(f"{stage}: failed to record run for {repo}#{number}: {exc}", file=sys.stderr)
