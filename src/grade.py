"""Grader CLI (T2.1) — ``python -m src.grade`` scores matured predictions against reality.

One command runs the Grader agent (:mod:`src.agents.grader`, T2.1) against the KB: every
matured, not-yet-graded prediction is resolved and the result appended to the KB's grade log,
then aggregate precision/recall/Brier over *every* recorded grade (not just this run's new
ones — the self-evolution signal is the running total, not one run's delta) is printed.
Mirrors :mod:`src.forecast`'s (T1.5) store-selection pattern.

T2.2: every run also proposes and applies a policy update
(:func:`~src.agents.policy_update.update_policy_from_grades`) — grading is only a diagnostic
unless it feeds back into the policy driving the next round. Run unconditionally, not gated on
this run having graded anything new: it's idempotent (a no-op re-run never bumps the policy
version — see that function's own docstring) and a backlog of already-graded-but-never-applied
predictions shouldn't have to wait for an unrelated new one to mature before it's ever
reflected. Left to raise loudly if no policy has ever been created yet (a bootstrapping gap to
notice and fix on the very first run, not silently paper over).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .agents import grader, policy_update
from .store import resolve_store


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: grade newly matured predictions and print aggregate metrics.

    Store selection is :func:`~src.store.resolve_store`'s shared contract: ``--data-dir``
    reads a :class:`~src.store.jsonl_store.JsonlStore` at that explicit path; otherwise
    :func:`~src.store.get_store` picks the backend from env ``STORE``.
    """
    ap = argparse.ArgumentParser(
        prog="python -m src.grade",
        description="Resolve matured predictions against reality and score precision/recall/Brier.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Read a JSONL store at this path instead of the STORE-selected backend "
            "(default: config.DATA_DIR, backend from env STORE=jsonl|firestore)."
        ),
    )
    args = ap.parse_args(argv)

    store, _ = resolve_store(args.data_dir)
    new_grades = grader.grade_store(store)
    print(f"graded {len(new_grades)} newly-matured prediction(s)")

    all_grades = grader.list_grades(store)
    metrics = grader.compute_metrics(all_grades)
    print(
        f"overall: n={metrics.n} precision={metrics.precision:.2f} "
        f"recall={metrics.recall:.2f} brier={metrics.brier:.2f}"
    )

    updated = policy_update.update_policy_from_grades(store, all_grades)
    if updated is not None:
        print(f"policy@{updated.version} active (scoring_weights updated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
