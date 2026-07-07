"""Analyst CLI (T1.4) — ``python -m src.analyze`` classifies newly collected items.

One command runs the Analyst agent (:mod:`src.agents.analyst`, T1.4) against the KB: every
item without a ``category`` yet is classified into the active taxonomy and the result is
written back to its own record. Mirrors :mod:`src.report`'s (T0.9) store-selection pattern —
``STORE=firestore`` classifies from Firestore just like the collector/report CLIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .agents import analyst
from .store import resolve_store


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: classify pending items and print how many were classified.

    Store selection is :func:`~src.store.resolve_store`'s shared contract: ``--data-dir``
    reads a :class:`~src.store.jsonl_store.JsonlStore` at that explicit path; otherwise
    :func:`~src.store.get_store` picks the backend from env ``STORE``.
    """
    ap = argparse.ArgumentParser(
        prog="python -m src.analyze",
        description="Classify newly collected items into the active taxonomy.",
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
    ap.add_argument(
        "--per-repo-limit",
        type=int,
        default=None,
        help=(
            "Classify at most N pending items per repo, sampling across every tracked repo "
            "instead of exhausting one large repo's backlog first -- for a bounded dry run "
            "over a KB with many pending items (e.g. right after a retarget, T3.19)."
        ),
    )
    args = ap.parse_args(argv)

    store, _ = resolve_store(args.data_dir)
    classified = analyst.analyze_store(store, per_repo_limit=args.per_repo_limit)
    print(f"classified {len(classified)} item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
