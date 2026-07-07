"""Scout CLI (T2.5) — ``python -m src.candidates`` prints the ranked contribution queue.

One command runs the Scout agent (:mod:`src.agents.scout`, T2.5) against the KB: every parity
gap, good-first-issue, and ROCm-reproducible candidate currently in the store, scored on risk/
effort/impact and ranked highest-priority first. Mirrors :mod:`src.grade`'s store-selection
pattern.

Known limitation, not fixed here: like :mod:`~src.agents.curator`'s proposals, this recomputes
the whole queue (one ``llm.complete`` call per candidate) from scratch on every run — no
persistence of what a human already picked up, and no dedup against a near-duplicate seen
before (T2.7's job). Cost grows with the number of open ROCm-relevant/good-first issues in the
KB, not with what's new since the last run -- ``--per-domain-limit`` (T3.19 support) bounds
that cost for a one-off dry run, but doesn't change this per-run-from-scratch design.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .agents import scout
from .analyze import _positive_int
from .store import resolve_store


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the ranked candidate queue, highest-priority first.

    Store selection is :func:`~src.store.resolve_store`'s shared contract: ``--data-dir``
    reads a :class:`~src.store.jsonl_store.JsonlStore` at that explicit path; otherwise
    :func:`~src.store.get_store` picks the backend from env ``STORE``.
    """
    ap = argparse.ArgumentParser(
        prog="python -m src.candidates",
        description="Print the ranked contribution candidate queue (risk/effort/impact).",
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
        "--per-domain-limit",
        type=_positive_int,
        default=None,
        help=(
            "Score at most N good-first-issue/ROCm-reproducible matches per domain "
            "(speech/rl/omni/engine) -- for a KB with hundreds of matches, each a separate "
            "LLM call, this bounds a dry run (e.g. right after a retarget, T3.19) while still "
            "sampling every domain. Parity-gap candidates are never capped. Must be positive."
        ),
    )
    args = ap.parse_args(argv)

    store, _ = resolve_store(args.data_dir)
    candidates = scout.discover_from_store(store, per_domain_limit=args.per_domain_limit)
    print(f"{len(candidates)} candidate(s)")
    for candidate in candidates:
        print(
            f"[{candidate.source}] risk={candidate.risk} effort={candidate.effort} "
            f"impact={candidate.impact} — {candidate.title} ({candidate.evidence})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
