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


def _positive_int(raw: str) -> int:
    """`argparse` type for `--per-domain-limit`: a clean CLI usage error for a non-positive
    value, rather than a raw `ValueError` traceback from :func:`~src.agents.analyst.analyze_store`
    — a `0`/negative limit used to silently classify nothing and print the exact same
    "classified 0 item(s)" a genuinely empty backlog would, with no signal the flag itself was
    the problem (caught on T3.19's own code review)."""
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive int, got {raw!r}")
    return value


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
        "--per-domain-limit",
        type=_positive_int,
        default=None,
        help=(
            "Classify at most N pending items per domain (speech/rl/omni/engine), giving every "
            "domain an equal-sized slice regardless of how many repos are tracked under it -- "
            "for a bounded dry run over a KB with many pending items (e.g. right after a "
            "retarget, T3.19). Must be a positive int."
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help=(
            "Classify at most N same-domain pending items per llm.complete call instead of one "
            "call per item (T4.11) -- cuts classification LLM spend at the source. Default: "
            "unbatched, one call per item, unchanged from before this flag existed. Must be a "
            "positive int."
        ),
    )
    args = ap.parse_args(argv)

    store, _ = resolve_store(args.data_dir)
    classified = analyst.analyze_store(
        store, per_domain_limit=args.per_domain_limit, batch_size=args.batch_size
    )
    print(f"classified {len(classified)} item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
