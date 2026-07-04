"""Forecaster CLI (T1.5) — ``python -m src.forecast`` logs predictions for newly classified items.

One command runs the Forecaster agent (:mod:`src.agents.forecaster`, T1.5) against the KB:
every classified item without an existing prediction is forecast and the result is appended
to the KB's prediction log. Mirrors :mod:`src.analyze`'s (T1.4) store-selection pattern.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .agents import forecaster
from .store import resolve_store


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: log new predictions and print how many were made.

    Store selection is :func:`~src.store.resolve_store`'s shared contract: ``--data-dir``
    reads a :class:`~src.store.jsonl_store.JsonlStore` at that explicit path; otherwise
    :func:`~src.store.get_store` picks the backend from env ``STORE``.
    """
    ap = argparse.ArgumentParser(
        prog="python -m src.forecast",
        description="Log calibrated predictions for newly classified items.",
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
    predictions = forecaster.forecast_store(store)
    print(f"logged {len(predictions)} prediction(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
