"""``python -m dashboard`` — serve the thin read-only M1 dashboard.

Store/reports-dir selection mirrors :mod:`src.report`'s CLI: with ``--data-dir``, always reads
a :class:`~src.store.jsonl_store.JsonlStore` at that path; without it, uses
:func:`src.store.get_store` (``STORE=jsonl|firestore``) so ``STORE=firestore`` serves from
Firestore just like the collector/reporter do.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.store import get_store
from src.store.base import Store
from src.store.jsonl_store import JsonlStore

from .server import serve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dashboard",
        description="Serve a local read-only dashboard over the vllm-forager KB.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Read a JSONL store at this path instead of the STORE-selected backend.",
    )
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    ap.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    args = ap.parse_args(argv)

    if args.data_dir is not None:
        data_dir = args.data_dir

        def store_factory() -> Store:
            return JsonlStore(data_dir)

    else:
        data_dir = config.DATA_DIR
        store_factory = get_store

    serve(store_factory, data_dir / "reports", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
