"""``python -m dashboard`` — serve the thin read-only M1 dashboard.

Store selection uses :func:`src.store.resolve_store` (T1.11), the same shared contract as
:mod:`src.report`/:mod:`src.analyze`/:mod:`src.forecast`: with ``--data-dir``, always reads a
:class:`~src.store.jsonl_store.JsonlStore` at that path; without it, uses
:func:`src.store.get_store` (``STORE=jsonl|firestore``) so ``STORE=firestore`` serves from
Firestore just like the collector/reporter do. Since T1.5.5, the dashboard reads everything
(the report tree included) through the `Store` — no local ``reports_dir`` dependency remains.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.store import resolve_store

from .server import serve


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and run the dashboard's HTTP server until interrupted.

    `argv` is parsed (defaults to ``sys.argv`` when None). Returns a process exit code: 0 on a
    normal (interrupted) shutdown, 1 if the bind port was already in use (see
    :func:`dashboard.server.serve`).
    """
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

    store, data_dir = resolve_store(args.data_dir)
    return serve(store, host=args.host, port=args.port, data_dir=data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
