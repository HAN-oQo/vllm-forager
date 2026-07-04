"""One-shot jsonl → Firestore migration (T0.6.3) — ``python -m src.store.migrate``.

Copies the M0 on-disk JSONL knowledge base into the M0.6 Firestore backend: every item across
every repo JSONL file in a data directory, plus the whole state cursor map, upserted into
:class:`~src.store.firestore_store.FirestoreStore`.

Idempotent by construction — items are upserted by ``(repo, number)`` (Firestore ``set``, not
``create``) and state keys are simply overwritten — so re-running (e.g. to pick up items
collected after an earlier migration pass) is safe and just re-applies the same data.

The source is always a :class:`~src.store.jsonl_store.JsonlStore` and the destination is
always :class:`~src.store.firestore_store.FirestoreStore`: this script's whole purpose is that
one direction, so unlike :func:`src.store.get_store` it does not branch on env ``STORE``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import config
from .firestore_store import FirestoreStore
from .jsonl_store import JsonlStore, load_state


def migrate(source_dir: Path, *, firestore_project: str | None) -> dict:
    """Migrate every item + state key under `source_dir` into Firestore. Returns a summary dict.

    `firestore_project` is passed straight to :class:`FirestoreStore` (``None`` uses ambient
    credentials' default project, same as :func:`~src.store.get_store`). Reads items via
    :meth:`JsonlStore.query` (no filter — every repo file in `source_dir`) rather than
    iterating :data:`config.REPOS`, so a repo no longer tracked still gets migrated if its
    JSONL file is still on disk. State is read via the on-disk ``state.json`` map directly
    (the abstract ``Store`` interface has no "list all state keys" method) for the same reason.
    """
    source = JsonlStore(source_dir)
    dest = FirestoreStore(project=firestore_project)

    items = source.query()
    item_totals = dest.upsert_items(items) if items else {}

    state = load_state(source_dir / "state.json")
    for key, value in state.items():
        dest.set_state(key, value)

    return {
        "items_migrated": len(items),
        "item_totals": item_totals,
        "state_keys_migrated": len(state),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: migrate a JSONL data directory into Firestore and print a summary."""
    ap = argparse.ArgumentParser(
        prog="python -m src.store.migrate",
        description="One-shot migration of the JSONL knowledge base into Firestore.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Source JSONL data directory to migrate from (default: config.DATA_DIR).",
    )
    args = ap.parse_args(argv)

    summary = migrate(args.data_dir, firestore_project=config.FIRESTORE_PROJECT)
    print(
        f"migrated {summary['items_migrated']} items across {len(summary['item_totals'])} "
        f"repos, {summary['state_keys_migrated']} state keys"
    )
    for repo, total in sorted(summary["item_totals"].items()):
        print(f"  {repo}: {total} total in Firestore")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
