"""One-shot jsonl → Firestore migration (T0.6.3/T0.6.4) — ``python -m src.store.migrate``.

Copies the M0 on-disk JSONL knowledge base into the M0.6 Firestore backend: every item across
every repo JSONL file in a data directory, the whole state cursor map, and every recorded run
(every ``stage="..."`` record — collection, verify, self-review, PR authoring/quality, the human
gate, etc.), all upserted/written into :class:`~src.store.firestore_store.FirestoreStore`.

**Items and state are idempotent by construction; runs are NOT — this script has two different
re-run safety stories, not one:**

- Items are upserted by ``(repo, number)`` (Firestore ``set``, not ``create``) and state keys are
  simply overwritten, so re-running (e.g. to pick up items collected after an earlier migration
  pass) is safe and just re-applies the same data. This holds because JSONL stays the
  still-growing, authoritative source in that flow: a previously-migrated Firestore doc's fields
  are always a subset of what's now in the (updated) JSONL file, so ``upsert_items``'
  merge-on-upsert semantics (T1.10) make no practical difference. It does NOT hold if this
  one-shot tool is re-run against a *stale* JSONL snapshot after other agents have already
  written new fields directly to Firestore (e.g. post-cutover to ``STORE=firestore``) — merge
  would then preserve those Firestore-only fields instead of resetting the doc to exactly mirror
  the stale source. Not a currently-triggered bug (nothing else in this codebase does that), but
  this script is a one-shot cutover tool, not a repeatable sync for this half either — don't
  re-run it against an out-of-date source after Firestore has moved on.
- Runs have **no natural identity key** to de-duplicate on, on either backend
  (:meth:`~src.store.jsonl_store.JsonlStore.record_run`/
  :meth:`~src.store.firestore_store.FirestoreStore.record_run` are both pure appends — see their
  own docstrings). Re-running this migration duplicates every run record it already copied,
  every time, with no upsert semantics to fall back on. **Run this script exactly once per
  cutover** — T0.6.4 added run migration specifically to close the gap where a cutover silently
  dropped a candidate's entire pipeline history (including a `stage="gate"` record showing an
  already-submitted `pr_url` — see T3.10.6's idempotency check, which reads exactly that history
  and would otherwise let a migrated candidate be resubmitted with no warning), but doing so
  introduced this asymmetric re-run risk; a synthetic dedup key for runs is a larger change to
  `record_run` itself, not fixed here.

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
    """Migrate every item + state key + run under `source_dir` into Firestore. Returns a summary
    dict. See module docstring for why runs (unlike items/state) are NOT safe to migrate twice.

    `firestore_project` is passed straight to :class:`FirestoreStore` (``None`` uses ambient
    credentials' default project, same as :func:`~src.store.get_store`). Reads items via
    :meth:`JsonlStore.query` (no filter — every repo file in `source_dir`) rather than
    iterating :data:`config.REPOS`, so a repo no longer tracked still gets migrated if its
    JSONL file is still on disk. State is read via the on-disk ``state.json`` map directly
    (the abstract ``Store`` interface has no "list all state keys" method) for the same reason.
    Runs are read via :meth:`JsonlStore.list_runs` with no filters — every run across every
    repo, candidate, and stage in `source_dir`'s ``runs.jsonl``.
    """
    source = JsonlStore(source_dir)
    dest = FirestoreStore(project=firestore_project)

    items = source.query()
    item_totals = dest.upsert_items(items)  # upsert_items([]) already returns {} on its own

    state = load_state(source_dir / "state.json")
    for key, value in state.items():
        dest.set_state(key, value)

    runs = source.list_runs()
    for run in runs:
        dest.record_run(run)

    return {
        "items_migrated": len(items),
        "item_totals": item_totals,
        "state_keys_migrated": len(state),
        "runs_migrated": len(runs),
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
        f"repos, {summary['state_keys_migrated']} state keys, {summary['runs_migrated']} runs"
    )
    for repo, total in sorted(summary["item_totals"].items()):
        print(f"  {repo}: {total} total in Firestore")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
