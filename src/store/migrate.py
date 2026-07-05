"""One-shot jsonl → Firestore migration (T0.6.3/T0.6.4) — ``python -m src.store.migrate``.

Copies the M0 on-disk JSONL knowledge base into the M0.6 Firestore backend: every item across
every repo JSONL file in a data directory, the whole state cursor map, and every recorded run
(every ``stage="..."`` record — collection, verify, self-review, PR authoring/quality, the human
gate, etc.), all upserted/written into :class:`~src.store.firestore_store.FirestoreStore`.

Safe to re-run against the same source (e.g. to pick up items/runs collected after an earlier
migration pass): items are upserted by ``(repo, number)`` (Firestore ``set``, not ``create``),
state keys are simply overwritten, and runs — which have **no natural identity key** on either
backend (:meth:`~src.store.jsonl_store.JsonlStore.record_run`/
:meth:`~src.store.firestore_store.FirestoreStore.record_run` are both pure appends) — are
de-duplicated by exact content match against what the destination already has for that
``(repo, number, stage)`` before writing (see :func:`migrate`). This makes re-running a true
no-op for anything already migrated unchanged, closing the gap T0.6.4 exists to fix: a cutover
must not silently lose a candidate's pipeline history (including a `stage="gate"` record
showing an already-submitted `pr_url` — see T3.10.6's idempotency check, which reads exactly
that history and would otherwise let a migrated candidate be resubmitted with no warning), and
an operator re-running the script by habit (as items/state already invite) must not silently
duplicate it either.

Content-equality dedup is not a real identity key, though: if the *same* logical event were ever
re-recorded with even one different field (shouldn't happen — `JsonlStore`/`FirestoreStore` are
both append-only, never mutating an existing run), it would be treated as a new, distinct run.
Not a currently-triggered gap, just a known limit of this heuristic versus a true key.

Items still have the one pre-existing caveat: re-running against a *stale* JSONL snapshot after
other agents have already written new fields directly to Firestore (e.g. post-cutover to
``STORE=firestore``) would have `upsert_items`' merge-on-upsert semantics (T1.10) preserve those
Firestore-only fields instead of resetting the doc to exactly mirror the stale source. Not a
currently-triggered bug (nothing else in this codebase does that), but this script is a one-shot
cutover tool, not a repeatable sync — don't re-run it against an out-of-date source after
Firestore has moved on.

The source is always a :class:`~src.store.jsonl_store.JsonlStore` and the destination is
always :class:`~src.store.firestore_store.FirestoreStore`: this script's whole purpose is that
one direction, so unlike :func:`src.store.get_store` it does not branch on env ``STORE``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import config
from .firestore_store import FirestoreStore
from .jsonl_store import JsonlStore, load_state


def migrate(source_dir: Path, *, firestore_project: str | None) -> dict:
    """Migrate every item + state key + run under `source_dir` into Firestore. Returns a summary
    dict.

    `firestore_project` is passed straight to :class:`FirestoreStore` (``None`` uses ambient
    credentials' default project, same as :func:`~src.store.get_store`). Reads items via
    :meth:`JsonlStore.query` (no filter — every repo file in `source_dir`) rather than
    iterating :data:`config.REPOS`, so a repo no longer tracked still gets migrated if its
    JSONL file is still on disk. State is read via the on-disk ``state.json`` map directly
    (the abstract ``Store`` interface has no "list all state keys" method) for the same reason.

    Runs are read via :meth:`JsonlStore.list_runs` with no filters — every run across every
    repo, candidate, and stage in `source_dir`'s ``runs.jsonl`` — then written one at a time,
    skipping any run that's an exact content match for one the destination already has under
    the same `(repo, number, stage)` (see module docstring: `record_run` has no identity key,
    so this dedup is by value, not by ID). `dest.list_runs` is only queried once per distinct
    `(repo, number, stage)` scope encountered, not once per run, to keep a re-run's cost
    proportional to the number of distinct scopes rather than the total run count.
    """
    source = JsonlStore(source_dir)
    dest = FirestoreStore(project=firestore_project)

    items = source.query()
    item_totals = dest.upsert_items(items)  # upsert_items([]) already returns {} on its own

    state = load_state(source_dir / "state.json")
    for key, value in state.items():
        dest.set_state(key, value)

    runs = source.list_runs()
    runs_migrated = 0
    existing_by_scope: dict[tuple, list[dict]] = {}
    for run in runs:
        scope = (run.get("repo"), run.get("number"), run.get("stage"))
        if scope not in existing_by_scope:
            existing_by_scope[scope] = dest.list_runs(
                repo=scope[0], number=scope[1], stage=scope[2]
            )
        if run in existing_by_scope[scope]:
            continue
        # Best-effort, like every other record_run call site in this codebase (stages.py's own
        # rationale: an already-completed result must never be discarded over a mere KB-write
        # hiccup) -- one oversized or malformed run (e.g. exceeding Firestore's per-document
        # size limit) must not abort the whole migration and lose every run after it. Unlike
        # `stages.record_run_best_effort`, this tracks success explicitly: a run that failed to
        # write must not be counted as migrated or added to the dedup cache (it should be
        # retried, not skipped, on a subsequent run of this script).
        try:
            dest.record_run(run)
        except Exception as exc:
            print(f"migrate: failed to record run {scope}: {exc}", file=sys.stderr)
            continue
        existing_by_scope[scope].append(run)
        runs_migrated += 1

    return {
        "items_migrated": len(items),
        "item_totals": item_totals,
        "state_keys_migrated": len(state),
        "runs_migrated": runs_migrated,
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
