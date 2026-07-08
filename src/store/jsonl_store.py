"""JSONL-backed :class:`~src.store.base.Store` — the M0 default.

On-disk layout under ``data_dir``:

    {owner}__{repo}.jsonl   one JSON record per line, upserted by ``number`` and sorted by
                            ``updated_at`` ascending (stable, greppable, diff-friendly).
    state.json              a single ``{key: value}`` JSON map (the per-repo cursor).

This preserves the exact format the M0 collector already wrote, so existing data files and
``src/stats.py`` (which reads the JSONL directly) keep working unchanged.

The module-level ``merge_jsonl`` / ``load_state`` / ``save_state`` helpers are the single
source of truth for the on-disk format; the collector's back-compat shims delegate to them.
Writes are **atomic** (temp file + rename) and reads **tolerate a corrupt/partial line**
(e.g. from an interrupted write) by skipping it — the robustness landed for the collector
in PR #11 and is preserved here so the store cannot regress it.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from .base import Store, merge_record

# --------------------------------------------------------------------- file-level helpers


def _iter_jsonl_records(path: Path) -> Iterator[tuple[int, dict]]:
    """Yield ``(lineno, record)`` for every parsed JSON line in `path` (nothing if absent).

    A corrupt/partial line (as an interrupted write can leave) is skipped with a warning
    rather than aborting the read, so a damaged file self-heals on the next rewrite — the one
    JSONL-parsing loop shared by :func:`_read_items` (items) and
    :meth:`JsonlStore.list_runs` (runs), so this recovery policy only needs to be right once.

    Splits on ``"\n"`` only — the delimiter used when writing. ``str.splitlines()`` also
    breaks on U+2028/U+2029/U+0085 etc.; because records are written with
    ``ensure_ascii=False``, those characters appear literally inside JSON string bodies and
    would otherwise shatter one record into unparseable fragments.
    """
    if not path.exists():
        return
    for lineno, line in enumerate(path.read_text().split("\n"), 1):
        if not line.strip():
            continue
        try:
            yield lineno, json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"  !! {path.name}:{lineno} skipping corrupt line ({exc})", file=sys.stderr)


def _read_items(path: Path) -> dict[int, dict]:
    """Read a repo JSONL into a ``{number: record}`` map (empty if the file is absent)."""
    items: dict[int, dict] = {}
    for lineno, rec in _iter_jsonl_records(path):
        # A syntactically-valid line without `number` can't be keyed — skip it too,
        # rather than let one bad record KeyError-abort the whole read.
        if "number" not in rec:
            print(f"  !! {path.name}:{lineno} skipping line with no 'number'", file=sys.stderr)
            continue
        items[rec["number"]] = rec
    return items


def _write_items(path: Path, items: dict[int, dict]) -> None:
    """Write a ``{number: record}`` map as JSONL, ordered by ``updated_at`` ascending.

    Writes to a temp file then renames — atomic on POSIX, so an interrupted write can never
    leave a half-written (corrupt) file in place.
    """
    ordered = sorted(items.values(), key=lambda r: r.get("updated_at") or "")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ordered) + "\n")
    tmp.replace(path)


def merge_jsonl(path: Path, records: list[dict]) -> int:
    """Upsert `records` (by ``number``) into the JSONL at `path`; return the total count.

    Each record is **merged** onto whatever's already stored for its ``number`` (T1.10) —
    fields the record doesn't mention are preserved from the existing one, fields it does
    mention overwrite the old value. Two records for the same ``number`` within one call merge
    in order (last one's fields win on overlap), same as merging one at a time against disk.
    """
    existing = _read_items(path)
    for rec in records:
        existing[rec["number"]] = merge_record(existing.get(rec["number"]), rec)
    _write_items(path, existing)
    return len(existing)


def load_state(state_path: Path) -> dict:
    """Load the state map from `state_path` (``{}`` if the file is absent)."""
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {}


def save_state(state_path: Path, state: dict) -> None:
    """Persist the whole state map to `state_path` (creating the parent dir if needed).

    Written atomically (temp file + rename), like the item files, so an interrupted write
    can't leave a truncated ``state.json`` that would crash the next run's ``load_state``.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)


# ----------------------------------------------------------------------------- the store


class JsonlStore(Store):
    """Store items as per-repo JSONL files and state as a single ``state.json`` map."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.state_path = self.data_dir / "state.json"
        self.runs_path = self.data_dir / "runs.jsonl"

    def _path_for(self, repo: str) -> Path:
        """``data_dir/{owner}__{repo}.jsonl`` for a repo slug."""
        return self.data_dir / (repo.replace("/", "__") + ".jsonl")

    # -- items ------------------------------------------------------------------
    def upsert_items(self, items: list[dict]) -> dict[str, int]:
        """Group `items` by repo and upsert each group into its JSONL file.

        Returns ``{repo: post_upsert_total}`` from each ``merge_jsonl`` call, so the caller
        gets the per-repo totals without re-reading the (potentially large) files.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        by_repo: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            by_repo[it["repo"]].append(it)
        return {repo: merge_jsonl(self._path_for(repo), recs) for repo, recs in by_repo.items()}

    def get_item(self, repo: str, number: int) -> dict | None:
        return _read_items(self._path_for(repo)).get(number)

    def query(
        self,
        *,
        repo: str | None = None,
        label: str | None = None,
        state: str | None = None,
        type: str | None = None,
    ) -> list[dict]:
        # One repo -> read its file; otherwise scan every repo JSONL in the dir. `runs.jsonl`
        # matches the same "*.jsonl" glob (it lives in the same data_dir) but holds run
        # records, not items -- excluded explicitly, or a run with a `number` field (nearly
        # all of them) gets misread as a pseudo-item by `_read_items`. A real, live bug found
        # via T0.6.4's own migration tests: `forager-data/runs.jsonl`'s real `stage="gate"`
        # record was being returned by `query()` and would have been written into Firestore's
        # `items` collection as a fake item by `migrate.py`.
        paths = (
            [self._path_for(repo)]
            if repo is not None
            else sorted(p for p in self.data_dir.glob("*.jsonl") if p != self.runs_path)
        )
        out: list[dict] = []
        for path in paths:
            for rec in _read_items(path).values():
                if label is not None and label not in (rec.get("labels") or []):
                    continue
                if state is not None and rec.get("state") != state:
                    continue
                if type is not None and rec.get("type") != type:
                    continue
                out.append(rec)
        out.sort(key=lambda r: r.get("updated_at") or "")
        return out

    # -- state ------------------------------------------------------------------
    def get_state(self, key: str) -> str | None:
        return load_state(self.state_path).get(key)

    def set_state(self, key: str, value: str) -> None:
        state = load_state(self.state_path)
        state[key] = value
        save_state(self.state_path, state)

    # -- runs -------------------------------------------------------------------
    def record_run(self, run: dict) -> None:
        """Append `run` as one line to ``runs.jsonl`` — never rewrites existing lines, so
        (unlike item writes) this doesn't need the temp-file-plus-rename atomicity dance for a
        *single writer*: a crash mid-append can only corrupt the last, in-flight line, which
        :meth:`list_runs` tolerates the same way item reads tolerate a corrupt item line.

        Not safe against **concurrent** writers, though: two processes appending large records
        (e.g. a big captured MI250 log) around the same time can have their underlying
        multi-syscall writes interleave, corrupting more than just a trailing line. T4.3
        ("Locking / idempotency", `src.locking.run_lock`) mitigates this for `orchestrator.main`
        invocations racing each other, but not project-wide — a standalone `collector.main`/
        `analyze.main`/etc. run (e.g. via `/collect-loop` or `scripts/collect.sh`'s own cron
        entry) takes no lock of its own and can still race an orchestrator tick's internal call
        to the same agent; see `src.locking`'s own docstring for that disclosed gap.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.runs_path.open("a") as f:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")

    def list_runs(
        self,
        *,
        repo: str | None = None,
        number: int | None = None,
        stage: str | None = None,
    ) -> list[dict]:
        out: list[dict] = []
        for _, rec in _iter_jsonl_records(self.runs_path):
            if repo is not None and rec.get("repo") != repo:
                continue
            if number is not None and rec.get("number") != number:
                continue
            if stage is not None and rec.get("stage") != stage:
                continue
            out.append(rec)
        return out
