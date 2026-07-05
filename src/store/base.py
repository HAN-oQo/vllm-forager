"""Storage interface — the contract every knowledge-base backend implements.

The collector (and later the intelligence/contribution agents) read and write the KB
through this interface only, so swapping the M0 on-disk JSONL for Firestore (M0.6) is a
backend change, not a caller change.

Item identity is **(repo, number)**: every stored record carries a `repo` slug
("owner/name") and an integer `number` (the GitHub issue/PR number). Backends may key
however they like internally (the planned Firestore store keys docs as "repo#number");
callers only ever pass `repo` + `number`.

State is a small key → string map kept separate from items — the incremental collection
cursor lives here, keyed by repo slug.

Runs (T3.2+) are a third, append-only kind of record — one MI250 repro/build/verify attempt
each — kept separate from items because they don't have items' (repo, number)-keyed
identity: several runs can (and do) exist for the same candidate over time, so a run is never
merged into a prior one the way :func:`merge_record` merges item upserts. A run record's shape
beyond ``repo``/``number`` (used only for filtering, not identity) is caller-defined — this
module doesn't fix a schema, the same way ``upsert_items`` doesn't fix one for items.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


def merge_record(old: dict | None, new: dict) -> dict:
    """The one shallow-merge every backend's ``upsert_items`` must apply: `new`'s fields win.

    A field present on `old` but absent from `new` is preserved; any field `new` does specify
    overwrites `old`'s value for that key (T1.10). `old=None` (no existing record) is just
    `new` itself. This is intentionally **shallow** — a field whose value is itself a nested
    dict is replaced wholesale, not recursively merged, matching plain Python dict-update
    semantics; a backend whose native merge is deeper (e.g. Firestore's ``set(merge=True)``
    recursively merges nested maps) must constrain itself to this shallow contract, not the
    other way around, so every backend behaves identically regardless of field shape.

    No opt-out exists (a caller can't currently force a full replace or delete a single field)
    — a known limitation, not a bug: nothing in this codebase needs it yet (see analyst.py/
    forecaster.py), but a future caller that does (e.g. resetting a field to force
    reclassification) will need a new primitive, not a workaround here.
    """
    return {**(old or {}), **new}


class Store(ABC):
    """Abstract KB store: a set of items (issues/PRs) plus a small state key-value map."""

    @abstractmethod
    def upsert_items(self, items: list[dict]) -> dict[str, int]:
        """Insert-or-update `items`, matched on their (repo, number).

        An item that already has a stored record is updated by **merging** the given fields
        onto it (see :func:`merge_record`), not replacing it wholesale: a field present on the
        existing record but absent from the new one is preserved, while any field the new
        record does specify overwrites the old value. (T1.10 — before this, a caller that
        writes a *partial* record, or a collector re-normalization that never carries forward a
        field another stage added, would silently erase that field.) Items may span multiple
        repos; the backend routes each to the right place. Returns ``{repo: total_item_count}``
        for **each repo touched** — the post-upsert total for that repo, so a caller (e.g. the
        collector's progress log) never needs a second full read to report it. Repos with no
        items in the batch are absent from the map; an empty `items` returns ``{}``.
        """

    @abstractmethod
    def get_item(self, repo: str, number: int) -> dict | None:
        """Return the stored record for (repo, number), or None if absent."""

    @abstractmethod
    def query(
        self,
        *,
        repo: str | None = None,
        label: str | None = None,
        state: str | None = None,
        type: str | None = None,
    ) -> list[dict]:
        """Return items matching **every** provided filter (AND); omitted filters don't constrain.

        - ``repo``  — exact repo slug ("owner/name")
        - ``label`` — membership in the record's ``labels`` list
        - ``state`` — exact match on ``state`` ("open" / "closed")
        - ``type``  — exact match on ``type`` ("issue" / "pr")

        Results are ordered by ``updated_at`` ascending.
        """

    @abstractmethod
    def get_state(self, key: str) -> str | None:
        """Return the state value for `key`, or None if unset."""

    @abstractmethod
    def set_state(self, key: str, value: str) -> None:
        """Set (and immediately persist) the state value for `key`."""

    @abstractmethod
    def record_run(self, run: dict) -> None:
        """Append `run` to the KB's `runs` collection (see module docstring) — never merged
        into a prior record, unlike :meth:`upsert_items`; each call adds a new one."""

    @abstractmethod
    def list_runs(
        self,
        *,
        repo: str | None = None,
        number: int | None = None,
        stage: str | None = None,
    ) -> list[dict]:
        """Return every recorded run matching the given filters (AND; omitted filters don't
        constrain, and a record simply missing a filtered-on key never matches it).

        Best-effort insertion order: exact on :class:`~src.store.jsonl_store.JsonlStore` (a
        strictly append-only file), but **not guaranteed** on
        :class:`~src.store.firestore_store.FirestoreStore` — its auto-generated document IDs
        carry no reliable chronological ordering, and no run record is required to carry an
        orderable timestamp field a query could sort by. A caller that needs a reliable "most
        recent run" must sort the returned list itself by whatever timestamp field its own run
        records carry (e.g. :attr:`~src.repro.ReproResult.recorded_at`).
        """
