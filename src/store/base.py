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
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Store(ABC):
    """Abstract KB store: a set of items (issues/PRs) plus a small state key-value map."""

    @abstractmethod
    def upsert_items(self, items: list[dict]) -> dict[str, int]:
        """Insert-or-update `items`, matched on their (repo, number).

        An item that already has a stored record is updated by **merging** the given fields
        onto it, not replacing it wholesale: a field present on the existing record but absent
        from the new one is preserved, while any field the new record does specify overwrites
        the old value. (T1.10 — before this, a caller that writes a *partial* record, or a
        collector re-normalization that never carries forward a field another stage added,
        would silently erase that field.) Items may span multiple repos; the backend routes
        each to the right place. Returns ``{repo: total_item_count}`` for **each repo
        touched** — the post-upsert total for that repo, so a caller (e.g. the collector's
        progress log) never needs a second full read to report it. Repos with no items in the
        batch are absent from the map; an empty `items` returns ``{}``.
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
