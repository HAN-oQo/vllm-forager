"""Versioned taxonomy schema (T1.2).

Later stages (T1.4 Analyst classifies items into it, T2.3 Curator proposes/retires
categories) need one shared, evolving definition of "what buckets does an item belong to."
Rather than mutate a single taxonomy in place, each change creates a new **immutable**
version (``taxonomy@vN``) and moves an **active pointer** to it — so grading (T2.2) and
history/audit questions ("what did the taxonomy look like when this item was classified?")
can always retrieve an older version, never just the latest.

Storage: taxonomy versions are small enough to live in the KB's generic state map
(:meth:`~src.store.base.Store.get_state` / ``set_state``), keyed ``taxonomy@1``,
``taxonomy@2``, ... plus one ``taxonomy_active_version`` pointer — no new store method
needed.

Known limitation (not fixed here): :func:`create_taxonomy`/:func:`add_category` are a
read-then-write over a plain key-value store with no locking or compare-and-swap, so two
concurrent callers evolving the taxonomy at once can race (one addition silently lost).
Real fixes belongs at the Store layer — see the still-open ``T4.3 Locking / idempotency``
DEVPLAN todo — rather than reinvented per caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .store.base import Store

_VERSION_KEY_PREFIX = "taxonomy@"
_ACTIVE_KEY = "taxonomy_active_version"


class TaxonomyError(RuntimeError):
    """Any taxonomy failure: no taxonomy exists yet, an unknown/corrupt version, or a
    duplicate category."""


@dataclass(frozen=True)
class Taxonomy:
    """One immutable taxonomy version: its number and ordered category names.

    ``categories`` is a ``tuple`` (not a ``list``) so the "immutable" claim is real —
    ``frozen=True`` alone only blocks reassigning the attribute, not mutating a list it
    points to.
    """

    version: int
    categories: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize for storage in the KB's state map."""
        return json.dumps({"version": self.version, "categories": list(self.categories)})

    @staticmethod
    def from_json(raw: str) -> Taxonomy:
        """Deserialize a value previously produced by :meth:`to_json`.

        Raises:
            TaxonomyError: `raw` isn't valid JSON, or isn't shaped like a taxonomy record.
        """
        try:
            data = json.loads(raw)
            return Taxonomy(version=data["version"], categories=tuple(data["categories"]))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise TaxonomyError(f"corrupt taxonomy record: {exc}") from exc


def _version_key(version: int) -> str:
    return f"{_VERSION_KEY_PREFIX}{version}"


def _parse_version(raw: str) -> int:
    """Parse the active-version pointer; raise :class:`TaxonomyError` (not ValueError)."""
    try:
        return int(raw)
    except ValueError as exc:
        raise TaxonomyError(f"corrupt active-version pointer: {raw!r}") from exc


def create_taxonomy(store: Store, categories: list[str]) -> Taxonomy:
    """Create version 1 of the taxonomy and make it active.

    Raises:
        TaxonomyError: a taxonomy already exists — evolve it with :func:`add_category`
            instead of creating a second v1.
    """
    if store.get_state(_ACTIVE_KEY) is not None:
        raise TaxonomyError("a taxonomy already exists; use add_category() to evolve it")
    taxonomy = Taxonomy(version=1, categories=tuple(categories))
    store.set_state(_version_key(1), taxonomy.to_json())
    store.set_state(_ACTIVE_KEY, str(taxonomy.version))
    return taxonomy


def get_taxonomy(store: Store, version: int) -> Taxonomy:
    """Return a specific, immutable taxonomy version — older versions stay retrievable.

    Raises:
        TaxonomyError: `version` doesn't exist, or the stored record's own version field
            doesn't match `version` (KB corruption — the two are stored redundantly).
    """
    raw = store.get_state(_version_key(version))
    if raw is None:
        raise TaxonomyError(f"taxonomy version {version} not found")
    taxonomy = Taxonomy.from_json(raw)
    if taxonomy.version != version:
        raise TaxonomyError(
            f"taxonomy@{version} record claims version {taxonomy.version} — KB corruption"
        )
    return taxonomy


def get_active(store: Store) -> Taxonomy:
    """Return the currently active (latest) taxonomy version.

    Raises:
        TaxonomyError: no taxonomy has been created yet, or the active pointer is corrupt.
    """
    raw = store.get_state(_ACTIVE_KEY)
    if raw is None:
        raise TaxonomyError("no taxonomy exists yet; call create_taxonomy() first")
    return get_taxonomy(store, _parse_version(raw))


def add_category(store: Store, name: str) -> Taxonomy:
    """Evolve the taxonomy: create version N+1 with `name` appended, and activate it.

    The prior version is left untouched in the KB (append-only), so anything that recorded
    "classified under taxonomy v2" can still look v2 up after v3 becomes active.

    Raises:
        TaxonomyError: no taxonomy exists yet, or `name` is already an active category
            (compared case- and whitespace-insensitively, since a clustering/LLM-driven
            proposal — T2.3's Curator — can easily emit "ROCm" vs "rocm" for one concept).
    """
    current = get_active(store)
    existing = {c.strip().casefold() for c in current.categories}
    if name.strip().casefold() in existing:
        raise TaxonomyError(f"{name!r} is already a category in v{current.version}")
    updated = Taxonomy(version=current.version + 1, categories=(*current.categories, name))
    store.set_state(_version_key(updated.version), updated.to_json())
    store.set_state(_ACTIVE_KEY, str(updated.version))
    return updated
