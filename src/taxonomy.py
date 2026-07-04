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
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .store.base import Store

_VERSION_KEY_PREFIX = "taxonomy@"
_ACTIVE_KEY = "taxonomy_active_version"


class TaxonomyError(RuntimeError):
    """Any taxonomy failure: no taxonomy exists yet, an unknown version, or a duplicate category."""


@dataclass(frozen=True)
class Taxonomy:
    """One immutable taxonomy version: its number and ordered category names."""

    version: int
    categories: list[str]

    def to_json(self) -> str:
        """Serialize for storage in the KB's state map."""
        return json.dumps({"version": self.version, "categories": self.categories})

    @staticmethod
    def from_json(raw: str) -> Taxonomy:
        """Deserialize a value previously produced by :meth:`to_json`."""
        data = json.loads(raw)
        return Taxonomy(version=data["version"], categories=list(data["categories"]))


def _version_key(version: int) -> str:
    return f"{_VERSION_KEY_PREFIX}{version}"


def create_taxonomy(store: Store, categories: list[str]) -> Taxonomy:
    """Create version 1 of the taxonomy and make it active.

    Raises:
        TaxonomyError: a taxonomy already exists — evolve it with :func:`add_category`
            instead of creating a second v1.
    """
    if store.get_state(_ACTIVE_KEY) is not None:
        raise TaxonomyError("a taxonomy already exists; use add_category() to evolve it")
    taxonomy = Taxonomy(version=1, categories=list(categories))
    store.set_state(_version_key(1), taxonomy.to_json())
    store.set_state(_ACTIVE_KEY, str(taxonomy.version))
    return taxonomy


def get_taxonomy(store: Store, version: int) -> Taxonomy:
    """Return a specific, immutable taxonomy version — older versions stay retrievable.

    Raises:
        TaxonomyError: `version` doesn't exist.
    """
    raw = store.get_state(_version_key(version))
    if raw is None:
        raise TaxonomyError(f"taxonomy version {version} not found")
    return Taxonomy.from_json(raw)


def get_active(store: Store) -> Taxonomy:
    """Return the currently active (latest) taxonomy version.

    Raises:
        TaxonomyError: no taxonomy has been created yet.
    """
    raw = store.get_state(_ACTIVE_KEY)
    if raw is None:
        raise TaxonomyError("no taxonomy exists yet; call create_taxonomy() first")
    return get_taxonomy(store, int(raw))


def add_category(store: Store, name: str) -> Taxonomy:
    """Evolve the taxonomy: create version N+1 with `name` appended, and activate it.

    The prior version is left untouched in the KB (append-only), so anything that recorded
    "classified under taxonomy v2" can still look v2 up after v3 becomes active.

    Raises:
        TaxonomyError: no taxonomy exists yet, or `name` is already an active category.
    """
    current = get_active(store)
    if name in current.categories:
        raise TaxonomyError(f"{name!r} is already a category in v{current.version}")
    updated = Taxonomy(version=current.version + 1, categories=[*current.categories, name])
    store.set_state(_version_key(updated.version), updated.to_json())
    store.set_state(_ACTIVE_KEY, str(updated.version))
    return updated
