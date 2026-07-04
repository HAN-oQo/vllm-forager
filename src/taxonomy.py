"""Versioned taxonomy schema (T1.2, hierarchical since T1.5.1).

Later stages (T1.4 Analyst classifies items into it, T2.3 Curator proposes/retires
categories) need one shared, evolving definition of "what buckets does an item belong to."
Rather than mutate a single taxonomy in place, each change creates a new **immutable**
version (``taxonomy@vN``) and moves an **active pointer** to it — so grading (T2.2) and
history/audit questions ("what did the taxonomy look like when this item was classified?")
can always retrieve an older version, never just the latest.

T1.5.1: a category is now a **path** — ``("ROCm/AMD", "DeepSeek-V4", "performance",
"attention")`` — not a single flat name, so the report/dashboard can nest
대(大) → 소 → 소소 → PRs instead of one flat bucket per item. A plain string is still accepted
everywhere a path is (:func:`create_taxonomy`, :func:`add_category`) and read back as a
depth-1 path, e.g. ``"rocm-build"`` → ``("rocm-build",)`` — this is what lets a
taxonomy record written before T1.5.1 (and any single-level category added after it) keep
working unchanged. :attr:`Taxonomy.labels` renders each path as one flattened string (joined
with `` > ``) for callers (T1.4's Analyst, until T1.5.2 teaches it to classify per-level) that
still want one flat label per category rather than a path.

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
from collections.abc import Sequence
from dataclasses import dataclass

from .store.base import Store

_VERSION_KEY_PREFIX = "taxonomy@"
_ACTIVE_KEY = "taxonomy_active_version"

CategoryPath = tuple[str, ...]


class TaxonomyError(RuntimeError):
    """Any taxonomy failure: no taxonomy exists yet, an unknown/corrupt version, or a
    duplicate category."""


def _normalize_path(entry: str | Sequence[str]) -> CategoryPath:
    """A single string is a depth-1 path; a sequence of strings is a multi-level path."""
    return (entry,) if isinstance(entry, str) else tuple(entry)


def _casefold_path(path: CategoryPath) -> CategoryPath:
    """Per-level ``.strip().casefold()`` — how two paths are compared for dedup."""
    return tuple(level.strip().casefold() for level in path)


@dataclass(frozen=True)
class Taxonomy:
    """One immutable taxonomy version: its number and ordered category paths.

    ``categories`` is a ``tuple`` of ``CategoryPath`` (itself a ``tuple[str, ...]``), not a
    ``list``, so the "immutable" claim is real — ``frozen=True`` alone only blocks
    reassigning the attribute, not mutating a list it points to.
    """

    version: int
    categories: tuple[CategoryPath, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        """Each category path flattened to one display/matching string, e.g.
        ``"ROCm/AMD > DeepSeek-V4 > performance"``. For depth-1 paths (every category created
        before T1.5.1, and any single-level one added since) this is just the flat name — so a
        caller that only wants one flat label per category (T1.4's Analyst, until T1.5.2
        teaches it to classify per-level) keeps working unchanged against a now-path-capable
        taxonomy.
        """
        return tuple(" > ".join(path) for path in self.categories)

    def to_json(self) -> str:
        """Serialize for storage in the KB's state map."""
        return json.dumps(
            {"version": self.version, "categories": [list(path) for path in self.categories]}
        )

    @staticmethod
    def from_json(raw: str) -> Taxonomy:
        """Deserialize a value previously produced by :meth:`to_json`.

        Each entry in the stored ``categories`` list is either a JSON string (a taxonomy
        written before T1.5.1, or a single-level category added since) or a JSON list (a
        multi-level path) — :func:`_normalize_path` reads both into a ``CategoryPath``.

        Raises:
            TaxonomyError: `raw` isn't valid JSON, or isn't shaped like a taxonomy record.
        """
        try:
            data = json.loads(raw)
            categories = tuple(_normalize_path(entry) for entry in data["categories"])
            return Taxonomy(version=data["version"], categories=categories)
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


def create_taxonomy(store: Store, categories: Sequence[str | Sequence[str]]) -> Taxonomy:
    """Create version 1 of the taxonomy and make it active.

    Each entry in `categories` is a single string (a depth-1 category, e.g. ``"rocm-build"``)
    or a sequence of strings (a multi-level path, e.g. ``["ROCm/AMD", "DeepSeek-V4"]``).

    Raises:
        TaxonomyError: a taxonomy already exists — evolve it with :func:`add_category`
            instead of creating a second v1.
    """
    if store.get_state(_ACTIVE_KEY) is not None:
        raise TaxonomyError("a taxonomy already exists; use add_category() to evolve it")
    taxonomy = Taxonomy(version=1, categories=tuple(_normalize_path(entry) for entry in categories))
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


def add_category(store: Store, path: str | Sequence[str]) -> Taxonomy:
    """Evolve the taxonomy: create version N+1 with `path` appended, and activate it.

    `path` is a single string (a depth-1 category) or a sequence of strings (a multi-level
    path) — see :class:`Taxonomy`. The prior version is left untouched in the KB
    (append-only), so anything that recorded "classified under taxonomy v2" can still look v2
    up after v3 becomes active.

    Raises:
        TaxonomyError: no taxonomy exists yet, or `path` is already an active category
            (compared level-by-level, case- and whitespace-insensitively, since a
            clustering/LLM-driven proposal — T2.3's Curator — can easily emit "ROCm" vs "rocm"
            for one concept).
    """
    current = get_active(store)
    new_path = _normalize_path(path)
    existing = {_casefold_path(p) for p in current.categories}
    if _casefold_path(new_path) in existing:
        raise TaxonomyError(f"{list(new_path)!r} is already a category in v{current.version}")
    updated = Taxonomy(version=current.version + 1, categories=(*current.categories, new_path))
    store.set_state(_version_key(updated.version), updated.to_json())
    store.set_state(_ACTIVE_KEY, str(updated.version))
    return updated
