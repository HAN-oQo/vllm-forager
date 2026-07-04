"""Tests for the versioned taxonomy schema (T1.2) — offline & deterministic.

Uses a real :class:`JsonlStore` on ``tmp_path`` (not a mock) since the module's whole job is
to round-trip through the generic ``Store.get_state``/``set_state`` contract correctly.
"""

from pathlib import Path

import pytest

from src import taxonomy
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m1


@pytest.fixture
def store(tmp_path: Path) -> JsonlStore:
    return JsonlStore(tmp_path)


def test_create_taxonomy_is_v1_and_active(store: JsonlStore) -> None:
    v1 = taxonomy.create_taxonomy(store, ["rocm-build", "performance"])
    assert v1.version == 1
    assert v1.categories == ["rocm-build", "performance"]
    assert taxonomy.get_active(store) == v1


def test_create_taxonomy_twice_raises(store: JsonlStore) -> None:
    taxonomy.create_taxonomy(store, ["rocm-build"])
    with pytest.raises(taxonomy.TaxonomyError, match="already exists"):
        taxonomy.create_taxonomy(store, ["other"])


def test_get_active_before_create_raises(store: JsonlStore) -> None:
    with pytest.raises(taxonomy.TaxonomyError, match="no taxonomy exists"):
        taxonomy.get_active(store)


def test_get_taxonomy_unknown_version_raises(store: JsonlStore) -> None:
    taxonomy.create_taxonomy(store, ["rocm-build"])
    with pytest.raises(taxonomy.TaxonomyError, match="version 2 not found"):
        taxonomy.get_taxonomy(store, 2)


def test_add_category_bumps_version_both_retrievable_active_is_latest(
    store: JsonlStore,
) -> None:
    """The DEVPLAN's named scenario: v1 -> add category -> v2; both retrievable; active = v2."""
    v1 = taxonomy.create_taxonomy(store, ["rocm-build"])
    v2 = taxonomy.add_category(store, "quantization")

    assert v2.version == 2
    assert v2.categories == ["rocm-build", "quantization"]

    # both versions still retrievable by number...
    assert taxonomy.get_taxonomy(store, 1) == v1
    assert taxonomy.get_taxonomy(store, 2) == v2
    # ...and active() returns the latest, not the first
    assert taxonomy.get_active(store) == v2


def test_add_category_duplicate_raises(store: JsonlStore) -> None:
    taxonomy.create_taxonomy(store, ["rocm-build"])
    with pytest.raises(taxonomy.TaxonomyError, match="already a category"):
        taxonomy.add_category(store, "rocm-build")


def test_add_category_before_create_raises(store: JsonlStore) -> None:
    with pytest.raises(taxonomy.TaxonomyError, match="no taxonomy exists"):
        taxonomy.add_category(store, "rocm-build")


def test_taxonomy_json_roundtrip() -> None:
    original = taxonomy.Taxonomy(version=3, categories=["a", "b"])
    assert taxonomy.Taxonomy.from_json(original.to_json()) == original
