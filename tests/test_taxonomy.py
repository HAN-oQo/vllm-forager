"""Tests for the versioned, hierarchical taxonomy schema (T1.2, path-capable since T1.5.1) —
offline & deterministic.

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
    assert v1.categories == (("rocm-build",), ("performance",))
    assert taxonomy.get_active(store) == v1


def test_create_taxonomy_accepts_multi_level_paths(store: JsonlStore) -> None:
    """T1.5.1's named scenario: a category is a path, not just a flat name."""
    v1 = taxonomy.create_taxonomy(
        store, [["ROCm/AMD", "DeepSeek-V4", "performance", "attention"], "quantization"]
    )
    assert v1.categories == (
        ("ROCm/AMD", "DeepSeek-V4", "performance", "attention"),
        ("quantization",),
    )
    assert v1.labels == ("ROCm/AMD > DeepSeek-V4 > performance > attention", "quantization")


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
    assert v2.categories == (("rocm-build",), ("quantization",))

    # both versions still retrievable by number...
    assert taxonomy.get_taxonomy(store, 1) == v1
    assert taxonomy.get_taxonomy(store, 2) == v2
    # ...and active() returns the latest, not the first
    assert taxonomy.get_active(store) == v2


def test_add_category_accepts_a_multi_level_path(store: JsonlStore) -> None:
    taxonomy.create_taxonomy(store, ["rocm-build"])
    v2 = taxonomy.add_category(store, ["Quantization", "FP8 KV cache"])
    assert v2.categories == (("rocm-build",), ("Quantization", "FP8 KV cache"))


def test_add_category_duplicate_raises(store: JsonlStore) -> None:
    taxonomy.create_taxonomy(store, ["rocm-build"])
    with pytest.raises(taxonomy.TaxonomyError, match="already a category"):
        taxonomy.add_category(store, "rocm-build")


@pytest.mark.parametrize("variant", ["ROCm-Build", "rocm-build ", " ROCM-BUILD"])
def test_add_category_duplicate_case_and_whitespace_insensitive_raises(
    store: JsonlStore, variant: str
) -> None:
    taxonomy.create_taxonomy(store, ["rocm-build"])
    with pytest.raises(taxonomy.TaxonomyError, match="already a category"):
        taxonomy.add_category(store, variant)


def test_add_category_duplicate_path_case_and_whitespace_insensitive_raises(
    store: JsonlStore,
) -> None:
    taxonomy.create_taxonomy(store, [["ROCm/AMD", "DeepSeek"]])
    with pytest.raises(taxonomy.TaxonomyError, match="already a category"):
        taxonomy.add_category(store, [" rocm/amd ", "DEEPSEEK"])


def test_add_category_before_create_raises(store: JsonlStore) -> None:
    with pytest.raises(taxonomy.TaxonomyError, match="no taxonomy exists"):
        taxonomy.add_category(store, "rocm-build")


def test_taxonomy_json_roundtrip() -> None:
    original = taxonomy.Taxonomy(version=3, categories=(("a",), ("b", "c")))
    assert taxonomy.Taxonomy.from_json(original.to_json()) == original


def test_taxonomy_from_json_reads_a_legacy_flat_category_as_a_depth_1_path() -> None:
    """T1.5.1's back-compat requirement: a taxonomy record written before T1.5.1 (a flat list
    of strings) must still load, with each string read as a depth-1 path."""
    legacy_raw = '{"version": 1, "categories": ["rocm-build", "performance"]}'
    loaded = taxonomy.Taxonomy.from_json(legacy_raw)
    assert loaded.categories == (("rocm-build",), ("performance",))
    assert loaded.labels == ("rocm-build", "performance")


def test_categories_field_is_a_real_immutable_tuple() -> None:
    """frozen=True alone doesn't stop a caller mutating a list field in place — tuple does."""
    t = taxonomy.Taxonomy(version=1, categories=(("a",),))
    assert isinstance(t.categories, tuple)
    with pytest.raises(AttributeError):
        t.categories.append("b")  # type: ignore[attr-defined]
    hash(t)  # must not raise — a frozen dataclass should be hashable


def test_get_active_corrupt_pointer_raises_taxonomyerror(store: JsonlStore) -> None:
    taxonomy.create_taxonomy(store, ["rocm-build"])
    store.set_state("taxonomy_active_version", "not-a-number")
    with pytest.raises(taxonomy.TaxonomyError, match="corrupt active-version pointer"):
        taxonomy.get_active(store)


def test_get_taxonomy_corrupt_json_raises_taxonomyerror(store: JsonlStore) -> None:
    taxonomy.create_taxonomy(store, ["rocm-build"])
    store.set_state("taxonomy@1", "not valid json")
    with pytest.raises(taxonomy.TaxonomyError, match="corrupt taxonomy record"):
        taxonomy.get_taxonomy(store, 1)


def test_get_taxonomy_missing_field_raises_taxonomyerror(store: JsonlStore) -> None:
    taxonomy.create_taxonomy(store, ["rocm-build"])
    store.set_state("taxonomy@1", '{"version": 1}')  # missing "categories"
    with pytest.raises(taxonomy.TaxonomyError, match="corrupt taxonomy record"):
        taxonomy.get_taxonomy(store, 1)


def test_get_taxonomy_version_mismatch_raises_taxonomyerror(store: JsonlStore) -> None:
    taxonomy.create_taxonomy(store, ["rocm-build"])
    store.set_state("taxonomy@1", '{"version": 99, "categories": ["x"]}')
    with pytest.raises(taxonomy.TaxonomyError, match="KB corruption"):
        taxonomy.get_taxonomy(store, 1)
