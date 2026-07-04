"""Tests for the versioned policy object (T1.3) — offline & deterministic.

Uses a real :class:`JsonlStore` on ``tmp_path`` (not a mock) since the module's whole job is
to round-trip through the generic ``Store.get_state``/``set_state`` contract correctly.
"""

from pathlib import Path
from types import MappingProxyType

import pytest

from src import policy
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m1


@pytest.fixture
def store(tmp_path: Path) -> JsonlStore:
    return JsonlStore(tmp_path)


def test_create_policy_is_v1_and_active(store: JsonlStore) -> None:
    v1 = policy.create_policy(
        store,
        scoring_weights={"rocm-build": 1.0},
        prompt_templates={"classify": "..."},
        active_taxonomy_version=1,
    )
    assert v1.version == 1
    assert v1.scoring_weights == {"rocm-build": 1.0}
    assert v1.prompt_templates == {"classify": "..."}
    assert v1.active_taxonomy_version == 1
    assert policy.get_active(store) == v1


def test_create_policy_twice_raises(store: JsonlStore) -> None:
    policy.create_policy(store, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1)
    with pytest.raises(policy.PolicyError, match="already exists"):
        policy.create_policy(
            store, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1
        )


def test_get_active_before_create_raises(store: JsonlStore) -> None:
    with pytest.raises(policy.PolicyError, match="no policy exists"):
        policy.get_active(store)


def test_get_policy_unknown_version_raises(store: JsonlStore) -> None:
    policy.create_policy(store, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1)
    with pytest.raises(policy.PolicyError, match="version 2 not found"):
        policy.get_policy(store, 2)


def test_update_policy_before_create_raises(store: JsonlStore) -> None:
    with pytest.raises(policy.PolicyError, match="no policy exists"):
        policy.update_policy(store, scoring_weights={"x": 1.0})


def test_update_policy_bumps_version_both_retrievable_active_is_latest(
    store: JsonlStore,
) -> None:
    """The DEVPLAN's named invariant: versions are append-only/immutable; get_active() latest."""
    v1 = policy.create_policy(
        store,
        scoring_weights={"rocm-build": 1.0},
        prompt_templates={"classify": "v1 template"},
        active_taxonomy_version=1,
    )
    v2 = policy.update_policy(store, scoring_weights={"rocm-build": 2.0})

    assert v2.version == 2
    assert v2.scoring_weights == {"rocm-build": 2.0}
    # unspecified fields carry over unchanged from v1
    assert v2.prompt_templates == {"classify": "v1 template"}
    assert v2.active_taxonomy_version == 1

    # both versions still retrievable by number, and v1 is untouched (append-only)...
    assert policy.get_policy(store, 1) == v1
    assert policy.get_policy(store, 2) == v2
    assert policy.get_policy(store, 1).scoring_weights == {"rocm-build": 1.0}
    # ...and active() returns the latest, not the first
    assert policy.get_active(store) == v2


def test_update_policy_changes_only_active_taxonomy_version(store: JsonlStore) -> None:
    policy.create_policy(
        store,
        scoring_weights={"a": 1.0},
        prompt_templates={"t": "x"},
        active_taxonomy_version=1,
    )
    v2 = policy.update_policy(store, active_taxonomy_version=2)
    assert v2.active_taxonomy_version == 2
    assert v2.scoring_weights == {"a": 1.0}
    assert v2.prompt_templates == {"t": "x"}


def test_scoring_weights_and_prompt_templates_are_immutable_mappings(
    store: JsonlStore,
) -> None:
    """frozen=True alone doesn't stop a plain dict field from being mutated in place."""
    v1 = policy.create_policy(
        store, scoring_weights={"a": 1.0}, prompt_templates={"t": "x"}, active_taxonomy_version=1
    )
    assert isinstance(v1.scoring_weights, MappingProxyType)
    assert isinstance(v1.prompt_templates, MappingProxyType)
    with pytest.raises(TypeError):
        v1.scoring_weights["a"] = 99.0  # type: ignore[index]


def test_get_active_corrupt_pointer_raises_policyerror(store: JsonlStore) -> None:
    policy.create_policy(store, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1)
    store.set_state("policy_active_version", "not-a-number")
    with pytest.raises(policy.PolicyError, match="corrupt active-version pointer"):
        policy.get_active(store)


def test_get_policy_corrupt_json_raises_policyerror(store: JsonlStore) -> None:
    policy.create_policy(store, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1)
    store.set_state("policy@1", "not valid json")
    with pytest.raises(policy.PolicyError, match="corrupt policy record"):
        policy.get_policy(store, 1)


def test_get_policy_missing_field_raises_policyerror(store: JsonlStore) -> None:
    policy.create_policy(store, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1)
    store.set_state("policy@1", '{"version": 1}')  # missing required fields
    with pytest.raises(policy.PolicyError, match="corrupt policy record"):
        policy.get_policy(store, 1)


def test_get_policy_version_mismatch_raises_policyerror(store: JsonlStore) -> None:
    policy.create_policy(store, scoring_weights={}, prompt_templates={}, active_taxonomy_version=1)
    store.set_state(
        "policy@1",
        '{"version": 99, "scoring_weights": {}, "prompt_templates": {}, '
        '"active_taxonomy_version": 1}',
    )
    with pytest.raises(policy.PolicyError, match="KB corruption"):
        policy.get_policy(store, 1)


def test_policy_json_roundtrip() -> None:
    original = policy.Policy(
        version=3,
        scoring_weights=MappingProxyType({"a": 1.0}),
        prompt_templates=MappingProxyType({"t": "x"}),
        active_taxonomy_version=2,
    )
    assert policy.Policy.from_json(original.to_json()) == original
