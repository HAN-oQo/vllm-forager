"""Store contract test (T0.6.1) — the same assertions must hold for every Store backend.

Parametrized over :class:`~src.store.jsonl_store.JsonlStore` (offline, ``tmp_path`` — runs by
default) and :class:`~src.store.firestore_store.FirestoreStore` (a live Firestore emulator,
``@pytest.mark.integration`` — skipped by default). To run the firestore parametrization:

    docker run -d -p 8081:8081 gcr.io/google.com/cloudsdktool/cloud-sdk:emulators \\
        gcloud emulators firestore start --host-port=0.0.0.0:8081 --database-mode=firestore-native
    FIRESTORE_EMULATOR_HOST=localhost:8081 pytest --run-integration tests/test_store_contract.py

These assertions used to live in :mod:`tests.test_store_jsonl` as ``JsonlStore``-only tests;
they're generalized here across the whole ``Store`` interface so a future backend only needs
to pass this one suite. :mod:`tests.test_store_jsonl` now covers only what's genuinely unique
to the on-disk JSONL layout (one file per repo, corrupt-line tolerance, etc.).
"""

from __future__ import annotations

import pytest
from firestore_emulator_helpers import clear_firestore_emulator, project_id_for

from src.store.firestore_store import FirestoreStore
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m0_6


def _make_jsonl(tmp_path, test_id):
    return JsonlStore(tmp_path)


def _make_firestore(tmp_path, test_id):
    project = project_id_for(test_id, prefix="forager-ct")
    clear_firestore_emulator(project)
    return FirestoreStore(project=project)


@pytest.fixture(
    params=[
        pytest.param(_make_jsonl, id="jsonl"),
        pytest.param(_make_firestore, id="firestore", marks=pytest.mark.integration),
    ]
)
def store(request, tmp_path):
    """A fresh :class:`Store` instance — parametrized over every backend under contract."""
    return request.param(tmp_path, request.node.name)


def _item(repo: str, number: int, **overrides) -> dict:
    """A normalized item like the collector produces, with sane defaults for overriding."""
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": f"t{number}",
        "state": "open",
        "labels": [],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "url": f"http://x/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


# ------------------------------------------------------------- upsert + get_item


def test_upsert_and_get_item(store):
    assert store.upsert_items([_item("o/r", 1), _item("o/r", 2)]) == {"o/r": 2}
    assert store.get_item("o/r", 1)["title"] == "t1"
    assert store.get_item("o/r", 99) is None  # absent → None

    # upsert is keyed on (repo, number): re-upserting #1 overwrites, doesn't duplicate.
    assert store.upsert_items([_item("o/r", 1, title="updated")]) == {"o/r": 2}
    assert store.get_item("o/r", 1)["title"] == "updated"
    assert len(store.query(repo="o/r")) == 2

    assert store.upsert_items([]) == {}  # empty batch → empty map


def test_upsert_dedupes_same_item_within_one_call(store):
    # Two records for the same (repo, number) in a SINGLE upsert_items call — last one wins,
    # not an error. (A Firestore backend that writes every record verbatim without deduping
    # first would reject this: its batch API forbids two writes to the same document in one
    # commit — see src/store/firestore_store.py's upsert_items docstring.)
    assert store.upsert_items(
        [_item("o/r", 1, title="first"), _item("o/r", 1, title="second")]
    ) == {"o/r": 1}
    assert store.get_item("o/r", 1)["title"] == "second"


def test_upsert_merges_fields_preserving_ones_not_in_the_new_record(store):
    """T1.10: a later upsert that omits a field must preserve it, not erase it — e.g. the
    collector re-normalizing an item must not wipe a `category` T1.4's Analyst added."""
    store.upsert_items([_item("o/r", 1, title="original")])

    # a partial record (only `category`) merges onto the existing one, not replaces it.
    store.upsert_items([{"repo": "o/r", "number": 1, "category": "build"}])
    record = store.get_item("o/r", 1)
    assert record["category"] == "build"
    assert record["title"] == "original"  # preserved — this upsert never mentioned it

    # a field the new record DOES specify still overwrites the old value.
    store.upsert_items([_item("o/r", 1, title="updated")])
    record = store.get_item("o/r", 1)
    assert record["title"] == "updated"
    assert record["category"] == "build"  # untouched — this upsert didn't mention it either


def test_upsert_dedupes_same_item_within_one_call_by_merging(store):
    """Two partial records for the same (repo, number) in one call merge in order, not just
    last-one-wins on the whole record — matching upsert's own field-level merge semantics."""
    assert store.upsert_items(
        [
            {"repo": "o/r", "number": 1, "title": "t1", "category": "build"},
            {"repo": "o/r", "number": 1, "title": "t1-renamed"},
        ]
    ) == {"o/r": 1}
    record = store.get_item("o/r", 1)
    assert record["title"] == "t1-renamed"  # the second record's field wins
    assert record["category"] == "build"  # preserved from the first record in the same batch


def test_upsert_routes_items_by_repo(store):
    assert store.upsert_items([_item("o/a", 1), _item("o/b", 1)]) == {"o/a": 1, "o/b": 1}
    assert store.get_item("o/a", 1) is not None
    assert store.get_item("o/b", 1) is not None
    assert len(store.query()) == 2  # no repo filter → across every repo


# -------------------------------------------------------------------- query filters


def test_query_filters_and_combination(store):
    store.upsert_items(
        [
            _item("o/r", 1, state="open", type="issue", labels=["bug", "rocm"]),
            _item("o/r", 2, state="closed", type="pr", labels=["rocm"]),
            _item("o/r", 3, state="open", type="pr", labels=[]),
            _item("o/s", 4, state="open", type="issue", labels=["bug"]),
        ]
    )
    nums = lambda items: {i["number"] for i in items}  # noqa: E731

    assert nums(store.query(repo="o/r")) == {1, 2, 3}
    assert nums(store.query(state="open")) == {1, 3, 4}
    assert nums(store.query(label="rocm")) == {1, 2}
    assert nums(store.query(type="pr")) == {2, 3}
    # filters are AND-combined
    assert nums(store.query(repo="o/r", state="open", type="pr")) == {3}
    # no match → empty list (not an error)
    assert store.query(repo="o/r", label="nonexistent") == []


def test_query_ordered_by_updated_at(store):
    store.upsert_items(
        [
            _item("o/r", 1, updated_at="2025-01-03T00:00:00Z"),
            _item("o/r", 2, updated_at="2025-01-01T00:00:00Z"),
            _item("o/r", 3, updated_at="2025-01-02T00:00:00Z"),
        ]
    )
    assert [i["number"] for i in store.query(repo="o/r")] == [2, 3, 1]


def test_query_empty_store(store):
    assert store.query() == []


# --------------------------------------------------------------------- state cursor


def test_state_roundtrip(store):
    assert store.get_state("o/r") is None  # unset → None
    store.set_state("o/r", "2025-01-01T00:00:00Z")
    assert store.get_state("o/r") == "2025-01-01T00:00:00Z"

    # setting a second key preserves the first.
    store.set_state("o/s", "2025-02-02T00:00:00Z")
    assert store.get_state("o/r") == "2025-01-01T00:00:00Z"
    assert store.get_state("o/s") == "2025-02-02T00:00:00Z"
