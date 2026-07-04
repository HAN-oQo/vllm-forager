"""Tests for the jsonl→Firestore migration script (T0.6.3).

``@pytest.mark.integration`` — needs a live Firestore emulator (skipped by default, same as
``tests/test_store_contract.py``). To run:

    docker run -d -p 8081:8081 gcr.io/google.com/cloudsdktool/cloud-sdk:emulators \\
        gcloud emulators firestore start --host-port=0.0.0.0:8081 --database-mode=firestore-native
    FIRESTORE_EMULATOR_HOST=localhost:8081 pytest --run-integration tests/test_store_migrate.py
"""

from __future__ import annotations

import pytest
from firestore_emulator_helpers import clear_firestore_emulator, project_id_for

from src.store import migrate
from src.store.firestore_store import FirestoreStore
from src.store.jsonl_store import JsonlStore

pytestmark = [pytest.mark.m0_6, pytest.mark.integration]


@pytest.fixture
def firestore_project(request):
    project = project_id_for(request.node.name, prefix="forager-mig")
    clear_firestore_emulator(project)
    return project


def _item(repo: str, number: int, **overrides) -> dict:
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


def test_migrate_writes_items_and_state_to_firestore(tmp_path, firestore_project):
    source = JsonlStore(tmp_path)
    source.upsert_items([_item("o/r", 1), _item("o/r", 2), _item("o/s", 3)])
    source.set_state("o/r", "2025-01-01T00:00:00Z")
    source.set_state("o/s", "2025-02-02T00:00:00Z")

    summary = migrate.migrate(tmp_path, firestore_project=firestore_project)
    assert summary["items_migrated"] == 3
    assert summary["item_totals"] == {"o/r": 2, "o/s": 1}
    assert summary["state_keys_migrated"] == 2

    dest = FirestoreStore(project=firestore_project)
    assert dest.get_item("o/r", 1)["title"] == "t1"
    assert dest.get_item("o/s", 3)["title"] == "t3"
    assert dest.get_state("o/r") == "2025-01-01T00:00:00Z"
    assert dest.get_state("o/s") == "2025-02-02T00:00:00Z"


def test_migrate_is_idempotent(tmp_path, firestore_project):
    source = JsonlStore(tmp_path)
    source.upsert_items([_item("o/r", 1)])
    source.set_state("o/r", "2025-01-01T00:00:00Z")

    migrate.migrate(tmp_path, firestore_project=firestore_project)
    # re-running (e.g. after collecting more items into the same jsonl dir) must not duplicate
    # or error — same item set in, same totals out.
    summary = migrate.migrate(tmp_path, firestore_project=firestore_project)
    assert summary["item_totals"] == {"o/r": 1}

    dest = FirestoreStore(project=firestore_project)
    assert len(dest.query(repo="o/r")) == 1


def test_migrate_empty_source_is_a_safe_noop(tmp_path, firestore_project):
    # A data dir with nothing collected yet (no jsonl files, no state.json) must not crash.
    summary = migrate.migrate(tmp_path, firestore_project=firestore_project)
    assert summary == {"items_migrated": 0, "item_totals": {}, "state_keys_migrated": 0}
