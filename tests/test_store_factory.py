"""Tests for the Store factory (T0.6.2) — offline & deterministic.

``get_store()`` selects the backend via env ``STORE=jsonl|firestore`` (default ``jsonl``). The
firestore path is exercised with ``firestore.Client`` mocked (no live emulator/credentials
needed) — this test is about env-based *routing*, not FirestoreStore's own behavior (that's
covered by ``tests/test_store_contract.py``).
"""

from types import SimpleNamespace

import pytest

from src import config
from src.store import get_store
from src.store.firestore_store import FirestoreStore
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m0_6


class _FakeFirestoreClient:
    """A stand-in for firestore.Client — construction + .collection() must not touch a network."""

    def __init__(self, *args, **kwargs) -> None:
        self.init_args = args
        self.init_kwargs = kwargs

    def collection(self, name: str) -> str:
        return name  # identity is enough; FirestoreStore just stores the return value


def test_get_store_defaults_to_jsonl(monkeypatch, tmp_path):
    monkeypatch.delenv("STORE", raising=False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    store = get_store()
    assert isinstance(store, JsonlStore)
    assert store.data_dir == tmp_path


def test_get_store_selects_jsonl_explicitly(monkeypatch, tmp_path):
    monkeypatch.setenv("STORE", "jsonl")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert isinstance(get_store(), JsonlStore)


def test_get_store_selects_firestore(monkeypatch):
    # firestore import mocked: patch the `firestore` name firestore_store.py binds at import
    # time, so FirestoreStore() never attempts a real connection or needs real credentials.
    import src.store.firestore_store as firestore_store_module

    monkeypatch.setenv("STORE", "firestore")
    monkeypatch.setattr(config, "FIRESTORE_PROJECT", "forager-factory-test")
    fake_firestore_module = SimpleNamespace(Client=_FakeFirestoreClient)
    monkeypatch.setattr(firestore_store_module, "firestore", fake_firestore_module)

    store = get_store()
    assert isinstance(store, FirestoreStore)


def test_get_store_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("STORE", "something-else")
    with pytest.raises(ValueError, match="unknown STORE backend"):
        get_store()
