"""Tests for the Store factory (T0.6.2) — offline & deterministic.

``get_store()`` selects the backend via :data:`src.config.STORE_BACKEND` (env ``STORE``,
default ``jsonl``). Tests monkeypatch the *config constant* directly, matching the pattern
every other config.py value uses (it's cached at import, so patching the env var after the
fact has no effect). The firestore path is exercised with ``firestore.Client`` mocked (no
live emulator/credentials needed) — this test is about backend *routing*, not FirestoreStore's
own behavior (that's covered by ``tests/test_store_contract.py``).
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
    monkeypatch.setattr(config, "STORE_BACKEND", "jsonl")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    store = get_store()
    assert isinstance(store, JsonlStore)
    assert store.data_dir == tmp_path


def test_get_store_selects_firestore(monkeypatch):
    # firestore import mocked: patch the `firestore` name firestore_store.py binds at import
    # time, so FirestoreStore() never attempts a real connection or needs real credentials.
    import src.store.firestore_store as firestore_store_module

    monkeypatch.setattr(config, "STORE_BACKEND", "firestore")
    monkeypatch.setattr(config, "FIRESTORE_PROJECT", "forager-factory-test")
    fake_firestore_module = SimpleNamespace(Client=_FakeFirestoreClient)
    monkeypatch.setattr(firestore_store_module, "firestore", fake_firestore_module)

    store = get_store()
    assert isinstance(store, FirestoreStore)
    # config.FIRESTORE_PROJECT actually reached firestore.Client(project=...), not just an
    # unused config value — the fake client records exactly what it was constructed with.
    assert store._client.init_kwargs == {"project": "forager-factory-test"}


def test_get_store_rejects_unknown_backend(monkeypatch):
    monkeypatch.setattr(config, "STORE_BACKEND", "something-else")
    with pytest.raises(ValueError, match="unknown STORE backend"):
        get_store()
