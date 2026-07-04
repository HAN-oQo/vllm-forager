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
from src.store import get_store, resolve_store
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


# --------------------------------------------------------------------- resolve_store (T1.11)


def test_resolve_store_with_data_dir_returns_jsonl_store_at_that_path(monkeypatch, tmp_path):
    # STORE=firestore would raise if resolve_store ever called get_store() here — proves
    # the --data-dir branch short-circuits the backend-selected path entirely.
    monkeypatch.setattr(config, "STORE_BACKEND", "firestore")
    explicit_dir = tmp_path / "explicit"

    store, data_dir = resolve_store(explicit_dir)

    assert isinstance(store, JsonlStore)
    assert store.data_dir == explicit_dir
    assert data_dir == explicit_dir


def test_resolve_store_without_data_dir_uses_get_store_and_config_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "STORE_BACKEND", "jsonl")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    store, data_dir = resolve_store(None)

    assert isinstance(store, JsonlStore)
    assert store.data_dir == tmp_path
    assert data_dir == tmp_path
