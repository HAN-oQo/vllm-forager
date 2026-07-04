"""Pluggable knowledge-base store.

`Store` (in ``base``) is the backend-agnostic contract; `JsonlStore` is the M0 default
on-disk backend; `FirestoreStore` (M0.6, T0.6.1) is the Firestore-backed alternative — both
implement this same interface. `get_store()` (T0.6.2) selects between them via
:data:`config.STORE_BACKEND` (env ``STORE``).
"""

from __future__ import annotations

from .. import config
from .base import Store
from .jsonl_store import JsonlStore

__all__ = ["Store", "JsonlStore", "get_store"]


def get_store() -> Store:
    """Return the configured Store backend, per :data:`config.STORE_BACKEND` (default ``jsonl``).

    - ``"jsonl"`` → :class:`JsonlStore` over :data:`config.DATA_DIR`.
    - ``"firestore"`` → :class:`~src.store.firestore_store.FirestoreStore`
      (:data:`config.FIRESTORE_PROJECT`), imported lazily here so a JSONL-only collector run
      never pays for loading the Firestore SDK.

    Raises ``ValueError`` on an unrecognized value — a typo in ``STORE`` should be loud, not a
    silent fallback to the wrong backend.
    """
    backend = config.STORE_BACKEND
    if backend == "jsonl":
        return JsonlStore(config.DATA_DIR)
    if backend == "firestore":
        from .firestore_store import FirestoreStore

        return FirestoreStore(project=config.FIRESTORE_PROJECT)
    raise ValueError(f"unknown STORE backend {backend!r} (expected 'jsonl' or 'firestore')")
