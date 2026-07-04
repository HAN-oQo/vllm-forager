"""Pluggable knowledge-base store.

`Store` (in ``base``) is the backend-agnostic contract; `JsonlStore` is the M0 default
on-disk backend; `FirestoreStore` (M0.6, T0.6.1) is the Firestore-backed alternative — both
implement this same interface. `get_store()` (T0.6.2) selects between them via
:data:`config.STORE_BACKEND` (env ``STORE``). `resolve_store()` (T1.11) additionally layers in
a CLI's ``--data-dir`` override, shared by every CLI in this repo that offers one.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from .base import Store
from .jsonl_store import JsonlStore

__all__ = ["Store", "JsonlStore", "get_store", "resolve_store"]


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


def resolve_store(data_dir: Path | None) -> tuple[Store, Path]:
    """Resolve a CLI's ``--data-dir`` flag into a ``(store, data_dir)`` pair (T1.11).

    `data_dir` is the parsed ``--data-dir`` argument (``None`` if the flag wasn't given). If
    provided, always returns a :class:`JsonlStore` at that path and `data_dir` itself — the
    pre-T0.6.2, JSONL-specific override (there's no equivalent "read Firestore instead" flag,
    so mixing the two isn't meaningful). Otherwise returns :func:`get_store`'s
    backend-selected Store (``STORE=jsonl|firestore``) and :data:`config.DATA_DIR`, matching
    the collector's own backend selection.

    Every CLI here that accepts ``--data-dir`` (``src.analyze``, ``src.forecast``,
    ``src.report``, ``dashboard``) calls this instead of re-implementing the same branch —
    before T1.11, this exact ``if data_dir is not None: JsonlStore(...) else: get_store()``
    logic was independently copy-pasted across all four.
    """
    if data_dir is not None:
        return JsonlStore(data_dir), data_dir
    return get_store(), config.DATA_DIR
