"""Pluggable knowledge-base store.

`Store` (in ``base``) is the backend-agnostic contract; `JsonlStore` is the M0 default
on-disk backend. A `STORE`-env-driven ``get_store()`` factory is added in T0.6.2, and a
Firestore backend in M0.6 — both against this same interface.
"""

from .base import Store
from .jsonl_store import JsonlStore

__all__ = ["Store", "JsonlStore"]
