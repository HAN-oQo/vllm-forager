"""Pytest configuration shared by the whole suite.

Enforces the repo convention (CLAUDE.md): "Tests are offline/deterministic by default;
anything hitting live GitHub / Firestore emulator / MI250 / a live LLM is
``@pytest.mark.integration`` (skipped in the default run)."

The marker alone doesn't skip anything — pytest still *collects and runs* marked tests. So
by default we skip everything marked ``integration``; pass ``--run-integration`` (or set
``RUN_INTEGRATION=1``) to actually exercise them. This keeps a bare ``pytest`` (and CI, which
has no ``claude`` CLI / network / MI250) green while the live smoke tests stay runnable
on demand.
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run @pytest.mark.integration tests (live network / CLI / LLM / hardware)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration") or os.getenv("RUN_INTEGRATION"):
        return
    skip = pytest.mark.skip(
        reason="integration test — pass --run-integration (or RUN_INTEGRATION=1)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _default_offline_store_backend(monkeypatch):
    """Never let an ambient ``STORE=firestore`` (real shell env or a shared ``.env``, per
    ``docs/RUNBOOK.md``'s shared-data-dir setup) silently switch a test onto FirestoreStore.

    `src.store.get_store` (T0.6.2) is env-driven via :data:`src.config.STORE_BACKEND` — which
    is cached at import time (like every other config.py value), so clearing the *env var*
    here would do nothing; the constant itself must be reset. `src.collector.main` /
    `src.report.main` call `get_store()` by default — a test that doesn't care about the
    backend must still get the offline JSONL default, not a live network dependency. A test
    that *wants* firestore sets ``config.STORE_BACKEND`` itself via `monkeypatch.setattr`,
    which overrides this within that test.
    """
    from src import config

    monkeypatch.setattr(config, "STORE_BACKEND", "jsonl")
