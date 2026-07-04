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
