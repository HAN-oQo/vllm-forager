"""Shared helpers for tests that talk to a live Firestore emulator (``@pytest.mark.integration``).

Used by ``tests/test_store_contract.py`` and ``tests/test_store_migrate.py`` — factored out so
the emulator-wipe REST call and the per-test-project naming scheme have exactly one definition
each, rather than two copies drifting apart.
"""

from __future__ import annotations

import os
import re

import requests


def project_id_for(test_id: str, *, prefix: str) -> str:
    """A Firestore project id unique to this test node (each test gets its own namespace).

    Derived from the pytest node id rather than a fixed shared project — a shared project
    cleared per-test only works under strictly serial execution; a unique project per test
    removes the cross-test/parallel-worker collision risk entirely (e.g. under pytest-xdist).
    `prefix` namespaces different test files' projects from each other (cosmetic only).
    """
    slug = re.sub(r"[^a-z0-9-]+", "-", test_id.lower()).strip("-")
    return f"{prefix}-{slug}"[:63]  # Firestore project ids are capped at 63 chars


def clear_firestore_emulator(project: str) -> None:
    """Wipe every document for `project` in the emulator — belt-and-suspenders isolation.

    Uses the emulator's admin REST endpoint (real Firestore has no such call; this only ever
    runs against ``FIRESTORE_EMULATOR_HOST``, never production). Each test already gets its
    own project (see :func:`project_id_for`), so this guards against leftover data from a
    previous *interrupted* run of the same test rather than cross-test pollution.
    """
    host = os.environ["FIRESTORE_EMULATOR_HOST"]
    url = f"http://{host}/emulator/v1/projects/{project}/databases/(default)/documents"
    requests.delete(url, timeout=10)
