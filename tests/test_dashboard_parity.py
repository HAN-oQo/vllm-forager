"""Tests for the dashboard parity-diagram data layer (T5.4) — offline & deterministic.

Per the DEVPLAN todo: the matrix endpoint returns cells + evidence + gap flags.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard import parity_view
from src import parity as parity_module
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5


def _item(repo: str, number: int, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "pr",
        "title": f"item {number}",
        "state": "closed",
        "labels": [],
        "created_at": "2026-01-05T00:00:00Z",
        "updated_at": "2026-01-05T00:00:00Z",
        "url": f"http://x/{repo}/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


def _two_engine_repos() -> list[dict]:
    return [
        {"slug": "vllm-project/vllm", "role": "primary", "domain": "speech"},
        {"slug": "other/engine", "role": "source", "domain": "speech"},
    ]


def test_parity_matrix_marks_a_never_discussed_capability_as_a_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core case: "fp8" is shipped on other/engine but vllm-project/vllm has never even
    had an item classified under it -- there's no ParityCell for that pair at all, yet the
    heatmap must still show a red (is_gap=True) cell there, not silently omit it."""
    monkeypatch.setattr(parity_module.config, "REPOS", _two_engine_repos())
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("other/engine", 1, category="fp8")])

    matrix = parity_view.parity_matrix(store)

    assert matrix["engines"] == ["other/engine", "vllm-project/vllm"]
    assert matrix["capabilities"] == ["fp8"]
    cells = {c["engine"]: c for c in matrix["cells"]}
    assert cells["other/engine"] == {
        "engine": "other/engine",
        "capability": "fp8",
        "present": True,
        "evidence": "http://x/other/engine/1",
        "is_gap": False,
    }
    assert cells["vllm-project/vllm"] == {
        "engine": "vllm-project/vllm",
        "capability": "fp8",
        "present": False,
        "evidence": None,
        "is_gap": True,
    }


def test_parity_matrix_derives_engines_and_capabilities_from_signal_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        parity_module.config,
        "REPOS",
        [*_two_engine_repos(), {"slug": "untouched/engine", "role": "radar", "domain": "speech"}],
    )
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("other/engine", 1, category="fp8")])

    matrix = parity_view.parity_matrix(store)

    assert "untouched/engine" not in matrix["engines"]


def test_parity_matrix_no_data_returns_an_empty_grid(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert parity_view.parity_matrix(store) == {"engines": [], "capabilities": [], "cells": []}


def test_parity_matrix_degrades_without_a_primary_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        parity_module.config, "REPOS", [{"slug": "o/r", "role": "source", "domain": "speech"}]
    )
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, category="fp8")])

    matrix = parity_view.parity_matrix(store)  # must not raise

    assert matrix["engines"] == ["o/r"]
    assert all(not c["is_gap"] for c in matrix["cells"])
