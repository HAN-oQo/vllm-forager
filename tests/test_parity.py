"""Tests for the parity matrix (T2.4) — offline & deterministic.

Per the DEVPLAN todo: synthetic capability signals → matrix cell populated + "present in fork,
missing upstream" gap flagged.
"""

import pytest

from src import parity

pytestmark = pytest.mark.m2


def _item(repo: str, number: int, category: str, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "pr",
        "title": "x",
        "state": "closed",
        "category": category,
        "url": f"https://github.com/{repo}/pull/{number}",
    }
    rec.update(overrides)
    return rec


# --------------------------------------------------------------------- build_matrix


def test_build_matrix_present_when_shipped() -> None:
    items = [_item("ROCm/vllm", 1, "fp8-kv-cache", state="closed", type="pr")]

    cells = parity.build_matrix(items)

    assert len(cells) == 1
    assert cells[0] == parity.ParityCell(
        engine="ROCm/vllm",
        capability="fp8-kv-cache",
        present=True,
        evidence="https://github.com/ROCm/vllm/pull/1",
    )


def test_build_matrix_absent_when_only_open_pr() -> None:
    items = [_item("ROCm/vllm", 1, "fp8-kv-cache", state="open", type="pr")]

    cells = parity.build_matrix(items)

    assert cells[0].present is False
    assert cells[0].evidence is None


def test_build_matrix_absent_when_only_an_issue() -> None:
    items = [_item("ROCm/vllm", 1, "fp8-kv-cache", type="issue", state="closed")]

    cells = parity.build_matrix(items)

    assert cells[0].present is False
    assert cells[0].evidence is None


def test_build_matrix_excludes_uncategorized_items() -> None:
    items = [_item("ROCm/vllm", 1, "fp8-kv-cache")]
    del items[0]["category"]

    assert parity.build_matrix(items) == []


def test_build_matrix_excludes_non_string_category() -> None:
    items = [_item("ROCm/vllm", 1, category=["not-a-string"])]

    assert parity.build_matrix(items) == []


def test_build_matrix_excludes_items_with_no_repo() -> None:
    items = [_item("ROCm/vllm", 1, "fp8-kv-cache")]
    del items[0]["repo"]

    assert parity.build_matrix(items) == []


def test_build_matrix_one_cell_per_engine_capability_pair() -> None:
    items = [
        _item("ROCm/vllm", 1, "fp8-kv-cache"),
        _item("ROCm/vllm", 2, "fp8-kv-cache", state="open"),  # same cell, still present overall
        _item("vllm-project/vllm", 3, "fp8-kv-cache", state="open"),  # different engine
    ]

    cells = {(c.engine, c.capability): c for c in parity.build_matrix(items)}

    assert cells[("ROCm/vllm", "fp8-kv-cache")].present is True
    assert cells[("vllm-project/vllm", "fp8-kv-cache")].present is False


# --------------------------------------------------------------------- find_gaps


def test_find_gaps_flags_fork_present_upstream_missing() -> None:
    items = [
        _item("ROCm/vllm", 1, "fp8-kv-cache", state="closed"),
        _item("vllm-project/vllm", 2, "fp8-kv-cache", state="open"),
    ]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(cells, fork_engine="ROCm/vllm", upstream_engine="vllm-project/vllm")

    assert gaps == [
        parity.Gap(capability="fp8-kv-cache", evidence="https://github.com/ROCm/vllm/pull/1")
    ]


def test_find_gaps_flags_when_upstream_has_no_cell_at_all() -> None:
    items = [_item("ROCm/vllm", 1, "fp8-kv-cache", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(cells, fork_engine="ROCm/vllm", upstream_engine="vllm-project/vllm")

    assert len(gaps) == 1
    assert gaps[0].capability == "fp8-kv-cache"


def test_find_gaps_no_gap_when_both_shipped() -> None:
    items = [
        _item("ROCm/vllm", 1, "fp8-kv-cache", state="closed"),
        _item("vllm-project/vllm", 2, "fp8-kv-cache", state="closed"),
    ]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(cells, fork_engine="ROCm/vllm", upstream_engine="vllm-project/vllm")

    assert gaps == []


def test_find_gaps_no_gap_when_fork_does_not_have_it() -> None:
    items = [_item("vllm-project/vllm", 1, "fp8-kv-cache", state="closed")]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(cells, fork_engine="ROCm/vllm", upstream_engine="vllm-project/vllm")

    assert gaps == []


def test_find_gaps_uses_config_defaults() -> None:
    """No explicit fork_engine/upstream_engine -- falls back to config.REPOS's fork/primary
    roles (ROCm/vllm, vllm-project/vllm)."""
    items = [
        _item("ROCm/vllm", 1, "fp8-kv-cache", state="closed"),
        _item("vllm-project/vllm", 2, "fp8-kv-cache", state="open"),
    ]
    cells = parity.build_matrix(items)

    gaps = parity.find_gaps(cells)

    assert [g.capability for g in gaps] == ["fp8-kv-cache"]
