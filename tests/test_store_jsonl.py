"""Tests for JSONL-backend-*specific* behavior (T0.6) — offline & deterministic, on ``tmp_path``.

The :class:`~src.store.base.Store` contract itself (upsert/get, query filters+ordering, state
round-trip) is pinned once for every backend by ``tests/test_store_contract.py`` (T0.6.1); this
file now covers only what's genuinely unique to the on-disk JSONL layout: one file per repo,
tolerance of a nonexistent data directory, and self-healing past a corrupt/partial line.
"""

import pytest

from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m0


def _item(repo: str, number: int, **overrides) -> dict:
    """A normalized item like the collector produces, with sane defaults for overriding."""
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": f"t{number}",
        "state": "open",
        "labels": [],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "url": f"http://x/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


def test_upsert_routes_items_to_one_jsonl_file_per_repo(tmp_path):
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/a", 1), _item("o/b", 1)])
    # one JSONL file per repo; the "/" becomes "__"
    assert (tmp_path / "o__a.jsonl").exists()
    assert (tmp_path / "o__b.jsonl").exists()


def test_query_empty_dir(tmp_path):
    # querying before anything is written must not raise (glob over an empty/absent dir)
    assert JsonlStore(tmp_path / "nope").query() == []


def test_read_tolerates_corrupt_and_numberless_lines(tmp_path):
    # A damaged file (a truncated line + a valid-JSON line missing `number`) must not abort
    # the read: the bad lines are skipped and the good record survives / self-heals.
    store = JsonlStore(tmp_path)
    path = tmp_path / "o__r.jsonl"
    path.write_text(
        '{"repo": "o/r", "number": 1, "updated_at": "2025-01-01T00:00:00Z"}\n'
        '{"repo": "o/r", "titl\n'  # truncated → JSONDecodeError, skipped
        '{"repo": "o/r", "noNumber": true}\n'  # valid JSON but no number, skipped
    )
    assert [i["number"] for i in store.query(repo="o/r")] == [1]
    # a later upsert rewrites the file cleanly with only the salvageable records
    assert store.upsert_items([_item("o/r", 2)]) == {"o/r": 2}
    assert {i["number"] for i in store.query(repo="o/r")} == {1, 2}
