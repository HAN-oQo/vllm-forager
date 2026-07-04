"""Tests for the JSONL store backend (T0.6) — offline & deterministic, on ``tmp_path``.

Covers the :class:`~src.store.base.Store` contract as implemented by
:class:`~src.store.jsonl_store.JsonlStore`: item upsert + get, query by repo/label/state/
type (AND-combined), updated_at ordering, per-repo file routing, and state round-trip.
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


# ------------------------------------------------------------- upsert + get_item


def test_upsert_and_get_item(tmp_path):
    store = JsonlStore(tmp_path)
    # upsert_items returns {repo: post-upsert total}
    assert store.upsert_items([_item("o/r", 1), _item("o/r", 2)]) == {"o/r": 2}
    assert store.get_item("o/r", 1)["title"] == "t1"
    assert store.get_item("o/r", 99) is None  # absent → None

    # upsert is keyed on (repo, number): re-upserting #1 overwrites, doesn't duplicate,
    # so the returned total stays 2 (not 3).
    assert store.upsert_items([_item("o/r", 1, title="updated")]) == {"o/r": 2}
    assert store.get_item("o/r", 1)["title"] == "updated"
    assert len(store.query(repo="o/r")) == 2

    assert store.upsert_items([]) == {}  # empty batch → empty map, no files touched


def test_upsert_routes_items_by_repo(tmp_path):
    store = JsonlStore(tmp_path)
    # per-repo totals come back keyed by repo
    assert store.upsert_items([_item("o/a", 1), _item("o/b", 1)]) == {"o/a": 1, "o/b": 1}
    # one JSONL file per repo; the "/" becomes "__"
    assert (tmp_path / "o__a.jsonl").exists()
    assert (tmp_path / "o__b.jsonl").exists()
    assert store.get_item("o/a", 1) is not None
    assert store.get_item("o/b", 1) is not None
    assert len(store.query()) == 2  # no repo filter → across all repo files


# -------------------------------------------------------------------- query filters


def test_query_filters_and_combination(tmp_path):
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, state="open", type="issue", labels=["bug", "rocm"]),
            _item("o/r", 2, state="closed", type="pr", labels=["rocm"]),
            _item("o/r", 3, state="open", type="pr", labels=[]),
            _item("o/s", 4, state="open", type="issue", labels=["bug"]),
        ]
    )
    nums = lambda items: {i["number"] for i in items}  # noqa: E731

    assert nums(store.query(repo="o/r")) == {1, 2, 3}
    assert nums(store.query(state="open")) == {1, 3, 4}
    assert nums(store.query(label="rocm")) == {1, 2}
    assert nums(store.query(type="pr")) == {2, 3}
    # filters are AND-combined
    assert nums(store.query(repo="o/r", state="open", type="pr")) == {3}
    # no match → empty list (not an error)
    assert store.query(repo="o/r", label="nonexistent") == []


def test_query_ordered_by_updated_at(tmp_path):
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, updated_at="2025-01-03T00:00:00Z"),
            _item("o/r", 2, updated_at="2025-01-01T00:00:00Z"),
            _item("o/r", 3, updated_at="2025-01-02T00:00:00Z"),
        ]
    )
    assert [i["number"] for i in store.query(repo="o/r")] == [2, 3, 1]


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


# --------------------------------------------------------------------- state cursor


def test_state_roundtrip(tmp_path):
    store = JsonlStore(tmp_path)
    assert store.get_state("o/r") is None  # unset → None
    store.set_state("o/r", "2025-01-01T00:00:00Z")
    assert store.get_state("o/r") == "2025-01-01T00:00:00Z"

    # persisted across store instances; setting a second key preserves the first.
    store2 = JsonlStore(tmp_path)
    store2.set_state("o/s", "2025-02-02T00:00:00Z")
    assert store2.get_state("o/r") == "2025-01-01T00:00:00Z"
    assert store2.get_state("o/s") == "2025-02-02T00:00:00Z"
