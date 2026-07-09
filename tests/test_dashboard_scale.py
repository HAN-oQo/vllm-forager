"""Tests for T5.16 (Dashboard scalability: aggregate-first + paginated + precomputed
snapshot) — offline & deterministic.

Per the DEVPLAN todo: a store seeded with N vs 100·N items renders pages of bounded,
near-constant size (counts scale, rendered leaf volume does not); a paginated leaf endpoint
returns page k + a correct total; the precomputed snapshot round-trips and a page renders from
it without a full ``Store.query()``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard import snapshot as snap
from dashboard.render import render_page, render_snapshot_page
from dashboard.server import _make_handler
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5


def _items(n: int, *, path: list[str] | None = ("cat",), start: int = 0) -> list[dict]:
    return [
        {
            "repo": "o/r",
            "number": start + i,
            "type": "issue",
            "title": f"item {start + i}",
            "state": "open",
            "url": f"http://x/{start + i}",
            "created_at": "2026-01-01T00:00:00Z",
            **({"path": list(path)} if path is not None else {}),
        }
        for i in range(n)
    ]


def _store_with_n_items(tmp_path: Path, n: int, *, path: list[str] | None = ("cat",)) -> JsonlStore:
    store = JsonlStore(tmp_path)
    store.upsert_items(_items(n, path=path))
    return store


class _FakeHandler:
    """A duck-typed stand-in for BaseHTTPRequestHandler (mirrors tests/test_dashboard.py's own
    `_FakeResponse`, kept local per this repo's convention of not cross-importing test
    fixtures) — enough surface for `_make_handler`'s `Handler.do_GET` to run end to end
    (routing included) without opening a real socket."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.body = b""

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, key: str, value: str) -> None:
        self.headers[key] = value

    def end_headers(self) -> None:
        pass

    class _Wfile:
        def __init__(self, outer: _FakeHandler) -> None:
            self._outer = outer

        def write(self, data: bytes) -> None:
            self._outer.body += data

    @property
    def wfile(self) -> _FakeHandler._Wfile:
        return self._Wfile(self)


# --------------------------------------------------------------------- bounded page size


def test_render_page_stays_bounded_as_item_count_grows_100x(tmp_path_factory) -> None:
    """DEVPLAN's own worked example: a store seeded with N vs 100·N items renders pages of
    bounded, near-constant size -- counts scale, rendered leaf volume does not."""
    small = _store_with_n_items(tmp_path_factory.mktemp("small"), n=5)
    large = _store_with_n_items(tmp_path_factory.mktemp("large"), n=500)  # 100x

    small_page = render_page(small)
    large_page = render_page(large)

    # 100x the items must not make the page anywhere near 100x bigger -- only the capped
    # top-N rows (plus one "show N more" control) are ever embedded, regardless of how many
    # more exist.
    assert len(large_page) < len(small_page) * 3
    # The count still reflects the true total -- aggregate info isn't lost, only per-row embed.
    assert '<span class="count">500</span>' in large_page
    assert f"Show {500 - snap.DEFAULT_MAX_PRS_PER_NODE} more" in large_page
    assert f"Show {5 - snap.DEFAULT_MAX_PRS_PER_NODE} more" not in small_page  # nothing to cap


def test_render_page_caps_the_unclassified_other_bucket_too(tmp_path: Path) -> None:
    """T5.16(d): the unclassified "Other" bucket is capped the same way as any real node --
    not a full, unbounded dump."""
    n = snap.DEFAULT_MAX_PRS_PER_NODE + 25
    store = _store_with_n_items(tmp_path, n, path=None)  # no `path` -> falls into Other

    page = render_page(store)

    assert '<span class="name">Other</span>' in page
    assert f'<span class="count">{n}</span>' in page
    assert "Show 25 more" in page


# --------------------------------------------------------------------- paginated leaf endpoint


def test_node_prs_endpoint_returns_page_k_and_a_correct_total(tmp_path: Path) -> None:
    """DEVPLAN's own worked example: a paginated leaf endpoint returns page k + a correct
    total."""
    store = _store_with_n_items(tmp_path, n=120)
    handler_cls = _make_handler(store, None)

    page0 = _FakeHandler("/api/node-prs?path=cat&offset=0&limit=50")
    handler_cls.do_GET(page0)  # type: ignore[arg-type]
    page1 = _FakeHandler("/api/node-prs?path=cat&offset=50&limit=50")
    handler_cls.do_GET(page1)  # type: ignore[arg-type]
    page2 = _FakeHandler("/api/node-prs?path=cat&offset=100&limit=50")
    handler_cls.do_GET(page2)  # type: ignore[arg-type]

    body0 = json.loads(page0.body)
    body1 = json.loads(page1.body)
    body2 = json.loads(page2.body)

    assert page0.status == 200
    assert body0["total"] == body1["total"] == body2["total"] == 120
    assert len(body0["prs"]) == 50
    assert len(body1["prs"]) == 50
    assert len(body2["prs"]) == 20  # 120 - 100
    # Every page's rows are distinct -- offset actually advances, not stuck re-serving page 0.
    ids0 = {pr["number"] for pr in body0["prs"]}
    ids1 = {pr["number"] for pr in body1["prs"]}
    ids2 = {pr["number"] for pr in body2["prs"]}
    assert ids0 | ids1 | ids2 == set(range(120))
    assert not (ids0 & ids1) and not (ids1 & ids2)


def test_node_prs_endpoint_caps_an_oversized_limit_request(tmp_path: Path) -> None:
    """A client (or bug) requesting an enormous `limit` must not recreate the exact
    "embed everything in one response" cost this todo removes from the main page."""
    store = _store_with_n_items(tmp_path, n=300)
    handler_cls = _make_handler(store, None)
    fake = _FakeHandler("/api/node-prs?path=cat&offset=0&limit=1000000")

    handler_cls.do_GET(fake)  # type: ignore[arg-type]

    body = json.loads(fake.body)
    assert len(body["prs"]) <= 200  # dashboard.server._MAX_PAGE_LIMIT


def test_node_prs_endpoint_defaults_a_missing_or_malformed_offset_limit(tmp_path: Path) -> None:
    store = _store_with_n_items(tmp_path, n=10)
    handler_cls = _make_handler(store, None)
    fake = _FakeHandler("/api/node-prs?path=cat&offset=not-a-number&limit=nope")

    handler_cls.do_GET(fake)  # must not raise / 500

    body = json.loads(fake.body)
    assert body["offset"] == 0
    assert body["total"] == 10
    assert len(body["prs"]) == 10


def test_node_prs_endpoint_serves_the_other_bucket_with_no_path_param(tmp_path: Path) -> None:
    store = _store_with_n_items(tmp_path, n=7, path=None)
    handler_cls = _make_handler(store, None)
    fake = _FakeHandler("/api/node-prs?path=&offset=0&limit=50")

    handler_cls.do_GET(fake)  # type: ignore[arg-type]

    assert json.loads(fake.body)["total"] == 7


def test_items_at_path_returns_the_pr_entry_shape(tmp_path: Path) -> None:
    store = _store_with_n_items(tmp_path, n=1)

    page, total = snap.items_at_path(store, ("cat",))

    assert total == 1
    assert page == [
        {
            "repo": "o/r",
            "number": 0,
            "title": "item 0",
            "url": "http://x/0",
            "state": "open",
            "type": "issue",
        }
    ]


def test_items_at_path_slices_before_mapping_to_pr_entry(tmp_path: Path) -> None:
    """A page request only transforms the rows actually returned, not every match -- offset
    and limit are applied before `pr_entry`, not after (a code-review efficiency finding)."""
    store = _store_with_n_items(tmp_path, n=10)

    page, total = snap.items_at_path(store, ("cat",), offset=8, limit=5)

    assert total == 10
    assert [pr["number"] for pr in page] == [8, 9]  # only 2 remain past offset 8


def test_items_at_path_limit_none_returns_every_remaining_item(tmp_path: Path) -> None:
    store = _store_with_n_items(tmp_path, n=5)

    page, total = snap.items_at_path(store, ("cat",), offset=2)

    assert total == 5
    assert [pr["number"] for pr in page] == [2, 3, 4]


# --------------------------------------------------------------------- precomputed snapshot


def test_snapshot_round_trips_through_json(tmp_path: Path) -> None:
    store = _store_with_n_items(tmp_path, n=3)

    built = snap.build_snapshot(store)
    loaded = snap.load_snapshot(snap.dump_snapshot(built))

    assert loaded == built


def test_a_page_renders_from_a_loaded_snapshot_without_a_store(tmp_path: Path) -> None:
    """DEVPLAN's own worked example: the precomputed snapshot round-trips and a page renders
    from it without a full Store.query() -- render_snapshot_page's own signature takes no
    Store at all, so this is provable by construction, not just by not calling a mock."""
    store = _store_with_n_items(tmp_path, n=3)
    raw = snap.dump_snapshot(snap.build_snapshot(store))

    loaded = snap.load_snapshot(raw)  # e.g. read back from a dashboard.json file
    page = render_snapshot_page(loaded)  # no `store` argument exists on this function at all

    assert "<h1>vllm-forager dashboard</h1>" in page
    assert '<span class="name">cat</span>' in page
    assert "item 0" in page


def test_build_snapshot_caps_prs_and_reports_the_true_total(tmp_path: Path) -> None:
    n = snap.DEFAULT_MAX_PRS_PER_NODE + 10
    store = _store_with_n_items(tmp_path, n)

    built = snap.build_snapshot(store)

    node = built["tree"][0]
    assert node["count"] == n
    assert len(node["prs"]) == snap.DEFAULT_MAX_PRS_PER_NODE
    assert node["prs_total"] == n
    assert node["prs_truncated"] is True


def test_build_snapshot_does_not_truncate_a_node_under_the_cap(tmp_path: Path) -> None:
    store = _store_with_n_items(tmp_path, n=3)

    node = snap.build_snapshot(store)["tree"][0]

    assert len(node["prs"]) == 3
    assert node["prs_total"] == 3
    assert node["prs_truncated"] is False
