"""Tests for the thin read-only dashboard (T1.9, tree UI since T1.5.5) — offline & deterministic.

Per T1.5.5's DEVPLAN todo: a seeded tree → render produces nested nodes + summaries + evidence
links; filtering to a term keeps only matching leaves. The filter itself is client-side
JavaScript (no headless-browser tool is available in this environment to execute it), so
"filtering keeps only matching leaves" is tested here as: the filter script is embedded and
targets the right elements, and every rendered PR row carries the text (id + title) the filter
matches substrings against — see ``test_tree_filter_script_and_matchable_text_are_present``.

A live server smoke is ``@pytest.mark.integration``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from dashboard import render
from dashboard.render import _safe_href, _state_chip, render_forecasts, render_page, render_trends
from dashboard.server import _make_handler, serve
from src import llm
from src.agents import summarizer
from src.agents.forecaster import Prediction, record_prediction
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m1

_ITEMS = [
    {
        "repo": "ROCm/vllm",
        "number": 1,
        "type": "issue",
        "title": "hipBLAS build fails",
        "state": "open",
        "url": "https://github.com/ROCm/vllm/issues/1",
        "created_at": "2026-01-05T00:00:00Z",
        "category": "build",
        "path": ["build"],
    },
    {
        "repo": "ROCm/vllm",
        "number": 2,
        "type": "pr",
        "title": "another build issue",
        "state": "closed",
        "url": "https://github.com/ROCm/vllm/pull/2",
        "created_at": "2026-01-06T00:00:00Z",
        "category": "build",
        "path": ["build"],
    },
    {
        "repo": "vllm-project/vllm",
        "number": 3,
        "type": "pr",
        "title": "FP8 quantization",
        "state": "open",
        "url": "https://github.com/vllm-project/vllm/pull/3",
        "created_at": "2026-01-12T00:00:00Z",
        "category": "quantization > FP8",
        "path": ["quantization", "FP8"],
    },
]

_PREDICTION = Prediction(
    claim="PR #3 will merge by end of Q1",
    resolution_rule="merged into main",
    prob=0.7,
    due_date="2026-03-31T00:00:00Z",
    evidence=("https://github.com/vllm-project/vllm/pull/3",),
    created_at="2026-01-15T00:00:00Z",
)


def _seeded_store(tmp_path: Path) -> JsonlStore:
    store = JsonlStore(tmp_path)
    store.upsert_items(_ITEMS)
    record_prediction(store, _PREDICTION)
    return store


# --------------------------------------------------------------------- render_trends/forecasts


def test_render_trends_matches_seeded_categories(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    series = render_trends(store)

    assert series == {
        "build": {"2026-W02": 2},
        "quantization > FP8": {"2026-W03": 1},
    }


def test_render_forecasts_returns_recorded_predictions(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    predictions = render_forecasts(store)

    assert predictions == [_PREDICTION]


# --------------------------------------------------------------------- _state_chip


def test_state_chip_issue() -> None:
    assert _state_chip({"type": "issue", "state": "open"}) == ("issue", "issue")


def test_state_chip_open_pr() -> None:
    assert _state_chip({"type": "pr", "state": "open"}) == ("open", "open pr")


def test_state_chip_closed_pr_is_merged() -> None:
    assert _state_chip({"type": "pr", "state": "closed"}) == ("merged", "merged")


def test_state_chip_missing_state_defaults_to_open() -> None:
    """A malformed/legacy record with no `state` at all renders as "open pr", not "merged" —
    claiming an unknown state is open is the less misleading of the two guesses."""
    assert _state_chip({"type": "pr"}) == ("open", "open pr")


# --------------------------------------------------------------------- _safe_href


def test_safe_href_allows_http_and_https() -> None:
    assert _safe_href("http://x/1") == "http://x/1"
    assert _safe_href("https://x/1") == "https://x/1"


def test_safe_href_rejects_javascript_scheme() -> None:
    assert _safe_href("javascript:alert(1)") == ""


def test_render_page_never_links_a_javascript_url(tmp_path: Path) -> None:
    """A malformed/non-GitHub item `url` must never become a clickable href — this dashboard
    has no auth and renders whatever the KB contains."""
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            {
                "repo": "o/r",
                "number": 1,
                "type": "pr",
                "title": "x",
                "state": "open",
                "url": "javascript:alert(document.cookie)",
                "created_at": "2026-01-01T00:00:00Z",
                "path": ["build"],
            }
        ]
    )

    page = render_page(store)

    assert "javascript:" not in page


# --------------------------------------------------------------------- report tree section


def test_tree_section_renders_nested_nodes_with_summary_and_evidence_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1.5.5's named scenario: a seeded tree → render produces nested nodes + summaries +
    evidence links."""
    store = _seeded_store(tmp_path)
    monkeypatch.setattr(
        llm, "complete", lambda *a, **k: {"summary": "FP8 quantization work is landing."}
    )
    summarizer.summarize_store(store)

    page = render_page(store)

    # nested nodes: both root categories, and the "FP8" child under "quantization"
    assert '<span class="name">build</span>' in page
    assert '<span class="name">quantization</span>' in page
    assert '<span class="name">FP8</span>' in page
    # rolled-up counts
    assert '<span class="count">2</span>' in page  # "build" has 2 items
    # the node summary (T1.5.3)
    assert "FP8 quantization work is landing." in page
    # evidence links: every leaf PR/issue cites its own URL
    assert "https://github.com/ROCm/vllm/issues/1" in page
    assert "https://github.com/vllm-project/vllm/pull/3" in page
    # state chips
    assert '<span class="chip issue">issue</span>' in page
    assert '<span class="chip merged">merged</span>' in page  # ROCm/vllm#2, closed
    assert '<span class="chip open">open pr</span>' in page  # vllm#3, open


def test_tree_section_empty_store_shows_placeholder(tmp_path: Path) -> None:
    """`tree_from_store` returns `[]` only for a literally empty store — see
    `_tree_section_html`'s own docstring for why unclassified-but-present items don't hit
    this (they render under `Other` instead; see the test below)."""
    store = JsonlStore(tmp_path)

    page = render_page(store)

    assert "No classified items yet" in page
    assert "<details" not in page


def test_tree_section_unclassified_items_render_under_other(tmp_path: Path) -> None:
    """An item with no `path` yet renders under a real `Other` node, not the placeholder —
    `build_tree` (T1.5.4) folds path-less items there rather than treating them as empty."""
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            {
                "repo": "o/r",
                "number": 1,
                "type": "issue",
                "title": "x",
                "state": "open",
                "url": "http://x/1",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
    )

    page = render_page(store)

    assert '<span class="name">Other</span>' in page
    assert "run <code>python -m src.analyze</code>" not in page


def test_tree_filter_script_and_matchable_text_are_present(tmp_path: Path) -> None:
    """The filter itself is client-side JS this test suite can't execute — this instead
    verifies (1) the filter script targets #tree/.pr/#q as the render functions produce them,
    and (2) every PR row's rendered text contains its id+title, which is exactly what a
    substring filter needs to match against."""
    store = _seeded_store(tmp_path)

    page = render_page(store)

    assert '<input id="q"' in page
    assert "getElementById('q')" in page
    assert "querySelectorAll('#tree .pr')" in page
    assert "classList.toggle('hidden'" in page
    assert "ROCm/vllm#1" in page and "hipBLAS build fails" in page  # matchable id + title


def test_render_page_escapes_untrusted_content(tmp_path: Path) -> None:
    """Item text originates from GitHub — must be HTML-escaped, not injected raw, since the
    dashboard has no auth and renders whatever the KB contains."""
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            {
                "repo": "o/r",
                "number": 1,
                "type": "issue",
                "title": "<script>alert(1)</script>",
                "state": "open",
                "url": "http://x/1",
                "created_at": "2026-01-01T00:00:00Z",
                "path": ["<script>alert(2)</script>"],
            }
        ]
    )

    page = render_page(store)

    assert "<script>alert(1)</script>" not in page
    assert "<script>alert(2)</script>" not in page
    assert "&lt;script&gt;" in page


def test_render_page_includes_trends_and_forecasts(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    page = render_page(store)

    assert "build" in page
    assert "quantization" in page
    assert "PR #3 will merge by end of Q1" in page


# --------------------------------------------------------------------- server error handling


class _FakeResponse:
    """A duck-typed stand-in for BaseHTTPRequestHandler's response-writing surface.

    do_GET only touches ``self.path``/``send_response``/``send_header``/``end_headers``/
    ``wfile.write`` — this records those calls so the exception-handling branch can be tested
    without opening a real socket or constructing a full (socket-backed)
    BaseHTTPRequestHandler.
    """

    def __init__(self, path: str = "/") -> None:
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
        def __init__(self, outer: _FakeResponse) -> None:
            self._outer = outer

        def write(self, data: bytes) -> None:
            self._outer.body += data

    @property
    def wfile(self) -> _FakeResponse._Wfile:
        return self._Wfile(self)


def test_do_get_returns_500_on_render_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render/store failure must yield a real HTTP 500, not a silently dropped connection.

    T5.12 repointed `GET /` at the Issues tab (`render.render_issues_tab` -> `api.tree`), which
    no longer touches the prediction log at all (that's the whole point of the fix: a tree-only
    read must not depend on, or crash over, an unrelated corrupt prediction record) -- so the
    original `prediction_count`-corruption trigger this test used no longer reaches `/` at all.
    Monkeypatching `api.tree` to raise exercises the same "any failure still gets a response"
    invariant directly, regardless of which store operation happens to fail."""
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(render.api, "tree", lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    handler_cls = _make_handler(store, None)
    fake = _FakeResponse()

    handler_cls.do_GET(fake)  # type: ignore[arg-type]

    assert fake.status == 500
    assert b"500 Internal Server Error" in fake.body


def test_serve_returns_1_on_port_already_in_use(tmp_path: Path) -> None:
    import socket

    store = JsonlStore(tmp_path)
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        exit_code = serve(store, host="127.0.0.1", port=port)
        assert exit_code == 1
    finally:
        blocker.close()


# --------------------------------------------------------------------- live smoke


@pytest.mark.integration
def test_dashboard_live_smoke(tmp_path: Path) -> None:
    import threading
    import time
    from http.server import HTTPServer

    store = _seeded_store(tmp_path)

    handler = _make_handler(store, None)
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.1)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "hipBLAS build fails" in body
    finally:
        httpd.shutdown()
        thread.join()


def test_serve_is_importable() -> None:
    # smoke: the serve() entry point exists with the expected signature (not called here —
    # it blocks forever; see test_dashboard_live_smoke for the real integration test).
    assert callable(serve)
