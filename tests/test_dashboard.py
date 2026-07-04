"""Tests for the thin read-only dashboard (T1.9) — offline & deterministic.

Per the DEVPLAN todo: seed a store fixture on tmp_path, assert the render functions return
the report body + correct trend series. A live server smoke is ``@pytest.mark.integration``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from dashboard.render import (
    latest_report_path,
    render_forecasts,
    render_page,
    render_report,
    render_trends,
)
from dashboard.server import _make_handler, serve
from src.agents.forecaster import Prediction, record_prediction
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m1

_ITEMS = [
    {
        "repo": "ROCm/vllm",
        "number": 1,
        "title": "hipBLAS build fails",
        "url": "https://github.com/ROCm/vllm/issues/1",
        "created_at": "2026-01-05T00:00:00Z",
        "category": "build",
    },
    {
        "repo": "ROCm/vllm",
        "number": 2,
        "title": "another build issue",
        "url": "https://github.com/ROCm/vllm/issues/2",
        "created_at": "2026-01-06T00:00:00Z",
        "category": "build",
    },
    {
        "repo": "vllm-project/vllm",
        "number": 3,
        "title": "FP8 quantization",
        "url": "https://github.com/vllm-project/vllm/issues/3",
        "created_at": "2026-01-12T00:00:00Z",
        "category": "quantization",
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


# --------------------------------------------------------------------- render_report


def test_render_report_reads_latest_report_file(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "2026-W01.md").write_text("# stale report", encoding="utf-8")
    (reports_dir / "2026-W02.md").write_text("# fresh report", encoding="utf-8")

    assert latest_report_path(reports_dir) == reports_dir / "2026-W02.md"
    assert render_report(reports_dir) == "# fresh report"


def test_render_report_no_reports_yet_is_a_placeholder(tmp_path: Path) -> None:
    assert "No report has been generated yet" in render_report(tmp_path / "reports")


# --------------------------------------------------------------------- render_trends/forecasts


def test_render_trends_matches_seeded_categories(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    series = render_trends(store)

    assert series == {
        "build": {"2026-W02": 2},
        "quantization": {"2026-W03": 1},
    }


def test_render_forecasts_returns_recorded_predictions(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    predictions = render_forecasts(store)

    assert predictions == [_PREDICTION]


# --------------------------------------------------------------------- render_page


def test_render_page_includes_report_trends_and_forecasts(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "2026-W02.md").write_text("# weekly digest body", encoding="utf-8")

    page = render_page(store, reports_dir)

    assert "weekly digest body" in page
    assert "build" in page
    assert "quantization" in page
    assert "PR #3 will merge by end of Q1" in page


def test_render_page_escapes_untrusted_content(tmp_path: Path) -> None:
    """Item/report/prediction text originates from GitHub — must be HTML-escaped, not
    injected raw, since the dashboard has no auth and renders whatever the KB contains."""
    store = JsonlStore(tmp_path)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "2026-W01.md").write_text("<script>alert(1)</script>", encoding="utf-8")

    page = render_page(store, reports_dir)

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


# --------------------------------------------------------------------- server error handling


class _FakeResponse:
    """A duck-typed stand-in for BaseHTTPRequestHandler's response-writing surface.

    do_GET only touches ``send_response``/``send_header``/``end_headers``/``wfile.write`` — this
    records those calls so the exception-handling branch can be tested without opening a real
    socket or constructing a full (socket-backed) BaseHTTPRequestHandler.
    """

    def __init__(self) -> None:
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


def test_do_get_returns_500_on_render_failure(tmp_path: Path) -> None:
    """A corrupt store record must yield a real HTTP 500, not a silently dropped connection."""
    store = JsonlStore(tmp_path)
    store.set_state("prediction_count", "not-a-number")  # forecaster._parse_count raises
    handler_cls = _make_handler(store, tmp_path / "reports")
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
        exit_code = serve(store, tmp_path / "reports", host="127.0.0.1", port=port)
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
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "2026-W02.md").write_text("# weekly digest body", encoding="utf-8")

    handler = _make_handler(store, reports_dir)
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.1)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "weekly digest body" in body
    finally:
        httpd.shutdown()
        thread.join()


def test_serve_is_importable() -> None:
    # smoke: the serve() entry point exists with the expected signature (not called here —
    # it blocks forever; see test_dashboard_live_smoke for the real integration test).
    assert callable(serve)
