"""A minimal ``http.server`` wrapper around :mod:`dashboard.render` — no new dependency for
this thin M1 slice (no Flask/FastAPI): a read-only, single-page, no-auth local tool doesn't
need a web framework.

`store_factory` is called **fresh on every request** (not once at startup) so a long-running
dashboard process reflects new items/predictions written by other processes (the collector,
the forecaster) without needing a restart — the same "always re-read, never cache" contract
:class:`~src.store.jsonl_store.JsonlStore`/``FirestoreStore`` already give every other caller.
"""

from __future__ import annotations

from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from src.store.base import Store

from .render import render_page


def _make_handler(
    store_factory: Callable[[], Store], reports_dir: Path
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler's naming convention)
            page = render_page(store_factory(), reports_dir).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # quiet by default; this is a local dev tool, not a production server

    return Handler


def serve(
    store_factory: Callable[[], Store],
    reports_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Start the dashboard's HTTP server and block, serving requests until interrupted."""
    handler = _make_handler(store_factory, reports_dir)
    with HTTPServer((host, port), handler) as httpd:
        print(f"vllm-forager dashboard: http://{host}:{port}/")
        httpd.serve_forever()
