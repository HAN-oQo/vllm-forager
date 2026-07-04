"""A minimal ``http.server`` wrapper around :mod:`dashboard.render` — no new dependency for
this thin M1 slice (no Flask/FastAPI): a read-only, single-page, no-auth local tool doesn't
need a web framework.

The `store` is constructed **once** by the caller and reused across every request — a
:class:`~src.store.base.Store` never caches its own reads (:class:`~src.store.jsonl_store.
JsonlStore` re-reads its files, :class:`~src.store.firestore_store.FirestoreStore` re-queries
Firestore, on every call), so a long-running dashboard process still reflects new items/
predictions without a restart. Constructing the store per-request instead (an earlier
version of this module did) would open a brand-new ``google.cloud.firestore.Client``/gRPC
channel on every single HTTP GET under ``STORE=firestore`` with nothing to close it — a
resource leak over a multi-hour ``serve_forever()`` run.

Known limitations of this thin M1 slice (not fixed here — acceptable for a single-user, local,
read-only tool; would need addressing before any wider/production use):
- Every request re-scans the **entire** store (`Store.query()`), re-reads the **entire**
  prediction log (one `get_state()` per historical prediction), and — since T1.5.5 —
  re-fetches a `get_state()` per taxonomy tree node for its stored summary — no caching,
  pagination, or limit anywhere. Cost grows linearly with KB size and multiplies per page view.
- The server is single-threaded (:class:`~http.server.HTTPServer`, not
  ``ThreadingHTTPServer``), so one slow request (a large KB, a slow Firestore round trip)
  blocks every other concurrent client until it completes.
"""

from __future__ import annotations

import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.store.base import Store

from .render import render_page


def _make_handler(store: Store) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler's naming convention)
            try:
                page = render_page(store).encode("utf-8")
            except Exception:  # noqa: BLE001 — any render/store failure must still get a response
                traceback.print_exc()
                body = b"500 Internal Server Error: failed to render the dashboard.\n"
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # quiet by default; this is a local dev tool, not a production server

    return Handler


def serve(store: Store, *, host: str = "127.0.0.1", port: int = 8765) -> int:
    """Start the dashboard's HTTP server and block, serving requests until interrupted.

    Returns a process exit code: 0 on a normal (interrupted) shutdown, 1 if `port` couldn't be
    bound (e.g. already in use) — the caller (:mod:`dashboard.__main__`) prints nothing further
    and just propagates this as its own exit code.
    """
    handler = _make_handler(store)
    try:
        httpd = HTTPServer((host, port), handler)
    except OSError as exc:
        print(f"vllm-forager dashboard: failed to bind {host}:{port}: {exc}")
        return 1
    with httpd:
        print(f"vllm-forager dashboard: http://{host}:{port}/")
        httpd.serve_forever()
    return 0
