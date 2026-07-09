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

T5.16 adds one more route, ``GET /api/node-prs``: a paginated, JSON view of one tree node's own
leaf rows past what the page's own capped inline list shows (see :mod:`dashboard.snapshot`'s
``DEFAULT_MAX_PRS_PER_NODE`` and :mod:`dashboard.render`'s ``_more_prs_html``/its embedded fetch
handler). Query params: ``path`` (the node's category path, ``src.taxonomy.LEVEL_SEPARATOR``-
joined — e.g. ``rl > post-training``, URL-encoded), ``offset`` (default ``0``), ``limit``
(default :data:`_DEFAULT_PAGE_LIMIT`, capped at :data:`_MAX_PAGE_LIMIT` so a malformed/malicious
request can't force one page to re-embed the exact "render everything" cost this todo removes
from the main page). Response: ``{"prs": [...], "total": N, "offset": O, "limit": L}``.

T5.12 replaces the single bundled page with a tabbed console: ``GET /`` and ``GET /tab/<name>``
(`name` one of :data:`~dashboard.render.TAB_LABELS`'s own slugs — an unrecognized name degrades
to the default "issues" tab, not a 404) each render one full page via
:func:`~dashboard.render.render_tab_page`, reading ``?report=`` (Reports/Trends: which archived
report to open) and ``?repo=&number=`` (Attempts: which worked candidate to open) the same
degrade-on-malformed way ``/api/node-prs`` already reads its own query params. Two ``POST``
routes are this dashboard's only writes (mirrors :mod:`dashboard.select`/:mod:`dashboard.
review`'s own "only two write paths" framing): ``POST /tab/candidates/decide`` (T5.10's "Work
this"/"Skip") and ``POST /tab/attempts/decide`` (T5.6's approve/hold) — both plain HTML-form
submissions (``application/x-www-form-urlencoded``, no JS, no JSON body), each answered with a
303 redirect back to the relevant ``GET`` tab so a page refresh never resubmits the form.

Known limitations of this thin M1 slice (not fixed here — acceptable for a single-user, local,
read-only tool; would need addressing before any wider/production use — see
:mod:`dashboard.snapshot`'s own docstring for the full detail behind these two):
- Every request re-scans the **entire** store at least twice (`build_snapshot`'s own two
  independent full scans), and every ``/api/node-prs`` request (including every "show N more"
  click) triggers **another** full, unfiltered scan of its own (`items_at_path`) — pagination
  here bounds response size and per-row transform work, not the number of store reads. No
  caching layer sits in front of a request yet; a scheduled precomputed-snapshot job is what
  would fix the read cost, not done here.
- Pagination is offset-based against a live-re-sorted collection — an item's `updated_at`
  changing between an initial page load and a later "show more" click for the same node can
  shift the sort order enough to duplicate or skip a row on the next page.
- The server is single-threaded (:class:`~http.server.HTTPServer`, not
  ``ThreadingHTTPServer``), so one slow request (a large KB, a slow Firestore round trip)
  blocks every other concurrent client until it completes.
"""

from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, quote, urlsplit

from src.store.base import Store
from src.taxonomy import LEVEL_SEPARATOR

from . import review, select
from .render import TAB_LABELS, render_tab_page
from .snapshot import items_at_path

_TAB_NAMES = frozenset(slug for slug, _ in TAB_LABELS)

# A page past this many rows must still be paginated further, not returned in one response --
# without a cap, a client (or a bug) requesting `limit=1000000` would recreate exactly the
# "the whole node in one payload" cost T5.16 removes from the main page.
_DEFAULT_PAGE_LIMIT = 50
_MAX_PAGE_LIMIT = 200


def _parse_int(values: list[str] | None, default: int) -> int:
    """The first value in `values` (a `urllib.parse.parse_qs` result) as an int, or `default`
    if absent/malformed — a bad `?offset=`/`?limit=` must degrade to the default, not 500."""
    if not values:
        return default
    try:
        return int(values[0])
    except ValueError:
        return default


def _respond(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
    """Write a complete HTTP response (status + headers + body) to `handler`'s socket."""
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _first(values: list[str] | None) -> str | None:
    """The first entry in `values` (a `urllib.parse.parse_qs` result), or `None` if `values`
    is missing/empty — one consistent "not given" sentinel, avoiding the `(x or [None])[0]`
    idiom (which mypy can't type-narrow past `list[str] | None` correctly)."""
    return values[0] if values else None


def _open_candidate_param(query: dict[str, list[str]]) -> tuple[str, int] | None:
    """`(repo, number)` from ``?repo=&number=``, or `None` if either is missing/malformed --
    degrades rather than 500s on a stale/hand-edited Attempts-tab URL."""
    repo = _first(query.get("repo"))
    number_raw = _first(query.get("number"))
    if not repo or number_raw is None:
        return None
    try:
        return repo, int(number_raw)
    except ValueError:
        return None


def _serve_tab(handler: BaseHTTPRequestHandler, store: Store, tab: str) -> None:
    """``GET /`` (aliases the default tab) and ``GET /tab/<name>`` — T5.12's tabbed console:
    one named tab's full page, reading `?report=` (Reports/Trends) and `?repo=&number=`
    (Attempts) the same way :func:`_serve_node_prs` already reads its own query params."""
    query = parse_qs(urlsplit(handler.path).query)
    report_stamp = _first(query.get("report"))
    open_candidate = _open_candidate_param(query)
    try:
        page = render_tab_page(
            store, tab, report_stamp=report_stamp, open_candidate=open_candidate
        ).encode("utf-8")
    except Exception:  # noqa: BLE001 — any render/store failure must still get a response
        traceback.print_exc()
        _respond(
            handler,
            500,
            b"500 Internal Server Error: failed to render the dashboard.\n",
            "text/plain; charset=utf-8",
        )
        return
    _respond(handler, 200, page, "text/html; charset=utf-8")


def _read_form(handler: BaseHTTPRequestHandler) -> dict[str, list[str]]:
    """Parse an ``application/x-www-form-urlencoded`` POST body off `handler`'s socket -- the
    only body shape either write action's plain HTML ``<form>`` (see `dashboard.render`'s
    `_candidate_row_html`/`_approve_hold_form_html`) ever submits."""
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b""
    return parse_qs(raw.decode("utf-8"))


def _form_str(form: dict[str, list[str]], key: str) -> str | None:
    return _first(form.get(key))


def _form_int(form: dict[str, list[str]], key: str) -> int | None:
    raw = _form_str(form, key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _redirect(handler: BaseHTTPRequestHandler, location: str) -> None:
    """303 See Other — the standard POST-then-GET pattern, so refreshing the page after a
    write action never resubmits the same form."""
    handler.send_response(303)
    handler.send_header("Location", location)
    handler.send_header("Content-Length", "0")
    handler.end_headers()


def _bad_request(handler: BaseHTTPRequestHandler, message: str) -> None:
    _respond(handler, 400, f"400 Bad Request: {message}\n".encode(), "text/plain; charset=utf-8")


def _serve_candidate_decision(handler: BaseHTTPRequestHandler, store: Store) -> None:
    """``POST /tab/candidates/decide`` — T5.10's "Work this"/"Skip" write action, one of only
    two write paths the whole dashboard has (see `dashboard.select`/`dashboard.review`'s own
    module docstrings)."""
    form = _read_form(handler)
    repo = _form_str(form, "repo")
    number = _form_int(form, "number")
    decision = _form_str(form, "decision")
    if repo is None or number is None or decision not in ("selected", "skip"):
        _bad_request(handler, "repo, number, and a valid decision are required")
        return
    if decision == "selected":
        select.select_candidate(store, repo, number)
    else:
        select.skip_candidate(store, repo, number)
    _redirect(handler, "/tab/candidates")


def _serve_attempt_decision(handler: BaseHTTPRequestHandler, store: Store) -> None:
    """``POST /tab/attempts/decide`` — T5.6's approve/hold write action, reached from the
    Attempts tab's own detail view (`dashboard.render._approve_hold_form_html`). `submit` is
    never exposed here — `review.review_decision`'s own hardcoded `submit=False` (see that
    module's docstring for the exact incident this guards against) is untouched by this route."""
    form = _read_form(handler)
    repo = _form_str(form, "repo")
    number = _form_int(form, "number")
    approve_raw = _form_str(form, "approve")
    if repo is None or number is None or approve_raw not in ("true", "false"):
        _bad_request(handler, "repo, number, and a valid approve flag are required")
        return
    review.review_decision(store, repo, number, approve=(approve_raw == "true"))
    _redirect(handler, f"/tab/attempts?repo={quote(repo)}&number={number}")


def _serve_node_prs(handler: BaseHTTPRequestHandler, store: Store) -> None:
    """``GET /api/node-prs`` — see this module's own docstring.

    The whole response (the read, the JSON encode, and the write) is one try/except, matching
    :func:`_serve_page`'s own "any failure still gets a response" invariant — a first version
    of this only guarded the `items_at_path` call, so a `json.dumps` failure on an unexpected
    item shape (a code-review finding) would have dropped the connection instead.
    """
    query = parse_qs(urlsplit(handler.path).query)
    raw_path = (query.get("path") or [""])[0]
    path = tuple(level for level in raw_path.split(LEVEL_SEPARATOR) if level)
    offset = max(_parse_int(query.get("offset"), 0), 0)
    limit = min(max(_parse_int(query.get("limit"), _DEFAULT_PAGE_LIMIT), 1), _MAX_PAGE_LIMIT)
    try:
        page, total = items_at_path(store, path, offset=offset, limit=limit)
        body = json.dumps({"prs": page, "total": total, "offset": offset, "limit": limit}).encode(
            "utf-8"
        )
    except Exception:  # noqa: BLE001 — a corrupt KB record must still get a response
        traceback.print_exc()
        _respond(
            handler,
            500,
            b"500 Internal Server Error: failed to read this node.\n",
            "text/plain; charset=utf-8",
        )
        return
    _respond(handler, 200, body, "application/json; charset=utf-8")


def _make_handler(store: Store) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler's naming convention)
            path = urlsplit(self.path).path
            if path == "/api/node-prs":
                _serve_node_prs(self, store)
            elif path.startswith("/tab/"):
                tab = path[len("/tab/") :].split("/", 1)[0]
                _serve_tab(self, store, tab if tab in _TAB_NAMES else "issues")
            else:
                _serve_tab(self, store, "issues")

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/tab/candidates/decide":
                _serve_candidate_decision(self, store)
            elif path == "/tab/attempts/decide":
                _serve_attempt_decision(self, store)
            else:
                _respond(self, 404, b"404 Not Found\n", "text/plain; charset=utf-8")

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
