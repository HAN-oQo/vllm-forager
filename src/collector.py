"""M0 minimal collector — incremental collection of GitHub issues/PRs.

Uses the GitHub REST API endpoint `GET /repos/{owner}/{repo}/issues`.
This endpoint returns issues and PRs together (PRs have a `pull_request` key), and the
`since` parameter allows incremental collection of items updated after a given time.

- Results are upserted (rewritten), not appended, into a per-repo JSONL
  (data/{owner}__{repo}.jsonl).
- The last collection time is stored per repo in data/state.json → the next run only
  fetches items after it.
- With GITHUB_TOKEN set, the rate limit goes from 60 to 5000/hr.
- Robustness (T0.10): each HTTP GET goes through :func:`_request`, which retries transient
  failures (5xx / timeout / connection) with exponential backoff and honors both primary and
  secondary (`Retry-After`) rate limits; records missing a required field are logged+skipped;
  one repo's failure is isolated (its cursor preserved) so the rest of the run continues.
- Review follow-ups (T0.12): a cursor stall (see `PAGE_CAP`) is recorded to the `data_quality`
  metric via `src.audit.record_stall`, not just stderr; `main`'s per-repo isolation logs a full
  traceback for non-network failures (a real bug) instead of the same message used for an
  expected transient skip.

Usage:
    python -m src.collector            # incremental collection for all repos
    python -m src.collector --full     # ignore state; re-fetch the lookback window
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # works even if python-dotenv isn't installed
    pass

from . import config
from .store.jsonl_store import JsonlStore, load_state, merge_jsonl, save_state

API = "https://api.github.com"


def _headers() -> dict:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "vllm-forager",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


# _load_state / _save_state / _merge_jsonl stay as thin wrappers over the store's format
# helpers (the single source of truth for the on-disk JSONL/state format) so the T0.3–T0.4
# named tests keep exercising the collector surface. `main()` now goes through the Store.
def _load_state() -> dict:
    return load_state(config.STATE_PATH)


def _save_state(state: dict) -> None:
    save_state(config.STATE_PATH, state)


def _sleep_for_rate_limit(resp: requests.Response) -> bool:
    """Wait until reset if we hit the **primary** rate limit. Returns True if we waited.

    The primary limit is signalled by ``403`` + ``X-RateLimit-Remaining: 0``; we sleep until
    ``X-RateLimit-Reset``. Secondary/abuse limits (``Retry-After``) are handled separately by
    :func:`_retry_after_wait`.
    """
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
        wait = max(reset - int(time.time()), 0) + 1
        print(f"  rate limit hit — waiting {wait}s", file=sys.stderr)
        time.sleep(wait)
        return True
    return False


def _retry_after_wait(resp: requests.Response) -> float | None:
    """Seconds to wait for a **secondary** rate limit (``Retry-After``), or None if not signalled.

    GitHub returns ``403``/``429`` with a ``Retry-After`` header (delta-seconds) for
    secondary/abuse rate limits — distinct from the primary limit handled by
    :func:`_sleep_for_rate_limit`. A ``403``/``429`` *without* ``Retry-After`` is a real error
    (auth, forbidden), not a rate limit, so we return None and let the caller surface it. A
    +1s cushion avoids retrying a hair too early.
    """
    if resp.status_code in (403, 429):
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0) + 1.0
            except ValueError:
                return None
    return None


def _backoff_wait(attempt: int, reason: str) -> None:
    """Sleep with exponential backoff before retrying a transient failure (attempt is 0-based)."""
    wait = config.BACKOFF_BASE_S * (2**attempt)
    print(
        f"  transient failure ({reason}) — retry {attempt + 1} in {wait:.0f}s",
        file=sys.stderr,
    )
    time.sleep(wait)


def _request(url: str, params: dict) -> requests.Response:
    """GET `url` with retry/backoff on transient failures and rate-limit handling.

    - Primary rate limit (``X-RateLimit-Remaining: 0``) → wait for reset, retry (no attempt used).
    - Secondary rate limit (``403``/``429`` + ``Retry-After``) → wait that long, retry (no attempt).
    - ``5xx`` and connection/timeout errors → exponential backoff, up to :data:`config.MAX_RETRIES`.
    - Anything else (``2xx`` / ``4xx`` incl. ``404``) is returned for the caller to interpret.

    Rate-limit waits do **not** consume a transient-retry (they aren't failures), but they are
    themselves bounded by :data:`config.MAX_RATE_LIMIT_RETRIES` so a stuck limiter (a persistent
    ``Retry-After`` or a past/stale reset) can't spin forever. Raises the underlying error
    (``HTTPError`` for a stuck ``5xx`` / exhausted rate limit, or the connection exception) once
    the relevant cap is reached — the per-repo isolation in :func:`main` then skips just that repo.
    """
    headers = _headers()  # static per run — build once, not on every retry
    attempt = 0
    rate_limit_waits = 0
    while True:
        try:
            resp = requests.get(
                url, headers=headers, params=params, timeout=config.REQUEST_TIMEOUT_S
            )
        except requests.RequestException as exc:  # timeout, connection reset, DNS — transient
            if attempt >= config.MAX_RETRIES:
                raise
            _backoff_wait(attempt, reason=type(exc).__name__)
            attempt += 1
            continue

        if _sleep_for_rate_limit(resp):  # primary limit — the helper already waited for reset
            rate_limit_waits += 1
            if rate_limit_waits > config.MAX_RATE_LIMIT_RETRIES:
                resp.raise_for_status()  # give up: a stuck limiter must not hang the run
            continue
        wait = _retry_after_wait(resp)
        if wait is not None:  # secondary limit
            rate_limit_waits += 1
            if rate_limit_waits > config.MAX_RATE_LIMIT_RETRIES:
                resp.raise_for_status()
            print(f"  secondary rate limit — waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            if attempt >= config.MAX_RETRIES:
                resp.raise_for_status()  # give up: surface the 5xx to the caller
            _backoff_wait(attempt, reason=f"HTTP {resp.status_code}")
            attempt += 1
            continue
        return resp


# A collected record must carry these before we store it — the identity/sort keys everything
# downstream depends on. Anything missing one is logged and skipped rather than persisted.
REQUIRED_FIELDS = ("number", "url", "updated_at", "type")


# GitHub's list endpoints refuse deep pagination past ~1000 items (page * per_page),
# returning HTTP 422. To collect more, we walk the `updated_at` cursor: page within a
# window until it fills PAGE_CAP pages, then restart from the last item's updated_at.
PAGE_CAP = 10


def fetch_repo(
    slug: str,
    since: str,
    *,
    on_stall: Callable[[str, str], None] | None = None,
) -> list[dict]:
    """Fetch all issues + PRs for `slug` updated at/after `since`.

    Works around GitHub's ~1000-item deep-pagination limit by advancing the `since`
    cursor (sort=updated, asc) whenever a window fills PAGE_CAP pages. Records are keyed
    by issue number, so the inclusive-boundary re-fetch between windows is deduped.

    `on_stall`, if given, is called with ``(slug, window_since)`` when the cursor cannot
    advance (>``PAGE_CAP``*``PER_PAGE`` items share one ``updated_at``) — `main` uses this
    to record the gap as a ``data_quality`` metric (T0.12) instead of only a stderr line.
    """
    owner, repo = slug.split("/", 1)
    url = f"{API}/repos/{owner}/{repo}/issues"
    collected: dict[int, dict] = {}
    window_since = since

    while True:
        last_updated = None
        page = 1
        while page <= PAGE_CAP:
            params: dict[str, Any] = {
                "since": window_since,
                "state": "all",
                "per_page": config.PER_PAGE,
                "sort": "updated",
                "direction": "asc",
                "page": page,
            }
            resp = _request(url, params)  # retry/backoff + rate-limit handling live here
            if resp.status_code == 404:
                print(f"  !! {slug} 404 — check the slug", file=sys.stderr)
                return list(collected.values())
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                return list(collected.values())
            for it in batch:
                # Advance the cursor on every item with a timestamp — even one we skip below —
                # so a window of malformed items can't stall forward progress.
                last_updated = it.get("updated_at") or last_updated
                rec = _normalize(it, slug)
                missing = [f for f in REQUIRED_FIELDS if rec.get(f) is None]
                if missing:
                    print(
                        f"  !! {slug} skipping malformed item #{it.get('number')}: "
                        f"missing {missing}",
                        file=sys.stderr,
                    )
                    continue
                collected[rec["number"]] = rec
            at = (last_updated or window_since)[:10]
            print(
                f"    … {slug}: +{len(batch)} ({len(collected)} so far, up to {at})",
                file=sys.stderr,
            )
            if len(batch) < config.PER_PAGE:
                return list(collected.values())
            page += 1

        # Window filled PAGE_CAP pages — advance the cursor to continue past the cap.
        if not last_updated or last_updated == window_since:
            # No forward progress (e.g. >1000 items share one timestamp); stop rather
            # than spin forever.
            print(f"  !! {slug} cursor stalled at {window_since} — stopping early", file=sys.stderr)
            if on_stall is not None:
                on_stall(slug, window_since)
            return list(collected.values())
        window_since = last_updated


def _normalize(it: dict, slug: str) -> dict:
    """Minimal normalization of the collection schema. Extended later in RAG/classification."""
    return {
        "repo": slug,
        "number": it.get("number"),
        "type": "pr" if "pull_request" in it else "issue",
        "title": it.get("title"),
        "state": it.get("state"),
        "labels": [lbl.get("name") for lbl in it.get("labels", [])],
        "created_at": it.get("created_at"),
        "updated_at": it.get("updated_at"),
        "url": it.get("html_url"),
        "body": (it.get("body") or "")[: config.BODY_MAX_CHARS],
    }


def _merge_jsonl(path, records: list[dict]) -> int:
    """Upsert by `number`, then rewrite atomically. Returns the total record count.

    Thin wrapper over :func:`src.store.jsonl_store.merge_jsonl` (see the note above): it
    tolerates a corrupt/partial line on read and rewrites atomically.
    """
    return merge_jsonl(path, records)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full", action="store_true", help="ignore state; re-fetch the lookback window"
    )
    args = ap.parse_args(argv)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(config.DATA_DIR)  # write via the pluggable Store interface (T0.6)
    now_dt = datetime.now(timezone.utc)
    now = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # First run / --full: start from a rolling lookback window, not the beginning of time.
    default_since = (now_dt - timedelta(days=config.INITIAL_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    if not os.getenv("GITHUB_TOKEN"):
        print(
            "warning: no GITHUB_TOKEN — rate limit 60/hr. Setting up .env is recommended.",
            file=sys.stderr,
        )

    # Local import: audit.py imports collector._headers, so importing it at module level
    # here would create a circular import; deferring to call time (after collector has
    # fully loaded) breaks the cycle.
    from . import audit

    def _on_stall(stalled_slug: str, window_since: str) -> None:
        # A stall means the REST cursor can't advance (>PAGE_CAP*PER_PAGE items share one
        # timestamp) — fall back to GraphQL to at least learn the window's true count and
        # persist it as a flagged data_quality record (T0.12), not just a stderr line. This
        # diagnostic must not itself abort collection of the records already fetched.
        # `remote_fetcher=audit.remote_counts` is a live attribute lookup at call time (not
        # record_stall's early-bound default), so tests can monkeypatch audit.remote_counts.
        try:
            audit.record_stall(
                store,
                stalled_slug,
                window_since,
                remote_fetcher=audit.remote_counts,
                checked_at=now,
            )
        except Exception as exc:
            print(f"  !! {stalled_slug} failed to record stall: {exc}", file=sys.stderr)

    for repo in config.REPOS:
        slug = repo["slug"]
        # --full ignores the stored cursor and re-fetches the whole lookback window.
        since = default_since if args.full else (store.get_state(slug) or default_since)
        print(f"[{slug}] since {since} …")
        try:
            records = fetch_repo(slug, since, on_stall=_on_stall)
        except requests.RequestException as exc:  # network/HTTP — expected & transient
            print(f"  !! {slug} failed: {exc} — skipping (cursor preserved)", file=sys.stderr)
            continue
        except Exception:  # NOT a network failure — a real bug; make it loud, then isolate
            print(f"  !! {slug} UNEXPECTED failure — skipping (cursor preserved):", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue
        totals = store.upsert_items(records)  # {repo: post-upsert total} — no re-read needed
        store.set_state(slug, now)  # persist progress per repo so a later failure can't lose it
        # `slug` is absent from totals only when there were no records to write this cycle.
        total_str = f" · {totals[slug]} total" if slug in totals else ""
        print(f"  +{len(records)} updated{total_str} → {slug.replace('/', '__')}.jsonl")

    print("done.")


if __name__ == "__main__":
    main()
