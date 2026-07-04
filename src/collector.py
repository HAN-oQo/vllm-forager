"""M0 minimal collector — incremental collection of GitHub issues/PRs.

Uses the GitHub REST API endpoint `GET /repos/{owner}/{repo}/issues`.
This endpoint returns issues and PRs together (PRs have a `pull_request` key), and the
`since` parameter allows incremental collection of items updated after a given time.

- Results are upserted (rewritten), not appended, into a per-repo JSONL
  (data/{owner}__{repo}.jsonl).
- The last collection time is stored per repo in data/state.json → the next run only
  fetches items after it.
- With GITHUB_TOKEN set, the rate limit goes from 60 to 5000/hr.

Usage:
    python -m src.collector            # incremental collection for all repos
    python -m src.collector --full     # ignore state; re-fetch the lookback window
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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
    """Wait until reset if we hit the rate limit. Returns True if we waited."""
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
        wait = max(reset - int(time.time()), 0) + 1
        print(f"  rate limit hit — waiting {wait}s", file=sys.stderr)
        time.sleep(wait)
        return True
    return False


# GitHub's list endpoints refuse deep pagination past ~1000 items (page * per_page),
# returning HTTP 422. To collect more, we walk the `updated_at` cursor: page within a
# window until it fills PAGE_CAP pages, then restart from the last item's updated_at.
PAGE_CAP = 10


def fetch_repo(slug: str, since: str) -> list[dict]:
    """Fetch all issues + PRs for `slug` updated at/after `since`.

    Works around GitHub's ~1000-item deep-pagination limit by advancing the `since`
    cursor (sort=updated, asc) whenever a window fills PAGE_CAP pages. Records are keyed
    by issue number, so the inclusive-boundary re-fetch between windows is deduped.
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
            resp = requests.get(url, headers=_headers(), params=params, timeout=30)
            if _sleep_for_rate_limit(resp):
                continue
            if resp.status_code == 404:
                print(f"  !! {slug} 404 — check the slug", file=sys.stderr)
                return list(collected.values())
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                return list(collected.values())
            for it in batch:
                rec = _normalize(it, slug)
                collected[rec["number"]] = rec
                last_updated = it.get("updated_at") or last_updated
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
        "body": (it.get("body") or "")[:4000],
    }


def _merge_jsonl(path, records: list[dict]) -> int:
    """Upsert by `number`, then rewrite atomically. Returns the total record count.

    Thin wrapper over :func:`src.store.jsonl_store.merge_jsonl` (see the note above): it
    tolerates a corrupt/partial line on read and rewrites atomically.
    """
    return merge_jsonl(path, records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full", action="store_true", help="ignore state; re-fetch the lookback window"
    )
    args = ap.parse_args()

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

    for repo in config.REPOS:
        slug = repo["slug"]
        # --full ignores the stored cursor and re-fetches the whole lookback window.
        since = default_since if args.full else (store.get_state(slug) or default_since)
        print(f"[{slug}] since {since} …")
        try:
            records = fetch_repo(slug, since)
        except Exception as exc:  # isolate: one repo's failure must not abort the rest
            print(f"  !! {slug} failed: {exc} — skipping (cursor preserved)", file=sys.stderr)
            continue
        totals = store.upsert_items(records)  # {repo: post-upsert total} — no re-read needed
        store.set_state(slug, now)  # persist progress per repo so a later failure can't lose it
        # `slug` is absent from totals only when there were no records to write this cycle.
        total_str = f" · {totals[slug]} total" if slug in totals else ""
        print(f"  +{len(records)} updated{total_str} → {slug.replace('/', '__')}.jsonl")

    print("done.")


if __name__ == "__main__":
    main()
