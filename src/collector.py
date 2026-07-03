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
import json
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


def _load_state() -> dict:
    if config.STATE_PATH.exists():
        return json.loads(config.STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.STATE_PATH.write_text(json.dumps(state, indent=2))


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
    """Upsert by `number` then rewrite. Returns the total record count."""
    existing: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["number"]] = r
    for r in records:
        existing[r["number"]] = r
    ordered = sorted(existing.values(), key=lambda r: r.get("updated_at") or "")
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ordered) + "\n")
    return len(ordered)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full", action="store_true", help="ignore state; re-fetch the lookback window"
    )
    args = ap.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    state = {} if args.full else _load_state()
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
        since = state.get(slug, default_since)
        print(f"[{slug}] since {since} …")
        try:
            records = fetch_repo(slug, since)
        except Exception as exc:  # isolate: one repo's failure must not abort the rest
            print(f"  !! {slug} failed: {exc} — skipping (cursor preserved)", file=sys.stderr)
            continue
        out_path = config.DATA_DIR / (slug.replace("/", "__") + ".jsonl")
        total = _merge_jsonl(out_path, records)
        state[slug] = now
        _save_state(state)  # persist progress per repo so a later failure can't lose it
        print(f"  +{len(records)} updated · {total} total → {out_path.name}")

    _save_state(state)
    print("done.")


if __name__ == "__main__":
    main()
