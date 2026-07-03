"""M0 minimal collector — incremental collection of GitHub issues/PRs.

Uses the GitHub REST API endpoint `GET /repos/{owner}/{repo}/issues`.
This endpoint returns issues and PRs together (PRs have a `pull_request` key), and the
`since` parameter allows incremental collection of items updated after a given time.

- Results are upserted (rewritten), not appended, into a per-repo JSONL (data/{owner}__{repo}.jsonl).
- The last collection time is stored per repo in data/state.json → the next run only fetches items after it.
- With GITHUB_TOKEN set, the rate limit goes from 60 to 5000/hr.

Usage:
    python -m src.collector            # incremental collection for all repos
    python -m src.collector --full     # ignore state and start from INITIAL_SINCE
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

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


def fetch_repo(slug: str, since: str) -> list[dict]:
    """Fetch all issues + PRs for slug (owner/repo) updated after `since`."""
    owner, repo = slug.split("/", 1)
    url = f"{API}/repos/{owner}/{repo}/issues"
    params = {
        "since": since,
        "state": "all",
        "per_page": config.PER_PAGE,
        "sort": "updated",
        "direction": "asc",
        "page": 1,
    }
    out: list[dict] = []
    while True:
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        if _sleep_for_rate_limit(resp):
            continue
        if resp.status_code == 404:
            print(f"  !! {slug} 404 — check the slug", file=sys.stderr)
            break
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for it in batch:
            out.append(_normalize(it, slug))
        if len(batch) < config.PER_PAGE:
            break
        params["page"] += 1
    return out


def _normalize(it: dict, slug: str) -> dict:
    """Minimal normalization of the collection schema. Extended later in RAG/classification."""
    return {
        "repo": slug,
        "number": it.get("number"),
        "type": "pr" if "pull_request" in it else "issue",
        "title": it.get("title"),
        "state": it.get("state"),
        "labels": [l.get("name") for l in it.get("labels", [])],
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
    ap.add_argument("--full", action="store_true", help="ignore state and start from INITIAL_SINCE")
    args = ap.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    state = {} if args.full else _load_state()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not os.getenv("GITHUB_TOKEN"):
        print("warning: no GITHUB_TOKEN — rate limit 60/hr. Setting up .env is recommended.", file=sys.stderr)

    for repo in config.REPOS:
        slug = repo["slug"]
        since = state.get(slug, config.INITIAL_SINCE)
        print(f"[{slug}] since {since} …")
        records = fetch_repo(slug, since)
        out_path = config.DATA_DIR / (slug.replace("/", "__") + ".jsonl")
        total = _merge_jsonl(out_path, records)
        state[slug] = now
        print(f"  +{len(records)} updated · {total} total → {out_path.name}")

    _save_state(state)
    print("done.")


if __name__ == "__main__":
    main()
