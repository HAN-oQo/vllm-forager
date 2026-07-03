"""M0 최소 수집기 — GitHub 이슈/PR 증분 수집.

GitHub REST API의 `GET /repos/{owner}/{repo}/issues` 를 사용한다.
이 엔드포인트는 이슈 + PR을 함께 반환하며(PR은 `pull_request` 키 존재),
`since` 파라미터로 updated_at 이후만 증분 수집할 수 있다.

- 결과는 레포별 JSONL(data/{owner}__{repo}.jsonl)에 append 대신 upsert(재작성).
- 마지막 수집 시각은 data/state.json 에 레포별로 저장 → 다음 실행은 그 이후만.
- GITHUB_TOKEN 있으면 rate limit 60→5000/hr.

사용:
    python -m src.collector            # 전체 레포 증분 수집
    python -m src.collector --full     # state 무시하고 INITIAL_SINCE 부터
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
except Exception:  # python-dotenv 미설치여도 동작
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
    """rate limit에 걸리면 reset까지 대기. 대기했으면 True."""
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
        wait = max(reset - int(time.time()), 0) + 1
        print(f"  rate limit hit — waiting {wait}s", file=sys.stderr)
        time.sleep(wait)
        return True
    return False


def fetch_repo(slug: str, since: str) -> list[dict]:
    """slug(owner/repo)의 이슈+PR을 since(updated) 이후로 전부 가져온다."""
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
            print(f"  !! {slug} 404 — slug 확인 필요", file=sys.stderr)
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
    """수집 스키마 최소 정규화. 나중 RAG/분류에서 확장."""
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
    """number 기준 upsert 후 재작성. 반환: 총 레코드 수."""
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
    ap.add_argument("--full", action="store_true", help="state 무시하고 INITIAL_SINCE부터")
    args = ap.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    state = {} if args.full else _load_state()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not os.getenv("GITHUB_TOKEN"):
        print("경고: GITHUB_TOKEN 없음 — rate limit 60/hr. .env 설정 권장.", file=sys.stderr)

    for repo in config.REPOS:
        slug = repo["slug"]
        since = state.get(slug, config.INITIAL_SINCE)
        print(f"[{slug}] since {since} …")
        records = fetch_repo(slug, since)
        out_path = config.DATA_DIR / (slug.replace("/", "__") + ".jsonl")
        total = _merge_jsonl(out_path, records)
        state[slug] = now
        print(f"  +{len(records)} updated · 총 {total}건 → {out_path.name}")

    _save_state(state)
    print("done.")


if __name__ == "__main__":
    main()
