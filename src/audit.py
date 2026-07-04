"""Collection data-quality guardrail (T0.11) — reconcile the local KB against GitHub, per run.

Guardrail 1b: after collection we must know we didn't *silently* lose data. Two independent
signals, because either alone has blind spots:

1. **Count reconciliation** — the local item count vs GitHub's authoritative counts over the
   collected window, via **GraphQL** search (`issueCount` for ``is:issue`` and ``is:pr``).
   A local total that trails the remote total means we dropped items.
2. **Gap scan** — missing issue/PR numbers within the collected range. Deleted or transferred
   items leave *legitimate* holes, so a single gap proves nothing; we alert on the gap
   **ratio** (fraction of the observed number range that is missing) crossing a threshold.

Each run writes a ``data_quality`` record (count delta, gap ratio, error count, flagged) to
the KB so the live health panel (T5.8) and the policy loop can track drift over time.

Storage: records are appended to ``<data_dir>/audit/data_quality.jsonl`` — a **subdirectory**
on purpose, since :meth:`JsonlStore.query` globs ``*.jsonl`` in the data-dir root and would
otherwise ingest these records as issues/PRs. (M0.6 folds this into the Firestore KB.)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import requests

from . import config
from .collector import _headers
from .store.base import Store

GRAPHQL_URL = "https://api.github.com/graphql"

# A remote-count fetcher: (repo, since) -> {"issues", "prs", "total"}. Injected in offline
# tests; defaults to the live GraphQL implementation below.
RemoteFetcher = Callable[[str, str], dict]


def gap_ratio(numbers: list[int]) -> float:
    """Fraction of the observed issue/PR-number range that is missing (0.0 when < 2 numbers).

    ``(span - unique_count) / span`` where ``span = max - min + 1``. Deleted/transferred items
    make some gaps expected, so callers threshold this rather than treating any gap as loss.
    Non-int / duplicate numbers are ignored.
    """
    nums = sorted({n for n in numbers if isinstance(n, int)})
    if len(nums) < 2:
        return 0.0  # a single (or no) number has no range to be missing from
    span = nums[-1] - nums[0] + 1
    missing = span - len(nums)
    return missing / span


def local_counts(store: Store, repo: str) -> dict:
    """Local KB counts for `repo`: issue/PR totals plus the collected numbers (for the gap scan)."""
    items = store.query(repo=repo)
    issues = sum(1 for it in items if it.get("type") == "issue")
    prs = sum(1 for it in items if it.get("type") == "pr")
    numbers = [it["number"] for it in items if isinstance(it.get("number"), int)]
    return {"issues": issues, "prs": prs, "total": issues + prs, "numbers": numbers}


def remote_counts(repo: str, since: str) -> dict:
    """GitHub GraphQL counts of issues + PRs for `repo` updated at/after `since` (a date/ISO ts).

    Uses the search connection's ``issueCount`` (``type: ISSUE`` covers both issues and PRs;
    the ``is:issue`` / ``is:pr`` qualifiers split them). Live network call — offline tests inject
    a fake via the `remote_fetcher` parameter of :func:`audit_repo`. Raises on HTTP or GraphQL
    errors so the caller records it as an error rather than a silent zero.
    """
    since_date = since[:10]  # search's updated: qualifier takes a date (or full ISO) bound
    query = (
        "query($qi: String!, $qp: String!) {"
        "  issues: search(query: $qi, type: ISSUE) { issueCount }"
        "  prs: search(query: $qp, type: ISSUE) { issueCount }"
        "}"
    )
    variables = {
        "qi": f"repo:{repo} is:issue updated:>={since_date}",
        "qp": f"repo:{repo} is:pr updated:>={since_date}",
    }
    resp = requests.post(
        GRAPHQL_URL,
        headers=_headers(),
        json={"query": query, "variables": variables},
        timeout=config.REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(f"GraphQL errors for {repo}: {payload['errors']}")
    data = payload["data"]
    issues = int(data["issues"]["issueCount"])
    prs = int(data["prs"]["issueCount"])
    return {"issues": issues, "prs": prs, "total": issues + prs}


def reconcile(
    repo: str,
    local: dict,
    remote: dict,
    *,
    errors: int = 0,
    gap_threshold: float | None = None,
) -> dict:
    """Build the ``data_quality`` record comparing `local` vs `remote` counts (pure, no I/O).

    ``count_delta = local.total - remote.total`` (negative ⇒ we hold fewer than GitHub reports
    ⇒ probable loss). The record is **flagged** when any signal is bad: a negative delta, a gap
    ratio over `gap_threshold` (defaults to :data:`config.DATA_QUALITY_GAP_RATIO_THRESHOLD`), or
    a non-zero `errors` count from collection. No timestamp here so the function stays
    deterministic; :func:`write_record` stamps ``checked_at`` when persisting.
    """
    threshold = config.DATA_QUALITY_GAP_RATIO_THRESHOLD if gap_threshold is None else gap_threshold
    ratio = gap_ratio(local.get("numbers", []))
    delta = local["total"] - remote["total"]
    flagged = delta < 0 or ratio > threshold or errors > 0
    return {
        "repo": repo,
        "local_total": local["total"],
        "remote_total": remote["total"],
        "count_delta": delta,
        "gap_ratio": round(ratio, 4),
        "error_count": errors,
        "flagged": flagged,
    }


def _data_quality_path(data_dir: Path) -> Path:
    """The audit sink — a subdir file so :meth:`JsonlStore.query`'s ``*.jsonl`` glob skips it."""
    return Path(data_dir) / "audit" / "data_quality.jsonl"


def write_record(data_dir: Path, record: dict, *, checked_at: str) -> Path:
    """Append `record` (stamped with `checked_at`) to the data-quality JSONL. Returns the path."""
    path = _data_quality_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {**record, "checked_at": checked_at}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stamped, ensure_ascii=False) + "\n")
    return path


def audit_repo(
    store: Store,
    repo: str,
    since: str,
    *,
    remote_fetcher: RemoteFetcher = remote_counts,
    errors: int = 0,
    checked_at: str,
    gap_threshold: float | None = None,
) -> dict:
    """Reconcile `repo` (local vs remote), persist the ``data_quality`` record, and return it.

    `remote_fetcher` is injectable so offline tests avoid the network. `since` bounds the remote
    count to the collected window; `errors` is the collection error count for this repo (folded
    into the flag). Requires `checked_at` (an ISO timestamp) from the caller so this stays
    deterministic under test.
    """
    local = local_counts(store, repo)
    remote = remote_fetcher(repo, since)
    record = reconcile(repo, local, remote, errors=errors, gap_threshold=gap_threshold)
    data_dir = getattr(store, "data_dir", config.DATA_DIR)
    write_record(data_dir, record, checked_at=checked_at)
    return record
