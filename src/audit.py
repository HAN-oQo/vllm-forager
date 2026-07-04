"""Collection data-quality guardrail (T0.11) — reconcile the local KB against GitHub, per run.

Guardrail 1b: after collection we must know we didn't *silently* lose data. Two independent
signals, because either alone has blind spots:

1. **Count reconciliation** — the local item count vs GitHub's authoritative count over the
   collected window, via **GraphQL** search ``issueCount`` (``type: ISSUE`` counts issues and
   PRs together). **Both sides are windowed by** ``since`` so they're comparable: a local total
   that *trails* remote means this run dropped items; a *positive* delta is deletions/transfers
   (expected) and is left to the gap scan.
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

# A remote-count fetcher: (repo, since) -> {"total": int}. Injected in offline tests; defaults
# to the live GraphQL implementation below.
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


def local_counts(store: Store, repo: str, since: str) -> dict:
    """Local KB counts for `repo` **within the collected window** (updated at/after `since`).

    Windowed to match :func:`remote_counts` (both bounded by the same ``since`` date), so their
    totals are comparable. Returns ``{"total", "numbers"}`` derived from the *same* windowed item
    set — ``total`` is the item count, ``numbers`` the int issue/PR numbers (for the gap scan).
    """
    since_date = since[:10]
    items = [it for it in store.query(repo=repo) if (it.get("updated_at") or "")[:10] >= since_date]
    numbers = [it["number"] for it in items if isinstance(it.get("number"), int)]
    return {"total": len(items), "numbers": numbers}


def remote_counts(repo: str, since: str) -> dict:
    """GitHub GraphQL count of issues + PRs for `repo` updated at/after `since` (a date/ISO ts).

    A single search (``type: ISSUE`` counts issues and PRs together) over the window; its
    ``issueCount`` is the authoritative total to reconcile against. Live network call — offline
    tests inject a fake via the `remote_fetcher` parameter of :func:`audit_repo`. Raises on HTTP
    errors, GraphQL errors, *or* a partial/None payload, so the caller records it as an error
    rather than a silent zero.
    """
    since_date = since[:10]  # search's updated: qualifier takes a date (or full ISO) bound
    query = "query($q: String!) { search(query: $q, type: ISSUE) { issueCount } }"
    variables = {"q": f"repo:{repo} updated:>={since_date}"}
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
    search = (payload.get("data") or {}).get("search")
    if not search or search.get("issueCount") is None:
        raise RuntimeError(f"unexpected GraphQL payload for {repo}: {payload}")
    return {"total": int(search["issueCount"])}


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
    # With both counts windowed, a NEGATIVE delta means we hold fewer than GitHub reports for
    # the window ⇒ probable silent loss. A positive delta is deletions/transfers (expected) and
    # is covered by the gap scan, so it is not itself a flag.
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
    data_dir: Path | None = None,
    gap_threshold: float | None = None,
) -> dict:
    """Reconcile `repo` (local vs remote), persist the ``data_quality`` record, and return it.

    `remote_fetcher` is injectable so offline tests avoid the network. `since` bounds *both* the
    remote and local counts to the collected window (comparable totals); `errors` is the
    collection error count for this repo (folded into the flag). Requires `checked_at` (an ISO
    timestamp) so this stays deterministic under test.

    Persistence target: `data_dir` if given, else the store's ``data_dir`` (JsonlStore). A store
    without one (a future non-JSONL backend) raises loudly rather than silently misrouting the
    records to disk — until M0.6 gives the ``data_quality`` record a home on the Store interface.
    """
    local = local_counts(store, repo, since)
    remote = remote_fetcher(repo, since)
    record = reconcile(repo, local, remote, errors=errors, gap_threshold=gap_threshold)
    resolved_dir = data_dir if data_dir is not None else getattr(store, "data_dir", None)
    if resolved_dir is None:
        raise TypeError(
            "audit_repo needs a JsonlStore or an explicit data_dir; "
            f"{type(store).__name__} exposes no data_dir"
        )
    write_record(resolved_dir, record, checked_at=checked_at)
    return record
