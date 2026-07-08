"""Collection data-quality guardrail (T0.11–T0.12) — reconcile the local KB against GitHub.

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
:func:`record_stall` (T0.12) writes the same kind of record for the collector's cursor-stall
case (see ``src.collector.PAGE_CAP``) — a >1000-item single-timestamp cluster REST pagination
can't get past — falling back to GraphQL for a count even though the items aren't recoverable.

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
        # A common discriminant with :func:`record_stall`'s "cursor_stall" reason — the two
        # variants share this sink (data_quality.jsonl) but not every field, so a consumer
        # (T5.8's health panel) must branch on `reason` rather than assume a uniform schema.
        "reason": "reconciliation",
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


def list_records(store: Store, *, data_dir: Path | None = None) -> list[dict]:
    """Every recorded ``data_quality`` guardrail check (reconciliation + cursor-stall records,
    each already carrying its own ``flagged`` boolean — computed once, at write time, by
    :func:`reconcile`/:func:`record_stall`, not re-derived here), oldest first.

    `data_dir` resolves the same way :func:`_resolve_data_dir` does for a write, but degrades
    to ``[]`` instead of raising when it can't be resolved (no explicit `data_dir` and `store`
    exposes none) or the sink doesn't exist yet (a fresh KB that has never run an audit check) —
    a read endpoint (T5.7's dashboard guardrail panel) shouldn't 500 over a config gap or a
    not-yet-populated sink the way a write correctly still refuses to guess past.
    """
    resolved = data_dir if data_dir is not None else getattr(store, "data_dir", None)
    if resolved is None:
        return []
    path = _data_quality_path(resolved)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resolve_data_dir(store: Store, data_dir: Path | None, *, caller: str) -> Path:
    """`data_dir` if given, else the store's ``data_dir`` (JsonlStore) — else raise loudly.

    A store without either (a future non-JSONL backend) must not silently misroute records to
    `config.DATA_DIR` — until M0.6 gives ``data_quality`` a home on the Store interface itself.
    """
    resolved = data_dir if data_dir is not None else getattr(store, "data_dir", None)
    if resolved is None:
        raise TypeError(
            f"{caller} needs a JsonlStore or an explicit data_dir; "
            f"{type(store).__name__} exposes no data_dir"
        )
    return resolved


def _persist(
    store: Store, record: dict, *, checked_at: str, data_dir: Path | None, caller: str
) -> dict:
    """Resolve the sink and append `record` — the shared tail of `audit_repo`/`record_stall`."""
    resolved_dir = _resolve_data_dir(store, data_dir, caller=caller)
    write_record(resolved_dir, record, checked_at=checked_at)
    return record


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
    timestamp) so this stays deterministic under test. See :func:`_resolve_data_dir` for the
    `data_dir` persistence target.
    """
    local = local_counts(store, repo, since)
    remote = remote_fetcher(repo, since)
    record = reconcile(repo, local, remote, errors=errors, gap_threshold=gap_threshold)
    return _persist(store, record, checked_at=checked_at, data_dir=data_dir, caller="audit_repo")


def record_stall(
    store: Store,
    repo: str,
    window_since: str,
    *,
    remote_fetcher: RemoteFetcher = remote_counts,
    checked_at: str,
    data_dir: Path | None = None,
) -> dict:
    """Record a collector cursor-stall (T0.12) as a flagged ``data_quality`` record.

    A stall means too many items (> ``PAGE_CAP`` * ``PER_PAGE``, collector.py) share one
    ``updated_at`` timestamp, so REST offset-pagination can't advance past it and the collector
    gives up on that window. This falls back to GraphQL (`remote_fetcher`, the same mechanism as
    :func:`audit_repo`) to learn the window's true item **count** — the items themselves aren't
    recoverable via this path, but the *size* of what was missed becomes visible in the
    data_quality history instead of only a stderr line. Always ``flagged`` (a stall is itself the
    signal). See :func:`_resolve_data_dir` for the `data_dir` persistence target.
    """
    remote = remote_fetcher(repo, window_since)
    record = {
        "repo": repo,
        "reason": "cursor_stall",
        "window_since": window_since,
        "remote_total": remote["total"],
        "flagged": True,
    }
    return _persist(store, record, checked_at=checked_at, data_dir=data_dir, caller="record_stall")
