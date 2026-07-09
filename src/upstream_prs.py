"""Upstream PRs tracker (T5.14): live review-state visibility over the subset of candidates
that actually have a real, currently **open** PR against an upstream repo.

Distinct from T5.11 (:mod:`src.attempt_report`, a static per-issue "did this attempt work"
snapshot) and from T3.11 itself (:mod:`src.review_loop`, which composes/posts replies) — this
is read-only, and it re-fetches live GitHub state on every call rather than reading a frozen KB
record, since a PR's review lifecycle keeps changing long after the attempt that opened it is
done.

**Discovery, then a live check, mirroring `gate.py`'s own T3.10.6 idempotency scan:**
:func:`find_submitted_candidates` scans every `stage="gate"` run for `submitted=True` + a real
`pr_url` — the same "every prior run, not just the latest" shape
:func:`~src.gate._already_open_pr_url` already uses for the identical "has this candidate ever
opened a real PR" question, just store-wide instead of per-candidate. That KB flag alone isn't
enough to call a PR "currently open" (it could have merged or closed since) —
:func:`fetch_upstream_pr` polls GitHub for the current `state` and only a real
`state == "open"` PR is ever returned.

**Composes over T3.11/T3.13, doesn't re-implement their `gh` calls:** comment fetching and the
bot's own identity resolution reuse :func:`~src.review_loop._fetch_comments`/
:func:`~src.review_loop._current_gh_login` directly; CI-status classification and the
mergeable-string mapping reuse :func:`~src.pr_followup._ci_status_from_rollup`/
:data:`~src.pr_followup._MERGEABLE`. The one `gh pr view` call this module makes itself (not
:func:`~src.pr_followup.fetch_pr_state`) asks for a superset of fields
(`+ reviewRequests`, for "who's involved") — reusing `fetch_pr_state` as-is would mean a
*second*, redundant `gh pr view` call for the same PR just to get that one extra field.

**"Outstanding" is stricter than T3.11's own "already answered":** `run_review_loop`'s
`already_answered` check (and `pr_followup.fetch_pr_state`'s own `has_new_comments`) only
checks whether a reply was ever *composed* (`stage="review_response"`) — this module's own
`outstanding` flag, per T5.14's own DEVPLAN line, checks whether one was actually **posted**
(`stage="review_post"` with `posted=True`), since a composed-but-not-yet-posted draft still
means nothing has actually reached the maintainer.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

from . import gate, pr_followup, review_loop
from .store.base import Store


@dataclasses.dataclass(frozen=True)
class UpstreamComment:
    """One maintainer-conversation comment on a tracked PR, plus whether it's still
    ``outstanding`` (see module docstring)."""

    comment_id: int
    author: str
    body: str
    outstanding: bool


@dataclasses.dataclass(frozen=True)
class UpstreamPR:
    """One candidate's live upstream PR state, as of a single poll — see module docstring for
    why this is never persisted to the KB."""

    repo: str
    number: int
    url: str
    state: str  # always "open" -- see list_upstream_prs's own filtering
    ci_status: str  # "success" | "failure" | "pending" | "unknown"
    review_decision: str | None
    mergeable: bool | None
    last_activity_at: str
    comments: tuple[UpstreamComment, ...]
    outstanding_count: int
    requested_reviewers: tuple[str, ...]


def find_submitted_candidates(store: Store) -> list[tuple[str, int]]:
    """Every `(repo, number)` with at least one `stage="gate"` run showing `submitted=True`
    and a real `pr_url` — the store-wide sibling of
    :func:`~src.gate._already_open_pr_url`'s identical per-candidate check. Whether that PR is
    *still* open is a separate, live question (see :func:`fetch_upstream_pr`) — a KB flag set
    at submission time never un-sets itself when a PR later closes/merges.
    """
    seen: dict[tuple[str, int], None] = {}
    for run in store.list_runs(stage="gate"):
        if not (run.get("submitted") and run.get("pr_url")):
            continue
        repo, number = run.get("repo"), run.get("number")
        if isinstance(repo, str) and isinstance(number, int):
            seen[(repo, number)] = None
    return list(seen.keys())


def _fetch_pr_view(repo: str, number: int) -> dict | None:
    """``gh pr view`` with this module's own superset of fields (`pr_followup.fetch_pr_state`'s
    own fields, plus `reviewRequests`) — or `None` if the call failed (logged, not raised, the
    same fetch-failure tolerance every sibling M3/T5.x live-state stage applies)."""
    cmd = [
        "gh",
        "pr",
        "view",
        str(number),
        "--repo",
        repo,
        "--json",
        "url,state,mergeable,reviewDecision,updatedAt,statusCheckRollup,reviewRequests",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=gate._DEFAULT_GH_TIMEOUT_S
        )
    except FileNotFoundError as exc:
        print(f"upstream_prs: `gh` not found on PATH: {exc}", file=sys.stderr)
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"upstream_prs: failed to fetch PR view for {repo}#{number}: {exc}", file=sys.stderr)
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(
            f"upstream_prs: gh pr view returned non-JSON for {repo}#{number}: {exc}",
            file=sys.stderr,
        )
        return None
    return data if isinstance(data, dict) else None


def _requested_reviewers(data: dict) -> tuple[str, ...]:
    """Every requested reviewer's login/name from `gh pr view`'s own `reviewRequests` field —
    a team is requested by `name`, a user by `login`; either is a real, displayable identity."""
    reviewers = []
    for entry in data.get("reviewRequests") or ():
        if isinstance(entry, dict):
            identity = entry.get("login") or entry.get("name")
            if identity:
                reviewers.append(identity)
    return tuple(reviewers)


def fetch_upstream_pr(store: Store, repo: str, number: int) -> UpstreamPR | None:
    """`repo`#`number`'s live upstream PR state, or `None` if it isn't currently a real, open
    PR (closed/merged since submission) or any needed `gh` call failed (logged, not raised —
    fails **closed**: a comments-fetch or identity-resolution failure must never be silently
    read as "nothing outstanding," mirroring :func:`~src.pr_followup.fetch_pr_state`'s own
    identical guard).
    """
    data = _fetch_pr_view(repo, number)
    if data is None:
        return None
    state = str(data.get("state") or "").lower()
    if state != "open":
        return None

    comments_raw = review_loop._fetch_comments(repo, number)
    if comments_raw is None:
        return None
    self_login = review_loop._current_gh_login()
    if self_login is None:
        print(
            f"upstream_prs: could not resolve the bot's own gh identity for {repo}#{number} -- "
            "skipping rather than risk misreporting outstanding comments",
            file=sys.stderr,
        )
        return None

    posted_comment_ids = {
        r.get("comment_id")
        for r in store.list_runs(repo=repo, number=number, stage="review_post")
        if r.get("posted")
    }

    comments = []
    for raw in comments_raw:
        if not isinstance(raw, dict) or not isinstance(raw.get("user"), dict):
            continue
        author = raw["user"].get("login") or ""
        if author == self_login:
            continue
        comment_id = raw.get("id")
        if not isinstance(comment_id, int):
            continue
        comments.append(
            UpstreamComment(
                comment_id=comment_id,
                author=author,
                body=raw.get("body") or "",
                outstanding=comment_id not in posted_comment_ids,
            )
        )

    return UpstreamPR(
        repo=repo,
        number=number,
        url=data.get("url") or "",
        state=state,
        ci_status=pr_followup._ci_status_from_rollup(data.get("statusCheckRollup") or []),
        review_decision=data.get("reviewDecision") or None,
        mergeable=pr_followup._MERGEABLE.get(str(data.get("mergeable") or "").upper()),
        last_activity_at=data.get("updatedAt") or "",
        comments=tuple(comments),
        outstanding_count=sum(1 for c in comments if c.outstanding),
        requested_reviewers=_requested_reviewers(data),
    )


def list_upstream_prs(store: Store) -> list[UpstreamPR]:
    """Every candidate with a real, currently open upstream PR — live-checks each candidate
    :func:`find_submitted_candidates` names, silently excluding one that's since closed/merged
    or whose live fetch failed (logged by :func:`fetch_upstream_pr`, not raised)."""
    prs = []
    for repo, number in find_submitted_candidates(store):
        pr = fetch_upstream_pr(store, repo, number)
        if pr is not None:
            prs.append(pr)
    return prs
