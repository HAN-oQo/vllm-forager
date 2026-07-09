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

**Composes over T3.11/T3.13, doesn't re-implement their `gh` calls:** comment fetching reuses
:func:`~src.review_loop._fetch_comments` directly; CI-status classification and the
mergeable-string mapping reuse :func:`~src.pr_followup._ci_status_from_rollup`/
:data:`~src.pr_followup._MERGEABLE`; the `gh pr view` call itself reuses
:func:`~src.pr_followup._gh_pr_view` with a superset of fields (`+ reviewRequests`, for "who's
involved") — an earlier version of this module hand-copied `pr_followup.fetch_pr_state`'s own
fetch+error-handling block into a sibling function just to vary that one field list; a
code-review finding extracted the shared mechanics into `_gh_pr_view` instead, in
`pr_followup.py`, so a fix to that contract only has to land once.

**Batched, not per-candidate (a code-review finding):** :func:`list_upstream_prs` resolves the
bot's own `gh` identity and scans `store.list_runs(stage="review_post")` **once** for every
tracked candidate, not once per candidate inside a loop — the identical "N separate reads/
calls" anti-pattern this codebase has already found and fixed for `dashboard.health`,
`dashboard.cost`, `src.selection.decisions_by_key`, and `src.attempt_report.list_worked`, here
reintroduced across live `gh` calls (a `gh api user` subprocess spawn is real, non-trivial
work) as well as a KB scan. :func:`fetch_upstream_pr` takes both as already-resolved
parameters rather than fetching either itself.

**"Outstanding" is stricter than T3.11's own "already answered":** `run_review_loop`'s
`already_answered` check (and `pr_followup.fetch_pr_state`'s own `has_new_comments`) only
checks whether a reply was ever *composed* (`stage="review_response"`) — this module's own
`outstanding` flag, per T5.14's own DEVPLAN line, checks whether one was actually **posted**
(`stage="review_post"` with `posted=True`), since a composed-but-not-yet-posted draft still
means nothing has actually reached the maintainer.
"""

from __future__ import annotations

import dataclasses
import sys
from collections import defaultdict

from . import pr_followup, review_loop
from .store.base import Store

_PR_FIELDS = "url,state,mergeable,reviewDecision,updatedAt,statusCheckRollup,reviewRequests"


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
    why this is never persisted to the KB.

    `state` is always ``"open"`` (:func:`fetch_upstream_pr` returns `None` for anything else) —
    kept as a real field rather than dropped, since a future extension (e.g. also surfacing a
    just-closed PR for one final look) would need it back; that extension isn't done here.
    `outstanding_count` is a computed property, not a stored field, so it can never drift out
    of sync with `comments`.
    """

    repo: str
    number: int
    url: str
    state: str
    ci_status: str  # "success" | "failure" | "pending" | "unknown"
    review_decision: str | None
    mergeable: bool | None
    last_activity_at: str
    comments: tuple[UpstreamComment, ...]
    requested_reviewers: tuple[str, ...]

    @property
    def outstanding_count(self) -> int:
        return sum(1 for c in self.comments if c.outstanding)


def find_submitted_candidates(store: Store) -> list[tuple[str, int]]:
    """Every `(repo, number)` with at least one `stage="gate"` run showing `submitted=True`
    and a real `pr_url` — the store-wide sibling of
    :func:`~src.gate._already_open_pr_url`'s identical per-candidate check. Whether that PR is
    *still* open is a separate, live question (see :func:`fetch_upstream_pr`) — a KB flag set
    at submission time never un-sets itself when a PR later closes/merges.
    """
    seen: set[tuple[str, int]] = set()
    for run in store.list_runs(stage="gate"):
        if not (run.get("submitted") and run.get("pr_url")):
            continue
        repo, number = run.get("repo"), run.get("number")
        if isinstance(repo, str) and isinstance(number, int):
            seen.add((repo, number))
    return list(seen)


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


def fetch_upstream_pr(
    repo: str,
    number: int,
    *,
    self_login: str,
    posted_comment_ids: frozenset[int],
) -> UpstreamPR | None:
    """`repo`#`number`'s live upstream PR state, or `None` if it isn't currently a real, open
    PR (closed/merged since submission) or any needed `gh` call failed (logged, not raised —
    fails **closed**: a comments-fetch failure must never be silently read as "nothing
    outstanding," mirroring :func:`~src.pr_followup.fetch_pr_state`'s own identical guard).

    `self_login`/`posted_comment_ids` are resolved once by :func:`list_upstream_prs` for every
    tracked candidate, not fetched here — see module docstring's own "batched, not
    per-candidate" note.
    """
    data = pr_followup._gh_pr_view(repo, number, _PR_FIELDS)
    if data is None:
        return None
    state = str(data.get("state") or "").lower()
    if state != "open":
        return None

    comments_raw = review_loop._fetch_comments(repo, number)
    if comments_raw is None:
        return None

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
        requested_reviewers=_requested_reviewers(data),
    )


def _posted_comment_ids_by_candidate(store: Store) -> dict[tuple[str, int], frozenset[int]]:
    """Every candidate's already-*posted* comment IDs, in one `store.list_runs(stage=
    "review_post")` fetch — see module docstring's "batched, not per-candidate" note."""
    by_candidate: dict[tuple[str, int], set[int]] = defaultdict(set)
    for run in store.list_runs(stage="review_post"):
        if not run.get("posted"):
            continue
        repo, number, comment_id = run.get("repo"), run.get("number"), run.get("comment_id")
        if isinstance(repo, str) and isinstance(number, int) and isinstance(comment_id, int):
            by_candidate[(repo, number)].add(comment_id)
    return {key: frozenset(ids) for key, ids in by_candidate.items()}


def list_upstream_prs(store: Store) -> list[UpstreamPR]:
    """Every candidate with a real, currently open upstream PR — live-checks each candidate
    :func:`find_submitted_candidates` names, silently excluding one that's since closed/merged
    or whose live fetch failed (logged by :func:`fetch_upstream_pr`, not raised).

    Resolves the bot's own `gh` identity once, up front, for every candidate this poll will
    check — if that fails, the whole poll fails closed (`[]`, logged) rather than silently
    treating every candidate's comments as unattributable one at a time.
    """
    candidates = find_submitted_candidates(store)
    if not candidates:
        return []

    self_login = review_loop._current_gh_login()
    if self_login is None:
        print(
            "upstream_prs: could not resolve the bot's own gh identity -- skipping this poll "
            "rather than risk misreporting outstanding comments",
            file=sys.stderr,
        )
        return []

    posted_by_candidate = _posted_comment_ids_by_candidate(store)

    prs = []
    for repo, number in candidates:
        pr = fetch_upstream_pr(
            repo,
            number,
            self_login=self_login,
            posted_comment_ids=posted_by_candidate.get((repo, number), frozenset()),
        )
        if pr is not None:
            prs.append(pr)
    return prs
