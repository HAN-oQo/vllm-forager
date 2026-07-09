"""PR follow-up steward (T3.13): tracks every submitted upstream PR through to merge/close and
picks the single next action per PR.

Opening a PR isn't the deliverable — a merge is. PRs die from unanswered reviews, red CI,
merge conflicts, or silence; this is the piece that keeps a submitted candidate moving toward
merge, across its whole lifecycle, the way `review_loop.py` (T3.11) only reacts to one comment
at a time.

**Priority, most-blocking first:** a merge conflict (`mergeable is False`) blocks a merge
regardless of anything else, so it's checked first; CI failure is next (nothing else matters
until the code itself is green); then an unanswered comment (something concrete to respond to
right now); then an outright approval with CI green (nothing left to do but tell the human).
**`review_decision == "CHANGES_REQUESTED"` alone, with no new comment, is deliberately NOT its
own branch** -- GitHub only clears that decision on an explicit re-review, so once every comment
has been answered the ball is in the *reviewer's* court, not something this steward can act on
again; forcing it back into `review_response` forever would mean `run_review_loop` composing
nothing every single poll with no path to ever escalate to a stale-nudge. Staleness is the
fallback for everything that isn't otherwise actionable (including a stuck `CHANGES_REQUESTED`
with nothing new to say). A closed/merged PR needs nothing.

**Nothing is posted/pushed without an explicit human-confirm**, mirroring `gate.py`/
`review_loop.py`'s own convention: :func:`next_action` is a pure, side-effect-free decision
(unit-testable against plain :class:`PRState` values with no store or network involved), and
only :func:`apply_action` performs a state's real-world effect — and only the two GitHub-adjacent
effects that actually reach outward (a `notify_ready`/`nudge_stale` phone push) are gated on
`approve=True`. Composing a review reply reuses `review_loop.run_review_loop` directly, which is
already itself compose-only/never-post, so it runs unconditionally — no separate approve gate
needed for a step that can't post on its own.

**`ci_fix`/`rebase` are picked, not automated.** Driving `engineer.py` to prepare an MI250-
verified CI fix, or running a real `git rebase`, from an unattended poll loop is a materially
bigger and riskier capability than this todo's own worked example needs — mirroring T3.7/T3.9/
T3.10's own "standalone, not wired into an orchestrator yet" notes, the steward reports which
candidates need one and leaves running it to a human (or a future M4 orchestrator that can wire
in the already-existing, already-MI250-gated `engineer.py` pipeline).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config, gate, review_loop
from .store import resolve_store
from .store.base import Store

# The same GitHub-API timestamp format collector.py/trends.py/forecaster.py already use --
# `updatedAt` is machine-written by GitHub, never LLM-generated, so strict parsing is right.
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_NOTIFY_SCRIPT = config.ROOT / "scripts" / "notify.sh"

STALE_AFTER_DAYS = 7

ACTION_REBASE = "rebase"
ACTION_CI_FIX = "ci_fix"
ACTION_REVIEW_RESPONSE = "review_response"
ACTION_NOTIFY_READY = "notify_ready"
ACTION_NUDGE_STALE = "nudge_stale"
ACTION_NONE = "none"

_MERGEABLE = {"MERGEABLE": True, "CONFLICTING": False}
# `CheckRun` (GitHub Actions/Checks API) failure conclusions.
_CI_FAILURE_CONCLUSIONS = {
    "FAILURE",
    "CANCELLED",
    "TIMED_OUT",
    "ACTION_REQUIRED",
    "STARTUP_FAILURE",
}
# `StatusContext` (legacy Commit Status API -- external CI like Buildkite/CircleCI/Jenkins)
# failure states. `statusCheckRollup` is a union of both shapes; a rollup entry carries either
# conclusion/status (CheckRun) or state (StatusContext), never both -- a code-review finding on
# this module pointed out that reading only conclusion/status silently misreads a failed legacy
# status as "pending" forever, since it has neither key.
_CI_FAILURE_STATES = {"FAILURE", "ERROR"}


@dataclasses.dataclass(frozen=True)
class PRState:
    """One open, submitted upstream PR's lifecycle state as of a single poll. `state` is
    lower-case (`"open"`/`"closed"`/`"merged"`, matching `gh pr view`'s own three values
    lower-cased). `mergeable` is `False` only for a real conflict, `None` when GitHub hasn't
    computed it yet (never treated the same as a confirmed conflict)."""

    repo: str
    number: int
    url: str
    state: str
    ci_status: str  # "success" | "failure" | "pending" | "unknown"
    review_decision: str | None  # "APPROVED" | "CHANGES_REQUESTED" | "REVIEW_REQUIRED" | None
    has_new_comments: bool
    mergeable: bool | None
    last_activity_at: str


@dataclasses.dataclass(frozen=True)
class FollowupAction:
    """The single next action the steward picked for one PR, plus why."""

    repo: str
    number: int
    action: str
    reason: str


def _parse_ts(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw, _TS_FORMAT).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def next_action(pr: PRState, *, now: datetime | None = None) -> FollowupAction:
    """The single next action for `pr` — see module docstring for the priority order."""
    if pr.state != "open":
        return FollowupAction(pr.repo, pr.number, ACTION_NONE, f"PR is {pr.state}, nothing to do")

    if pr.mergeable is False:
        return FollowupAction(
            pr.repo, pr.number, ACTION_REBASE, "merge conflict with the base branch"
        )

    if pr.ci_status == "failure":
        return FollowupAction(pr.repo, pr.number, ACTION_CI_FIX, "CI is red")

    if pr.has_new_comments:
        return FollowupAction(
            pr.repo, pr.number, ACTION_REVIEW_RESPONSE, "reviewer feedback needs a response"
        )

    if pr.review_decision == "APPROVED" and pr.ci_status == "success":
        return FollowupAction(
            pr.repo, pr.number, ACTION_NOTIFY_READY, "approved and ready to merge"
        )

    when = now or datetime.now(timezone.utc)
    last_activity = _parse_ts(pr.last_activity_at)
    if last_activity is not None and (when - last_activity).days >= STALE_AFTER_DAYS:
        return FollowupAction(
            pr.repo,
            pr.number,
            ACTION_NUDGE_STALE,
            f"no activity in >= {STALE_AFTER_DAYS} days",
        )

    return FollowupAction(pr.repo, pr.number, ACTION_NONE, "no action needed yet")


def _notify(
    message: str, *, title: str = "vllm-forager:pr-followup", url: str | None = None
) -> bool:
    """Best-effort phone push via `scripts/notify.sh` -- `True` only if the script actually ran
    and exited zero, `False` otherwise (logged, never raised: a failed notification must never
    fail the steward's own decision-making pass, but its caller still needs to know it failed
    rather than silently reporting success)."""
    cmd = [str(_NOTIFY_SCRIPT), message, title]
    if url:
        cmd.append(url)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=gate._DEFAULT_GH_TIMEOUT_S
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"pr_followup: notify failed: {exc}", file=sys.stderr)
        return False
    if result.returncode != 0:
        print(
            f"pr_followup: notify.sh exited {result.returncode}: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    return True


def apply_action(
    store: Store,
    action: FollowupAction,
    pr: PRState,
    *,
    approve: bool = False,
    drafts_dir: Path | None = None,
) -> str:
    """Perform `action`'s real-world effect and return a human-readable summary of what
    happened. `approve` must be the literal `True` (matching `gate.py`'s own convention) to
    actually send a `notify_ready`/`nudge_stale` phone push -- every other branch either never
    touches the network (`review_response` delegates to the already compose-only
    `review_loop.run_review_loop`) or is report-only (`ci_fix`/`rebase`, see module docstring)."""
    if action.action == ACTION_REVIEW_RESPONSE:
        responses = review_loop.run_review_loop(
            store, action.repo, action.number, drafts_dir=drafts_dir
        )
        if not responses:
            return (
                f"{action.repo}#{action.number}: no new maintainer comment to compose a reply for"
            )
        plural = "y" if len(responses) == 1 else "ies"
        return (
            f"{action.repo}#{action.number}: drafted {len(responses)} repl{plural} -- review "
            "and post with `python -m src.review_loop post` (never auto-posted)"
        )

    if action.action == ACTION_NOTIFY_READY:
        if not approve:
            return (
                f"{action.repo}#{action.number}: approved and ready to merge "
                "(pass --approve to notify)"
            )
        sent = _notify(
            f"PR {action.repo}#{action.number} is approved and ready to merge.", url=pr.url
        )
        if sent:
            return f"{action.repo}#{action.number}: notified -- approved and ready to merge"
        return (
            f"{action.repo}#{action.number}: notify failed (see stderr) -- "
            "approved and ready to merge"
        )

    if action.action == ACTION_NUDGE_STALE:
        if not approve:
            return f"{action.repo}#{action.number}: stale (pass --approve to notify)"
        sent = _notify(
            f"PR {action.repo}#{action.number} has been silent for >= {STALE_AFTER_DAYS} days.",
            url=pr.url,
        )
        if sent:
            return f"{action.repo}#{action.number}: notified -- stale, needs a nudge"
        return f"{action.repo}#{action.number}: notify failed (see stderr) -- stale, needs a nudge"

    if action.action == ACTION_CI_FIX:
        return (
            f"{action.repo}#{action.number}: CI is red -- needs the engineer to prepare an "
            "MI250-verified fix (not automated here, see module docstring)"
        )

    if action.action == ACTION_REBASE:
        return (
            f"{action.repo}#{action.number}: merge conflict -- needs a manual rebase (not "
            "automated here, see module docstring)"
        )

    return f"{action.repo}#{action.number}: no action needed"


def _ci_status_from_rollup(rollup: list) -> str:
    """`"failure"` if any check concluded badly, `"pending"` if any is still running,
    `"success"` if every check completed cleanly, else `"unknown"` (no checks reported yet).
    Handles both `statusCheckRollup` node shapes (see `_CI_FAILURE_STATES`'s comment)."""
    if not rollup:
        return "unknown"
    saw_pending = False
    saw_completed = False
    for check in rollup:
        if not isinstance(check, dict):
            continue
        conclusion = str(check.get("conclusion") or "").upper()
        status = str(check.get("status") or "").upper()
        state = str(check.get("state") or "").upper()
        if conclusion in _CI_FAILURE_CONCLUSIONS or state in _CI_FAILURE_STATES:
            return "failure"
        if state:
            if state == "SUCCESS":
                saw_completed = True
            else:
                saw_pending = True
        elif not conclusion or (status and status != "COMPLETED"):
            saw_pending = True
        else:
            saw_completed = True
    if saw_pending:
        return "pending"
    return "success" if saw_completed else "unknown"


_PR_STATE_FIELDS = "url,state,mergeable,reviewDecision,updatedAt,statusCheckRollup"


def _gh_pr_view(repo: str, number: int, fields: str) -> dict | None:
    """One ``gh pr view --json <fields>`` call, or `None` if it failed (logged, not raised).

    Extracted (a code-review finding on T5.14) so the fetch+error-handling ladder itself --
    the subprocess invocation, the three failure modes, the JSON-decode guard -- lives in
    exactly one place. :func:`fetch_pr_state` (this module) and
    :func:`~src.upstream_prs.fetch_upstream_pr` both need a live ``gh pr view`` for the same
    PR but ask for different field sets (T5.14's own tracker also needs `reviewRequests`, for
    "who's involved") -- before this extraction, `upstream_prs.py` had hand-copied this entire
    block into its own sibling function just to vary the `--json` argument, so a future fix to
    this contract (a new `gh` failure mode, a retry policy) had to be applied twice by hand."""
    cmd = ["gh", "pr", "view", str(number), "--repo", repo, "--json", fields]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=gate._DEFAULT_GH_TIMEOUT_S
        )
    except FileNotFoundError as exc:
        print(f"pr_followup: `gh` not found on PATH: {exc}", file=sys.stderr)
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"pr_followup: failed to fetch PR state for {repo}#{number}: {exc}", file=sys.stderr)
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(
            f"pr_followup: gh pr view returned non-JSON for {repo}#{number}: {exc}", file=sys.stderr
        )
        return None
    if not isinstance(data, dict):
        print(
            f"pr_followup: gh pr view returned unexpected JSON for {repo}#{number}", file=sys.stderr
        )
        return None
    return data


def fetch_pr_state(store: Store, repo: str, number: int) -> PRState | None:
    """Poll GitHub for `repo`#`number`'s current lifecycle state, or `None` if any `gh` call
    needed to answer it failed (logged, not raised -- the same fetch-failure tolerance every
    sibling M3 stage applies, but fails **closed**: a comments-fetch or identity-resolution
    failure must never be silently read as "no new comments," since that could hide reviewer
    feedback in the same way `review_loop.run_review_loop` explicitly refuses to guess who
    posted a comment when its own identity check fails).

    `has_new_comments` reuses `review_loop`'s own already-answered bookkeeping (the same
    `stage="review_response"` KB records `run_review_loop` itself checks) and the same
    `get_item_or_skip`-style KB-item gate `run_review_loop` applies, so the steward's decision to
    run a review-response pass always agrees with whether that pass would actually find
    something new to compose."""
    data = _gh_pr_view(repo, number, _PR_STATE_FIELDS)
    if data is None:
        return None

    state = str(data.get("state") or "").lower()

    has_new_comments = False
    # `next_action` discards has_new_comments for a non-open PR anyway -- skip the
    # comments/identity/store round trips a closed/merged PR would otherwise still pay for.
    # `run_review_loop` (the thing ACTION_REVIEW_RESPONSE actually delegates to) is also a no-op
    # without a KB item for this candidate -- match that gate here too, so a picked action always
    # agrees with what applying it would do (a code-review finding on this module).
    if state == "open" and store.get_item(repo, number) is not None:
        comments = review_loop._fetch_comments(repo, number)
        if comments is None:
            print(f"pr_followup: failed to fetch comments for {repo}#{number}", file=sys.stderr)
            return None
        if comments:
            self_login = review_loop._current_gh_login()
            if self_login is None:
                print(
                    f"pr_followup: could not resolve the bot's own gh identity for "
                    f"{repo}#{number} -- skipping this poll rather than risk treating its own "
                    "comments as new maintainer feedback",
                    file=sys.stderr,
                )
                return None
            already_answered = {
                r.get("comment_id")
                for r in store.list_runs(repo=repo, number=number, stage="review_response")
            }
            has_new_comments = any(
                isinstance(comment, dict)
                and isinstance(comment.get("user"), dict)
                and comment.get("id") not in already_answered
                and comment["user"].get("login") != self_login
                for comment in comments
            )

    return PRState(
        repo=repo,
        number=number,
        url=data.get("url") or "",
        state=state,
        ci_status=_ci_status_from_rollup(data.get("statusCheckRollup") or []),
        review_decision=(data.get("reviewDecision") or None),
        has_new_comments=has_new_comments,
        mergeable=_MERGEABLE.get(str(data.get("mergeable") or "").upper()),
        last_activity_at=data.get("updatedAt") or "",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: poll one submitted candidate's PR, print the picked next action, and
    apply it (only sending a phone-push notification when `--approve` is passed — see module
    docstring)."""
    ap = argparse.ArgumentParser(
        prog="python -m src.pr_followup",
        description="Poll a submitted PR and drive it toward merge (T3.13).",
    )
    ap.add_argument(
        "--candidate", required=True, type=gate._parse_candidate, help="owner/repo#number"
    )
    ap.add_argument(
        "--approve", action="store_true", help="actually send a notify push (default: dry-run)"
    )
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    repo, number = args.candidate

    store, resolved_data_dir = resolve_store(args.data_dir)
    pr = fetch_pr_state(store, repo, number)
    if pr is None:
        print(f"{repo}#{number}: could not fetch PR state (see stderr for why).")
        return 1

    action = next_action(pr)
    message = apply_action(
        store,
        action,
        pr,
        approve=args.approve,
        drafts_dir=resolved_data_dir / review_loop._REVIEW_DRAFTS_SUBDIR,
    )
    print(f"[{action.action}] {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
