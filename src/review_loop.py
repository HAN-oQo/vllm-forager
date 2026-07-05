"""Review-response loop (T3.11): watch maintainer comments on a submitted upstream PR, draft a
point-by-point reply (plus an illustrative, unverified proposed diff when relevant) for each new
one, and — only on a separate, explicit human-approved call — post the reply as a real comment.

Part of M3.5: T3.6's own docstring puts it plainly — "PRs merge through the review conversation,
not the first push; an unanswered review is a dead PR." This module is the loop that keeps a
submitted candidate's PR alive after `gate.py`'s `--submit` opens it, without ever auto-pushing
anything to that public PR on its own.

**Two-step split, not two flags on one call:** `run_review_loop` (compose) and `post_reply`
(post) are two separate functions/CLI subcommands, not two flags checked in one function —
unlike `gate.py`'s original `--approve`/`--submit` design (which the T3.6 incident showed can be
passed together on one command line out of habit), there is no single call site here where both
"compose" and "post" could accidentally happen together. `run_review_loop` never touches `gh`
for posting; it only reads comments and writes a local draft (mirroring `gate.py`'s own
`_write_pr_draft` pattern) plus a `stage="review_response"` KB record. `post_reply` requires that
KB record to already exist — it is structurally impossible to post a reply this module never
composed and persisted in an earlier, separate invocation.

**Deliberately does not push a follow-up commit, even when it drafts a `proposed_diff`.** A
diff suggested here is composed by the same LLM that wrote the reply — it has not been through
`engineer.py`'s repro→patch→**MI250 verification** pipeline, the "empirical verification oracle"
this whole project is built around (see `docs/CONTEXT.md`/`docs/PLAN.md`). Auto-applying and
pushing an unverified diff straight onto a live, public, already-open upstream PR would bypass
that oracle entirely — a materially different (and materially worse) risk than opening a draft
PR a human must first approve. `proposed_diff` is included in the draft purely as a
copy-pasteable starting point for the human (or a future, MI250-gated automation) to apply,
verify, and push through the *existing* `engineer.py`/`gate.py` pipeline — not something this
module ever runs `git apply`/`git push` on itself.

Maintainer feedback is read via `gh api repos/{repo}/issues/{number}/comments` — GitHub's
general PR-conversation thread (a PR is an issue in GitHub's API model). Inline
line-by-line *review* comments (`.../pulls/{number}/comments`, attached to a specific diff
hunk) are a known, separate feed this module doesn't read yet — most maintainer "please change
X" feedback on a small project's PRs lands as a plain conversation comment, so this covers the
common case first.

A comment authored by the bot/agent's own authenticated `gh` identity (a previously-posted
reply) is never treated as a new maintainer comment needing a response — see
:func:`_current_gh_login`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config, gate
from .agents.forecaster import TS_FORMAT
from .stages import complete_or_none, get_item_or_skip, latest_run, record_run_best_effort
from .store import resolve_store
from .store.base import Store

_DEFAULT_GH_TIMEOUT_S = 60.0
_DIFF_CONTEXT_CHARS = 4000
_COMMENT_CONTEXT_CHARS = 4000

# Where composed-but-not-yet-posted replies land, mirroring gate.py's own `_PR_DRAFTS_DIR`
# convention exactly (a human reads the file before the separate, later `post_reply` call).
_REVIEW_DRAFTS_SUBDIR = "review_drafts"
_REVIEW_DRAFTS_DIR = config.DATA_DIR / _REVIEW_DRAFTS_SUBDIR

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "proposed_diff": {"type": ["string", "null"]},
    },
    "required": ["reply", "proposed_diff"],
}


@dataclasses.dataclass(frozen=True)
class ReviewResponse:
    """One composed (not yet posted) response to one maintainer comment."""

    repo: str
    number: int
    comment_id: int
    comment_author: str
    comment_body: str
    reply: str
    proposed_diff: str | None
    recorded_at: str

    def to_run_record(self) -> dict:
        record = dataclasses.asdict(self)
        record["stage"] = "review_response"
        return record

    def format(self) -> str:
        """Human-readable draft — what a human reads before ever running `post_reply`."""
        sections = [
            f"# Reply to {self.comment_author}'s comment on {self.repo}#{self.number}\n",
            f"## Their comment\n{self.comment_body}\n",
            f"## Proposed reply\n{self.reply}\n",
        ]
        if self.proposed_diff:
            sections.append(
                "## Proposed diff (UNVERIFIED -- not run through MI250, not auto-applied)\n"
                f"```diff\n{self.proposed_diff}\n```\n"
            )
        return "\n".join(sections)


@dataclasses.dataclass(frozen=True)
class ReviewPostResult:
    """The outcome of one :func:`post_reply` call."""

    repo: str
    number: int
    comment_id: int
    posted: bool
    posted_comment_url: str | None = None


def _current_gh_login() -> str | None:
    """The authenticated `gh` user's login, or `None` if the call fails — used to never treat
    this bot's own previously-posted replies as a new maintainer comment needing a response."""
    try:
        result = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True,
            text=True,
            check=True,
            timeout=_DEFAULT_GH_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"review_loop: failed to resolve the authenticated gh user: {exc}", file=sys.stderr)
        return None
    login = result.stdout.strip()
    return login or None


def _fetch_comments(repo: str, number: int) -> list[dict] | None:
    """Every issue-style (general PR-conversation) comment on `repo`#`number`, or `None` if the
    call failed (logged, not raised) — see module docstring for why this feed, not inline
    review comments."""
    cmd = ["gh", "api", f"repos/{repo}/issues/{number}/comments"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=_DEFAULT_GH_TIMEOUT_S
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"review_loop: failed to fetch comments for {repo}#{number}: {exc}", file=sys.stderr)
        return None
    try:
        comments = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"review_loop: gh api returned non-JSON for {repo}#{number}: {exc}", file=sys.stderr)
        return None
    return comments if isinstance(comments, list) else None


def _response_prompt(comment_body: str, title: str, diff: str) -> str:
    return (
        "A maintainer left the following review comment on your open pull request. Draft a "
        "clear, specific, point-by-point reply that directly addresses what they raised -- "
        "acknowledge valid points, explain your reasoning where you disagree, and say exactly "
        "what you'll do next. If their comment amounts to a concrete, small code-change request "
        "you can sketch, include an illustrative unified diff in `proposed_diff` (this will be "
        "shown to a human and never auto-applied, so it's fine to include one whenever it would "
        "help, even if imperfect) -- otherwise set `proposed_diff` to `null`.\n\n"
        f"PR title: {title[:500]}\n\n"
        f"Current diff (for context):\n{diff[:_DIFF_CONTEXT_CHARS]}\n\n"
        f"Maintainer's comment:\n{comment_body[:_COMMENT_CONTEXT_CHARS]}"
    )


def _compose_response(comment_body: str, title: str, diff: str) -> tuple[str, str | None] | None:
    """`(reply, proposed_diff)`, or `None` if the LLM call failed or the reply didn't shape into
    a usable response (excluded, not a fabricated reply -- the same per-candidate failure
    isolation every sibling M3 stage applies)."""
    reply_obj = complete_or_none(
        _response_prompt(comment_body, title, diff),
        _RESPONSE_SCHEMA,
        stage="review_loop",
        subject=title,
    )
    if reply_obj is None:
        return None
    reply = reply_obj.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        return None
    proposed_diff = reply_obj.get("proposed_diff")
    if not isinstance(proposed_diff, str) or not proposed_diff.strip():
        proposed_diff = None
    return reply.strip(), proposed_diff


def _draft_path(repo: str, number: int, comment_id: int, *, drafts_dir: Path | None = None) -> Path:
    """Where a composed-but-not-posted response lands — matches `gate.py`'s own
    `repo.replace('/', '-')` + `None`-sentinel-resolved-at-call-time conventions."""
    return (drafts_dir or _REVIEW_DRAFTS_DIR) / f"{repo.replace('/', '-')}-{number}-{comment_id}.md"


def run_review_loop(
    store: Store,
    repo: str,
    number: int,
    *,
    drafts_dir: Path | None = None,
    now: datetime | None = None,
) -> tuple[ReviewResponse, ...]:
    """Fetch `repo`#`number`'s maintainer comments, compose a reply (+ an illustrative,
    unverified `proposed_diff` when relevant) for each one not already responded to, and
    persist + draft-file each result. Never posts anything -- see module docstring.

    Returns the newly-composed responses (empty if there's nothing new, the item doesn't exist,
    or the `gh` fetch failed -- all skip, logged, not raised).
    """
    item = get_item_or_skip(store, repo, number, stage="review_loop")
    if item is None:
        return ()

    comments = _fetch_comments(repo, number)
    if comments is None:
        return ()

    self_login = _current_gh_login()
    already_answered = {
        r.get("comment_id")
        for r in store.list_runs(repo=repo, number=number, stage="review_response")
    }

    bundle = gate.assemble_bundle(store, repo, number)
    diff = bundle.diff if bundle is not None else ""
    title = item.get("title") or ""

    when = now or datetime.now(timezone.utc)
    responses = []
    for comment in comments:
        comment_id = comment.get("id")
        author = (comment.get("user") or {}).get("login")
        if comment_id is None or comment_id in already_answered:
            continue
        if self_login is not None and author == self_login:
            continue
        composed = _compose_response(comment.get("body") or "", title, diff)
        if composed is None:
            continue
        reply, proposed_diff = composed

        response = ReviewResponse(
            repo=repo,
            number=number,
            comment_id=comment_id,
            comment_author=author or "",
            comment_body=comment.get("body") or "",
            reply=reply,
            proposed_diff=proposed_diff,
            recorded_at=when.strftime(TS_FORMAT),
        )
        record_run_best_effort(
            store, response.to_run_record(), stage="review_response", repo=repo, number=number
        )
        path = _draft_path(repo, number, comment_id, drafts_dir=drafts_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.format())
        responses.append(response)

    return tuple(responses)


def _post_comment(repo: str, number: int, body: str) -> str | None:
    """`gh api repos/{repo}/issues/{number}/comments -f body=...` — the comment URL on success,
    or `None` if the call failed (logged, not raised)."""
    cmd = [
        "gh",
        "api",
        f"repos/{repo}/issues/{number}/comments",
        "-f",
        f"body={body}",
        "--jq",
        ".html_url",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=_DEFAULT_GH_TIMEOUT_S
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"review_loop: failed to post reply on {repo}#{number}: {exc}", file=sys.stderr)
        return None
    return result.stdout.strip() or None


def post_reply(
    store: Store, repo: str, number: int, comment_id: int, *, post: bool = False
) -> ReviewPostResult | None:
    """Post the already-composed reply for (`repo`, `number`, `comment_id`) as a real PR
    comment, but ONLY if `post` is the literal `True` (matching `gate.py`'s own "not plain
    truthiness" convention). Requires a `stage="review_response"` record for this exact
    `comment_id` to already exist — composed by an earlier, separate :func:`run_review_loop`
    call; there is no code path that composes and posts in the same call (see module
    docstring).

    Returns `None` if no composed response exists for this `comment_id` yet (skip, logged).
    """
    matching = [
        r
        for r in store.list_runs(repo=repo, number=number, stage="review_response")
        if r.get("comment_id") == comment_id
    ]
    response_run = latest_run(matching)
    if response_run is None:
        print(
            f"review_loop: no composed response for {repo}#{number} comment {comment_id} -- "
            "run the review loop first",
            file=sys.stderr,
        )
        return None

    posted = False
    posted_url = None
    if post is True:
        posted_url = _post_comment(repo, number, response_run.get("reply") or "")
        posted = posted_url is not None

    result = ReviewPostResult(
        repo=repo,
        number=number,
        comment_id=comment_id,
        posted=posted,
        posted_comment_url=posted_url,
    )
    record_run_best_effort(
        store,
        {**dataclasses.asdict(result), "stage": "review_post"},
        stage="review_post",
        repo=repo,
        number=number,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: `compose` fetches comments and drafts replies (never posts); `post`
    posts one already-composed reply for `--comment-id` (only with `--post`, see module
    docstring)."""
    ap = argparse.ArgumentParser(
        prog="python -m src.review_loop",
        description="Compose (and, separately, post) replies to maintainer PR comments.",
    )
    sub = ap.add_subparsers(dest="action", required=True)

    compose_ap = sub.add_parser("compose", help="fetch comments and draft replies (never posts)")
    compose_ap.add_argument("--candidate", required=True, type=gate._parse_candidate)
    compose_ap.add_argument("--data-dir", type=Path, default=None)

    post_ap = sub.add_parser("post", help="post one already-composed reply")
    post_ap.add_argument("--candidate", required=True, type=gate._parse_candidate)
    post_ap.add_argument("--comment-id", required=True, type=int)
    post_ap.add_argument("--post", action="store_true", help="actually post (default: dry-run)")
    post_ap.add_argument("--data-dir", type=Path, default=None)

    args = ap.parse_args(argv)
    repo, number = args.candidate
    store, resolved_data_dir = resolve_store(args.data_dir)

    if args.action == "compose":
        drafts_dir = resolved_data_dir / _REVIEW_DRAFTS_SUBDIR
        responses = run_review_loop(store, repo, number, drafts_dir=drafts_dir)
        if not responses:
            print(f"No new maintainer comments to respond to for {repo}#{number}.")
            return 0
        for response in responses:
            path = _draft_path(repo, number, response.comment_id, drafts_dir=drafts_dir)
            print(
                f"Drafted a reply to comment {response.comment_id} by "
                f"{response.comment_author}: {path}"
            )
        return 0

    result = post_reply(store, repo, number, args.comment_id, post=args.post)
    if result is None:
        print(f"No composed response for {repo}#{number} comment {args.comment_id}.")
        return 1
    if result.posted:
        print(f"Posted: {result.posted_comment_url}")
    else:
        print("Not posted (pass --post to actually post the already-composed reply).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
