"""Attempt report (T3.12): a human-readable report per worked candidate, written on **both**
success and failure.

Every M3 stage up to this one records a machine-shaped `stage="..."` run for its own use
(`gate.py`'s bundle, `pr_quality.py`'s vote tally); nothing yet writes a *human*-readable
narrative of "what did the agent actually try, and can I re-run it myself" — the DEVPLAN's own
"transparency + learning + reproducibility" framing, and the direct feed for the M5 "Attempts"
tab (T5.11). A candidate that never got a patch attempt (only `stage="repro"`, no
`stage="verify"` yet) isn't "worked" in the sense this module cares about — it skips (`None`,
logged), the same per-item failure isolation every sibling M3 stage applies.

`render_attempt_report`'s `outcome` is read straight from the **most recent** `stage="verify"`
run's own `verified` field — never a judgment call layered on top of it. A `verified=False` run
is not a lesser attempt: DEVPLAN explicitly asks for failure reports too, since "what was tried,
why it failed" is exactly the reproducible-learning value a human can't get from a bare
`verified=False` KB record alone.

**Reproduce block:** built entirely from the verify run's own `patch`/`command` fields (a
`git apply` heredoc + the exact repro/verify command T3.3 reran) — never from a branch name
alone, since a failed attempt's branch may never have been pushed anywhere a human could fetch
it from. This makes every reproduce block self-contained and copy-pasteable regardless of
whether the candidate ever reached the fork-push step (T3.7).

**Rendered view (verified + gate-ready only):** per a note added during T3.6's first real
attempt, a candidate that's actually reached the human gate gets an additional, styled HTML
render (title + composed body if `pr_author.py` has run, diff, self-review vote, quality-gate
verdict) — the same kind of view produced by hand for `vllm-project/vllm#47600`'s sign-off, now
part of the KB artifact itself rather than an ad hoc one-off. Built from `gate.verified_diff`,
not `gate.assemble_bundle` -- the latter's risk-badge LLM call is pure overhead here (this
module never displays risk/effort/impact), the same "don't pay for `assemble_bundle` when
`verified_diff` already has what you need" rule `pr_quality.py` already follows.

**Listing worked candidates:** `list_worked` scans `store.list_runs(stage="verify")` and groups
by `(repo, number)` -- the KB is the source of truth for "who's been worked," not a scan of
already-written `.md`/`.html` files under `_ATTEMPTS_DIR`, so the M5 Attempts tab (T5.11) shows
a candidate as soon as it's verified, without requiring `python -m src.attempt_report` to have
already run for it. Mirrors `dashboard.review.review_bundle`'s own "compute live from the KB,
don't require a pre-written file" precedent.

Known limitations, not fixed here:
- Like every sibling M3 stage, this recomputes and rewrites the report from scratch on every
  call, with no idempotency check against an already-written, unchanged report for the same
  candidate -- the same gap `pr_author.py`'s own docstring documents for itself.
- The rendered HTML is a plain, self-contained document (inline CSS, no build step) rather than
  a polished design-system artifact -- appropriate for a deterministic, testable KB output; the
  M5 dashboard (T5.11) is where a livelier, interactive presentation belongs.
"""

from __future__ import annotations

import argparse
import dataclasses
import html as html_module
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import config, gate
from .agents.forecaster import TS_FORMAT
from .agents.reporter import evidence_url
from .stages import get_item_or_skip, latest_run
from .store import resolve_store
from .store.base import Store

_ATTEMPTS_SUBDIR = "attempts"
# Module-level, re-read at call time via `attempts_dir or _ATTEMPTS_DIR` (not bound into a
# function's signature at def-time) -- the same None-sentinel convention `gate.py`'s own
# `_PR_DRAFTS_DIR`/`_pr_draft_path` use, so tests can `monkeypatch.setattr` this attribute.
_ATTEMPTS_DIR = config.DATA_DIR / _ATTEMPTS_SUBDIR

_REPRO_LOG_CHARS = 2000


@dataclasses.dataclass(frozen=True)
class AttemptReport:
    """One candidate's attempt report — the three DEVPLAN-required sections plus outcome, and
    an optional rendered HTML view for a verified, gate-ready candidate."""

    repo: str
    number: int
    verified: bool
    issue_overview: str
    approach: str
    reproduce: str
    outcome: str
    rendered_html: str | None
    recorded_at: str


def _slug(repo: str, number: int) -> str:
    """Matches `engineer._branch_name`'s own `repo.replace("/", "-")` convention so a report and
    the branch it describes are trivially correlated by eye."""
    return f"{repo.replace('/', '-')}-{number}"


def _report_path(repo: str, number: int, *, attempts_dir: Path | None = None) -> Path:
    return (attempts_dir or _ATTEMPTS_DIR) / f"{_slug(repo, number)}.md"


def _html_path(repo: str, number: int, *, attempts_dir: Path | None = None) -> Path:
    return (attempts_dir or _ATTEMPTS_DIR) / f"{_slug(repo, number)}.html"


def _reproduce_block(verify_run: dict) -> str:
    """A self-contained, copy-pasteable repro sequence: apply the exact patch this attempt
    produced, then rerun the exact command that was used to check it -- built entirely from the
    verify run's own fields, never a branch name alone (see module docstring)."""
    patch = verify_run.get("patch") or ""
    command = verify_run.get("command") or ""
    lines = ["set -e"]
    if patch:
        lines += ["git apply <<'ATTEMPT_PATCH'", patch.rstrip("\n"), "ATTEMPT_PATCH"]
    if command:
        lines.append(command)
    return "```bash\n" + "\n".join(lines) + "\n```"


def _approach(repro_run: dict | None, verify_run: dict, verified: bool) -> str:
    """What was tried and what actually happened -- the repro run's baseline failing signal (if
    one was recorded) followed by the patch attempt's own outcome."""
    parts = []
    if repro_run is not None and repro_run.get("reproduced"):
        parts.append(
            f"Reproduced the failure via `{repro_run.get('command') or '(command not captured)'}`."
        )
    branch = verify_run.get("branch") or "(no branch recorded)"
    parts.append(
        f"Patched on branch `{branch}` and re-ran the same check on MI250; "
        f"the signal {'flipped to passing' if verified else 'did not flip'}."
    )
    return " ".join(parts)


def _outcome(verify_run: dict, verified: bool) -> str:
    log_tail = (verify_run.get("log") or "")[-_REPRO_LOG_CHARS:]
    if verified:
        return f"**VERIFIED** — the check passed after the patch.\n\n```\n{log_tail}\n```"
    reason = log_tail or "(no log captured)"
    return f"**FAILED** — the check still did not pass after the patch.\n\n```\n{reason}\n```"


def render_attempt_report(
    store: Store, repo: str, number: int, *, now: datetime | None = None
) -> AttemptReport | None:
    """Build (but don't write) `repo`#`number`'s attempt report, or `None` if there's no
    `stage="verify"` run yet -- a candidate the engineer hasn't attempted a patch for isn't
    "worked" in the sense this module reports on (see module docstring)."""
    item = get_item_or_skip(store, repo, number, stage="attempt_report")
    if item is None:
        return None

    verify_run = latest_run(store.list_runs(repo=repo, number=number, stage="verify"))
    if verify_run is None:
        print(f"attempt_report: no verify run for {repo}#{number}", file=sys.stderr)
        return None
    repro_run = latest_run(store.list_runs(repo=repo, number=number, stage="repro"))

    verified = bool(verify_run.get("verified"))
    issue_overview = (
        f"**{item.get('title') or '(no title)'}**\n{evidence_url(item)}\n\n"
        f"{(item.get('body') or '(no description)')[:500]}"
    )

    rendered_html = None
    if verified:
        verified_result = gate.verified_diff(store, repo, number)
        if verified_result is not None:
            diff, verify_recorded_at = verified_result
            all_runs = store.list_runs(repo=repo, number=number)
            self_review_run = latest_run(
                [
                    r
                    for r in all_runs
                    if r.get("stage") == "self_review"
                    and r.get("verify_recorded_at") == verify_recorded_at
                ]
            )
            pr_author_run = latest_run(
                [
                    r
                    for r in all_runs
                    if r.get("stage") == "pr_author"
                    and r.get("verify_recorded_at") == verify_recorded_at
                ]
            )
            pr_quality_run = None
            if pr_author_run is not None:
                pr_author_recorded_at = pr_author_run.get("recorded_at")
                pr_quality_run = latest_run(
                    [
                        r
                        for r in all_runs
                        if r.get("stage") == "pr_quality"
                        and r.get("pr_author_recorded_at") == pr_author_recorded_at
                    ]
                )
            rendered_html = _render_html(
                item,
                diff=diff,
                self_review_run=self_review_run,
                pr_author_run=pr_author_run,
                pr_quality_run=pr_quality_run,
            )

    when = now or datetime.now(timezone.utc)
    return AttemptReport(
        repo=repo,
        number=number,
        verified=verified,
        issue_overview=issue_overview,
        approach=_approach(repro_run, verify_run, verified),
        reproduce=_reproduce_block(verify_run),
        outcome=_outcome(verify_run, verified),
        rendered_html=rendered_html,
        recorded_at=when.strftime(TS_FORMAT),
    )


def list_worked(store: Store) -> list[dict]:
    """Every worked candidate -- one with at least one recorded `stage="verify"` run -- as
    ``{"repo", "number", "verified", "title", "recorded_at"}`` (the *latest* verify run's own
    outcome + timestamp), newest first. Backs the M5 "Attempts" tab (T5.11); the scan/group-by
    logic lives here rather than in `dashboard/attempts.py`, matching T5.9's own review-driven
    "scanning belongs in `src`, the dashboard module is thin glue" correction.

    A `stage="verify"` run missing `repo`/`number` (malformed) is skipped rather than raised
    on -- every other read endpoint in this codebase degrades past one bad record the same way
    instead of crashing a whole dashboard page over it.

    A candidate whose KB item record no longer exists (deleted/pruned after being verified) is
    also skipped -- a code-review finding: an earlier version listed it anyway (title falling
    back to `""`), but `render_attempt_report` requires a live item via `get_item_or_skip`, so
    that row would list here and then open to `None` when clicked, silently breaking this
    tab's own "click -> the full report renders" contract. Skipping keeps `list_worked`'s
    notion of "worked" exactly as strict as `render_attempt_report`'s.

    Titles are batched **one `store.query(repo=...)` per distinct repo**, not one
    `store.get_item` per candidate -- a code-review finding: `get_item` re-reads and
    re-parses that repo's entire JSONL file on `JsonlStore`, so a naive per-candidate call
    would re-read the same file once per worked candidate in it, the exact N-separate-reads
    anti-pattern this milestone's own review already found and fixed twice
    (`dashboard.pipeline_diagram`, `dashboard.health`) and once more in
    `selection.filter_selected`'s `_decisions_by_key`.
    """
    runs_by_candidate: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in store.list_runs(stage="verify"):
        repo = run.get("repo")
        number = run.get("number")
        if repo is None or number is None:
            continue
        runs_by_candidate[(repo, number)].append(run)

    items_by_repo: dict[str, dict[int, dict]] = {}
    summaries = []
    for (repo, number), runs in runs_by_candidate.items():
        latest = latest_run(runs)
        assert latest is not None  # `runs` is non-empty by construction
        if repo not in items_by_repo:
            items_by_repo[repo] = {i["number"]: i for i in store.query(repo=repo)}
        item = items_by_repo[repo].get(number)
        if item is None:
            continue
        summaries.append(
            {
                "repo": repo,
                "number": number,
                "verified": bool(latest.get("verified")),
                "title": item.get("title") or "",
                "recorded_at": latest.get("recorded_at"),
            }
        )
    summaries.sort(key=lambda s: s["recorded_at"] or "", reverse=True)
    return summaries


def _format_markdown(report: AttemptReport) -> str:
    sections = [
        f"# Attempt report: {report.repo}#{report.number}",
        "## Issue overview\n" + report.issue_overview,
        "## Approach\n" + report.approach,
        "## Reproduce\n" + report.reproduce,
        "## Outcome\n" + report.outcome,
    ]
    if report.rendered_html is not None:
        sections.append(
            "## Rendered draft\nThis candidate is verified and gate-ready — see "
            f"`{_slug(report.repo, report.number)}.html` for a fully rendered view of the "
            "composed PR title/body, diff, self-review vote, and quality-gate verdict."
        )
    return "\n\n".join(sections) + "\n"


def _esc(text: str) -> str:
    return html_module.escape(text)


_SAFE_URL_SCHEMES = ("http://", "https://")


def _safe_href(url: str) -> str:
    """`url` if it's http(s), else `""` -- same guard `dashboard/render.py`'s `_safe_href` uses:
    this is rendered HTML with no auth, so a stray `javascript:`/`data:` value in a malformed KB
    record must never become a clickable link."""
    return url if url.startswith(_SAFE_URL_SCHEMES) else ""


def _render_html(
    item: dict,
    *,
    diff: str,
    self_review_run: dict | None,
    pr_author_run: dict | None,
    pr_quality_run: dict | None,
) -> str:
    """A plain, self-contained HTML view of the evidence a human needs to review/approve this
    candidate -- see module docstring for why this is deliberately simple rather than a design
    exercise."""
    title = _esc((pr_author_run.get("title") if pr_author_run else None) or item.get("title") or "")
    body_html = (
        f"<pre>{_esc(pr_author_run['body'])}</pre>"
        if pr_author_run and pr_author_run.get("body")
        else "<p><em>No composed PR narrative (T3.9) yet — showing the raw diff only.</em></p>"
    )
    self_review_html = (
        f"<p>Self-review: {self_review_run.get('approve_count')}/"
        f"{self_review_run.get('total_votes')} approve</p>"
        if self_review_run
        else "<p>Self-review: (no record)</p>"
    )
    quality_html = (
        f"<p>Quality gate: {'PASSED' if pr_quality_run.get('passes') is True else 'NOT PASSED'} "
        f"({pr_quality_run.get('approve_count')}/{pr_quality_run.get('total_votes')} acceptable)"
        "</p>"
        if pr_quality_run
        else "<p>Quality gate: (no record)</p>"
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;"
        "line-height:1.5;color:#1b1d1f}"
        "pre{white-space:pre-wrap;background:#f4f5f2;padding:1rem;border-radius:8px;"
        "overflow-x:auto}"
        "h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:2rem}"
        "</style></head><body>"
        f"<h1>{title}</h1>"
        f'<p><a href="{_esc(_safe_href(evidence_url(item)))}">{_esc(evidence_url(item))}</a></p>'
        f"<h2>Body</h2>{body_html}"
        f"<h2>Evidence</h2>{self_review_html}{quality_html}"
        f"<h2>Diff</h2><pre>{_esc(diff)}</pre>"
        "</body></html>"
    )


def write_attempt_report(
    store: Store,
    repo: str,
    number: int,
    *,
    attempts_dir: Path | None = None,
    now: datetime | None = None,
) -> Path | None:
    """Render and write `repo`#`number`'s attempt report to disk, returning the `.md` path (or
    `None` if there's nothing to report yet — see `render_attempt_report`). Also writes an
    accompanying `.html` file when the candidate is verified and gate-ready."""
    report = render_attempt_report(store, repo, number, now=now)
    if report is None:
        return None
    dir_path = attempts_dir or _ATTEMPTS_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    md_path = _report_path(repo, number, attempts_dir=dir_path)
    md_path.write_text(_format_markdown(report))
    if report.rendered_html is not None:
        _html_path(repo, number, attempts_dir=dir_path).write_text(report.rendered_html)
    return md_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: write `--candidate`'s attempt report and print where it landed."""
    ap = argparse.ArgumentParser(
        prog="python -m src.attempt_report",
        description="Write a human-readable attempt report (T3.12) for a worked candidate.",
    )
    ap.add_argument(
        "--candidate", required=True, type=gate._parse_candidate, help="owner/repo#number"
    )
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    repo, number = args.candidate

    store, resolved_data_dir = resolve_store(args.data_dir)
    path = write_attempt_report(
        store, repo, number, attempts_dir=resolved_data_dir / _ATTEMPTS_SUBDIR
    )
    if path is None:
        if store.get_item(repo, number) is None:
            print(f"{repo}#{number} is not in the KB (see stderr for why).")
        else:
            print(f"{repo}#{number} has no verify run yet (see stderr for why).")
        return 1
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
