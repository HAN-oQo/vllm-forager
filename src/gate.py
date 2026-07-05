"""Human gate (T3.5): assemble the full evidence bundle for a verified, self-reviewed candidate
and open a draft PR — but **only** on an explicit `approve=True`.

This is CLAUDE.md's mandatory human checkpoint: **HARD requirement — nothing reaches upstream
without a person seeing the full evidence bundle and approving.** `run_gate` checks `approve is
True` explicitly (not plain truthiness) before ever calling :func:`_create_draft_pr` — a caller
passing anything other than the literal `True` (a truthy string, a non-empty container) does
**not** trigger a PR, so the hard rule holds regardless of what a future caller passes, not just
for today's CLI (whose `argparse action="store_true"` always produces a real `bool` anyway). The
CLI's own `main()` prints the bundle to the terminal *before* ever calling into the approval
path — CLAUDE.md's word is "seeing," and a human running `--approve` interactively must see the
bundle in their own terminal before the PR exists, not after. This module's own test suite
proves the unapproved path never touches `gh` at all — see `tests/test_gate.py`'s own name for
that guarantee. Nothing here ever merges anything either (`gh pr create --draft` opens a *draft*;
CLAUDE.md's separate rule that agents never call `gh pr merge` is enforced at the environment
level, not by this module, but this module doesn't need or attempt to touch it).

A candidate only reaches the gate once the full T3.2→T3.4 chain agrees: a **specific** verify
attempt was `verified=True`, *and* the self-review run for **that same attempt** (matched via
:attr:`~src.self_review.SelfReviewResult.verify_recorded_at`, not just "the latest self-review
regardless") had `advance=True`. Both `assemble_bundle` and `run_gate` skip (`None`, logged) —
not raise — for any candidate that isn't there yet: no verify run, the latest verify run isn't
verified, no self-review matching that specific verify attempt, or that self-review didn't
advance. This mirrors every sibling M3 stage's per-item failure isolation, applied one notch
stricter: a self-review `hold` must **not** reach the human's attention at all (T3.4's own
"Why": the adversarial vote exists specifically to filter what the human sees).

The **risk badge** the DEVPLAN asks for isn't persisted anywhere by an earlier stage — T2.5's
Scout scores candidates at discovery time but (by its own documented design) never writes that
score back to the KB. Rather than invent a second, redundant risk-scoring KB field, this module
reuses :func:`~src.agents.scout._score` directly (the same LLM call Scout itself makes) to
produce a fresh risk/effort/impact readout for the gate — see :func:`_risk_badge`. Known
cosmetic limitation: `_score`'s own failure log line is prefixed `"scout:"`, since it doesn't
know it's being called from here — a real hiccup shows up in stderr under the wrong module name,
but the bundle still assembles fine (`risk=None`), so this doesn't affect correctness.

Unlike every other M3 stage, `run_gate` persists its own outcome (`stage="gate"`: `approved`,
`pr_url`, `recorded_at`) — the terminal, human-facing checkpoint deserves the same KB audit
trail T3.2–T3.4 already leave, and a durable record is also what a future caller would need to
notice "this candidate was already approved" before re-approving it (not implemented here —
this module doesn't check its own gate history before opening a second PR on a re-run; that
idempotency check is future work once T3.6 actually exercises repeat runs).

Known limitation, not fixed here: `engineer.py`'s patch is committed to a branch on the MI250
host's own local checkout, not pushed to the real GitHub remote — `gh pr create --draft --head
<branch>` requires that branch to already exist on the remote. Pushing it there (from the MI250
host, since that's where the commit physically lives) is a prerequisite this module doesn't
perform itself; DEVPLAN's own T3.5 test bullet scopes this module to "gh invoked" (mocked), not
to the push step, and guessing at that integration's shape now — before T3.6's first real PR
actually exercises it — would be premature. Also not fixed: `approve` records no approver
identity, only that *an* approval happened and when — acceptable for a single-operator project
(the person with shell access on `ce-master`), but a real gap if this ever runs with more than
one person able to invoke it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .agents.forecaster import TS_FORMAT
from .agents.reporter import evidence_url
from .agents.scout import _score
from .stages import get_item_or_skip, latest_run, record_run_best_effort
from .store import resolve_store
from .store.base import Store

_DEFAULT_GH_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class EvidenceBundle:
    """Everything a human needs to approve or reject one candidate — the DEVPLAN's own
    ``{diff, risk badge, repro evidence, MI250 logs, self-review votes}`` shape."""

    repo: str
    number: int
    evidence_url: str
    title: str
    branch: str
    diff: str
    risk: str | None
    effort: str | None
    impact: str | None
    repro_command: str
    repro_log: str
    verify_log: str
    self_review_votes: tuple[dict, ...]
    approve_count: int
    total_votes: int

    def format(self) -> str:
        """A human-readable rendering of the bundle — what the CLI prints before asking for
        approval, and the body of the draft PR :func:`run_gate` opens once approved. Reads each
        vote defensively (`.get`, not `[...]`) since a vote dict's shape is only as trustworthy
        as whatever wrote it to the store, unlike the rest of this method's own fields, which
        already default at construction time in :func:`assemble_bundle`."""
        votes_lines = "\n".join(
            f"  - {'✅' if v.get('looks_correct') else '❌'} {v.get('reason', '(no reason given)')}"
            for v in self.self_review_votes
        )
        risk_line = f"risk={self.risk} effort={self.effort} impact={self.impact}"
        return (
            f"# Candidate {self.repo}#{self.number}: {self.title}\n\n"
            f"Evidence: {self.evidence_url}\n"
            f"Risk badge: {risk_line}\n\n"
            f"## Repro\n```\n{self.repro_command}\n```\n{self.repro_log}\n\n"
            f"## Patch (branch `{self.branch}`)\n```diff\n{self.diff}\n```\n\n"
            f"## Verify (MI250 log)\n```\n{self.verify_log}\n```\n\n"
            f"## Self-review ({self.approve_count}/{self.total_votes} approve)\n{votes_lines}\n"
        )


@dataclass(frozen=True)
class GateResult:
    """The outcome of one :func:`run_gate` call."""

    repo: str
    number: int
    bundle: EvidenceBundle
    approved: bool
    pr_url: str | None


def _risk_badge(title: str, body: str) -> tuple[str | None, str | None, str | None]:
    """`(risk, effort, impact)` via the same LLM call T2.5's Scout uses, or all `None` if that
    call fails (a missing risk badge is a display gap, not a reason to withhold the rest of
    the bundle — see module docstring)."""
    scores = _score(title, body, "verified-candidate")
    if scores is None:
        return None, None, None
    return scores["risk"], scores["effort"], scores["impact"]


def assemble_bundle(store: Store, repo: str, number: int) -> EvidenceBundle | None:
    """Gather the full evidence bundle for (`repo`, `number`), or `None` if it isn't ready for
    the gate yet (see module docstring for the exact readiness conditions — all skip, not
    raise)."""
    item = get_item_or_skip(store, repo, number, stage="gate")
    if item is None:
        return None

    all_runs = store.list_runs(repo=repo, number=number)
    verify_run = latest_run([r for r in all_runs if r.get("stage") == "verify"])
    if verify_run is None:
        print(f"gate: no verify run for {repo}#{number}", file=sys.stderr)
        return None
    if not verify_run.get("verified"):
        print(f"gate: most recent verify run for {repo}#{number} is not verified", file=sys.stderr)
        return None

    verify_recorded_at = verify_run.get("recorded_at")
    matching_reviews = [
        r
        for r in all_runs
        if r.get("stage") == "self_review"
        and verify_recorded_at is not None
        and r.get("verify_recorded_at") == verify_recorded_at
    ]
    self_review_run = latest_run(matching_reviews)
    if self_review_run is None:
        print(f"gate: no self-review for {repo}#{number}'s current verify run", file=sys.stderr)
        return None
    if not self_review_run.get("advance"):
        print(f"gate: self-review for {repo}#{number} did not advance", file=sys.stderr)
        return None

    reproduced_runs = [r for r in all_runs if r.get("stage") == "repro" and r.get("reproduced")]
    repro_run = latest_run(reproduced_runs) or {}

    title = item.get("title") or ""
    risk, effort, impact = _risk_badge(title, item.get("body") or "")

    return EvidenceBundle(
        repo=repo,
        number=number,
        evidence_url=evidence_url(item),
        title=title,
        branch=verify_run.get("branch") or "",
        diff=verify_run.get("patch") or "",
        risk=risk,
        effort=effort,
        impact=impact,
        repro_command=repro_run.get("command") or "",
        repro_log=repro_run.get("log") or "",
        verify_log=verify_run.get("log") or "",
        self_review_votes=tuple(self_review_run.get("critiques") or ()),
        approve_count=self_review_run.get("approve_count") or 0,
        total_votes=self_review_run.get("total_votes") or 0,
    )


def _create_draft_pr(bundle: EvidenceBundle) -> str | None:
    """`gh pr create --draft` for `bundle`'s branch — the PR URL `gh` prints on success, or
    `None` if the call failed (logged, not raised: the evidence bundle was still validly
    assembled and approved, only the PR creation step itself failed)."""
    cmd = [
        "gh",
        "pr",
        "create",
        "--draft",
        "--repo",
        bundle.repo,
        "--head",
        bundle.branch,
        "--title",
        f"[vllm-forager] Fix for {bundle.repo}#{bundle.number}",
        "--body",
        bundle.format(),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=_DEFAULT_GH_TIMEOUT_S
        )
    except FileNotFoundError as exc:
        print(f"gate: `gh` not found on PATH: {exc}", file=sys.stderr)
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(
            f"gate: gh pr create failed for {bundle.repo}#{bundle.number}: {exc}", file=sys.stderr
        )
        return None
    return result.stdout.strip()


def _finalize(
    store: Store,
    repo: str,
    number: int,
    bundle: EvidenceBundle,
    *,
    approve: bool,
    now: datetime | None = None,
) -> GateResult:
    """Approve-or-not `bundle` (`approve is True` is the only thing that ever opens a PR — see
    module docstring), record the `stage="gate"` outcome, and return the result. Shared by
    :func:`run_gate` (assemble+decide in one call) and `main` (which prints `bundle` in between
    assembling and calling this, so the human sees it before any PR exists)."""
    approved = approve is True
    pr_url = _create_draft_pr(bundle) if approved else None

    when = now or datetime.now(timezone.utc)
    record_run_best_effort(
        store,
        {
            "repo": repo,
            "number": number,
            "stage": "gate",
            "approved": approved,
            "pr_url": pr_url,
            "recorded_at": when.strftime(TS_FORMAT),
        },
        stage="gate",
        repo=repo,
        number=number,
    )
    return GateResult(repo=repo, number=number, bundle=bundle, approved=approved, pr_url=pr_url)


def run_gate(
    store: Store,
    repo: str,
    number: int,
    *,
    approve: bool = False,
    now: datetime | None = None,
) -> GateResult | None:
    """Assemble the evidence bundle for (`repo`, `number`) and, **only if `approve` is the
    literal `True`**, open a draft PR for it. Records the outcome as a `stage="gate"` run
    regardless (see module docstring).

    Returns:
        `None` if the candidate isn't ready for the gate (see :func:`assemble_bundle`; skipped,
        not raised). Otherwise a :class:`GateResult` — `pr_url` is `None` whenever `approve`
        isn't `True` (by construction: :func:`_create_draft_pr` is only ever called in that
        branch) or if the `gh` call itself failed.
    """
    bundle = assemble_bundle(store, repo, number)
    if bundle is None:
        return None
    return _finalize(store, repo, number, bundle, approve=approve, now=now)


def _parse_candidate(raw: str) -> tuple[str, int]:
    repo, _, number = raw.rpartition("#")
    if not repo or not number.isdigit():
        raise argparse.ArgumentTypeError(f"--candidate must be 'owner/repo#number', got {raw!r}")
    return repo, int(number)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the evidence bundle for one candidate, then — **only** if
    `--approve` was passed — open a draft PR for it. The bundle is always printed *before* the
    approval branch runs (see module docstring: CLAUDE.md's "seeing the bundle" requirement
    means the human's own terminal must show it before any PR exists, even when `--approve` is
    passed on the very first invocation).
    """
    ap = argparse.ArgumentParser(
        prog="python -m src.gate",
        description="Print a candidate's evidence bundle; --approve opens a draft PR for it.",
    )
    ap.add_argument("--candidate", required=True, type=_parse_candidate, help="owner/repo#number")
    ap.add_argument("--approve", action="store_true", help="open a draft PR (default: print only)")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Read a JSONL store at this path instead of the STORE-selected backend "
            "(default: config.DATA_DIR, backend from env STORE=jsonl|firestore)."
        ),
    )
    args = ap.parse_args(argv)
    repo, number = args.candidate

    store, _ = resolve_store(args.data_dir)
    bundle = assemble_bundle(store, repo, number)
    if bundle is None:
        print(f"{repo}#{number} is not ready for the gate (see stderr for why).")
        return 1

    print(bundle.format())

    result = _finalize(store, repo, number, bundle, approve=args.approve)
    if args.approve:
        print(f"\nOpened draft PR: {result.pr_url}" if result.pr_url else "\ngh pr create failed.")
    else:
        print("\n(not approved -- pass --approve to open a draft PR)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
