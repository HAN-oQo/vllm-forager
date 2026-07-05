"""Human gate (T3.5/T3.7): assemble the full evidence bundle for a verified, self-reviewed
candidate; `approve` writes a local PR-draft artifact; `submit` (only together with `approve`)
is the **only** path that ever opens a real PR against `bundle.repo`.

**Incident this module now defends against:** T3.6's first real run opened a genuine, public
draft PR directly against `vllm-project/vllm` (vllm-project/vllm#47645, since withdrawn) from a
single `--approve` flag whose only job, before this fix, was "open the PR." A person granting
that one flag does not necessarily register that "draft" here means a live, public PR on someone
else's repository, not a local, contained artifact — the two are conflated by ordinary English
but not by what actually happens. `approve` and `submit` are now two separate flags precisely so
that the dangerous action (a real `gh pr create` against a real repo) requires an explicit,
second, unambiguous decision, distinct from "this candidate looks correct."

- `approve=True, submit=False` (or omitted): writes a PR-draft file to disk
  (:func:`_write_pr_draft`) and records the outcome — **no network call to GitHub for PR
  creation happens.** This is the safe default for "I've looked at the bundle and it's correct."
- `approve=True, submit=True`: additionally calls :func:`_create_draft_pr` — the one and only
  code path in this whole codebase that opens a PR against an upstream repo. Checked as `approve
  is True and submit is True` (not plain truthiness — see below), so a caller passing anything
  other than the literal `True` for either flag never submits, regardless of what a future caller
  passes. **`submit` only takes effect if a draft for this exact candidate already existed
  *before* this call** (`GateResult.draft_is_new` reports which) — a review finding on this very
  fix pointed out that two required flags checked in one function call are still just as easy to
  pass together, out of habit, on a single command line as the one flag they replaced. Requiring
  the draft to predate the submit call forces at least two *separate* invocations: one that
  writes the draft (nothing to review yet — `submit` is silently inert on this first call even
  if passed), and a later one, after a human has had the chance to actually open and read the
  file, that submits it. This still can't verify anyone *did* read it — no code-only mechanism
  can — but it closes the specific gap of "both flags, one command, no real gap in between."
- `approve=False`: no draft, no submission, regardless of `submit` — matches CLAUDE.md's mandatory
  human checkpoint (**HARD requirement: nothing reaches upstream without a person seeing the full
  evidence bundle and approving**). The CLI's own `main()` still prints the bundle *before* ever
  calling into `_finalize`, for the same "seeing before deciding" reason as before this fix.

Nothing here ever merges anything either (`gh pr create --draft` opens a *draft*; CLAUDE.md's
separate rule that agents never call `gh pr merge` is enforced at the environment level, not by
this module).

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

`run_gate` persists its own outcome (`stage="gate"`: `approved`, `submitted`, `pr_url`,
`fork_owner`, `draft_path`, `recorded_at`) — the terminal, human-facing checkpoint deserves the
same KB audit trail T3.2–T3.4 already leave, and a durable record is also what a future caller
would need to notice "this candidate was already submitted" before re-submitting it (not
implemented here — this module doesn't check its own gate history before opening a second PR on
a re-run; that idempotency check is future work).

`_create_draft_pr`'s `fork_owner` parameter (`--head {fork_owner}:{branch}`) is required for a
real cross-repo PR — plain `--head <branch>` makes `gh` look for the branch *inside* `bundle.repo`
itself and fails ("No commits between main and <branch>") for a fork-hosted branch. Not
automated: this module doesn't derive `fork_owner` from anything (a caller must know and pass
it) — see `docs/DEVPLAN.md`'s M3.5 for the larger contribution-quality work this incident also
motivated (T3.8–T3.11).

Also not fixed: `approve`/`submit` record no approver identity, only that *an* approval/
submission happened and when — acceptable for a single-operator project (the person with shell
access on `ce-master`), but a real gap if this ever runs with more than one person able to
invoke it.

**T3.10.5 wiring:** for a while, T3.9's composed narrative (`pr_author.py`) and T3.10's quality
verdict (`pr_quality.py`) each existed as standalone modules with no caller in this module —
`--submit` could still open a real PR carrying `EvidenceBundle.format()`'s raw evidence dump
(the same `# Candidate … Risk badge` shape that got #47645 withdrawn), with no quality check in
the way at all. `_finalize`/`run_gate` now take `pr_body: str | None` (used as the draft/PR body
instead of `bundle.format()` when given) and `quality_passed: bool | None` (`submit` is now
inert unless this is the literal `True`, on top of the existing three conditions). This module
deliberately does **not** import `pr_author`/`pr_quality` at module level — both already import
`gate`, so that would be a real circular import, not a style nit. Instead `main()` *reads* (never
re-runs) the latest persisted `stage="pr_author"`/`stage="pr_quality"` KB records directly via
`store.list_runs`, cross-checking `recorded_at` so a quality verdict for an older, superseded
narrative is never read as covering the current one. Reading rather than re-invoking also avoids
a subtler bug: re-running `pr_author.run_pr_author` on the second (`--submit`) call would make a
fresh, non-deterministic LLM call and could compose a body different from the one a human
actually read in the draft file written by the first call — the exact "the human saw something
different from what got submitted" gap this whole module exists to close.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .agents.forecaster import TS_FORMAT
from .agents.reporter import evidence_url
from .agents.scout import _score
from .stages import get_item_or_skip, latest_run, record_run_best_effort
from .store import resolve_store
from .store.base import Store

_DEFAULT_GH_TIMEOUT_S = 60.0

# Where approved-but-not-yet-submitted PR bodies land -- a real file a human can open and read
# at leisure, not just fast-scrolling terminal output (see module docstring's own incident note
# on why "seeing the bundle" print alone wasn't a strong enough safeguard). A single named
# constant, not a literal repeated at both the default (below) and the CLI's --data-dir override
# (in main()), so the two can't drift out of sync.
_PR_DRAFTS_SUBDIR = "pr_drafts"
_PR_DRAFTS_DIR = config.DATA_DIR / _PR_DRAFTS_SUBDIR


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
    # Which verify run's `patch`/`log` this bundle was built from -- lets a later stage
    # (pr_author.py, pr_quality.py) detect a candidate that was re-verified after they last
    # looked at it, instead of silently comparing against a stale pairing. Defaulted (appended
    # last) so existing direct `EvidenceBundle(...)` construction call sites don't break.
    verify_recorded_at: str = ""

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
    """The outcome of one :func:`run_gate` call.

    `submitted` means submission was *attempted* (`approved`, `submit is True`, and the draft
    already existed from an earlier call — see module docstring) — it does **not** mean `gh pr
    create` succeeded; check `pr_url` for that (`submitted=True, pr_url=None` is exactly the
    "gh call failed" case). `approved` alone (`submitted=False`) means only `draft_path` was
    written, no PR was opened. `draft_is_new` is `True` when *this* call is the one that wrote
    the candidate's draft file for the first time — the reason `submitted` came back `False`
    even though `submit=True` was passed is almost always this. `fork_owner` records what was
    actually passed (or `None`) — the KB's own audit trail for a cross-repo PR should show which
    fork the head branch was addressed against, not just that approval happened. `quality_passed`
    records what was passed in (`None` if no T3.10 verdict was available) — a `submitted=False`
    despite `submit=True` and a pre-existing draft is otherwise this, not `draft_is_new`."""

    repo: str
    number: int
    bundle: EvidenceBundle
    approved: bool
    submitted: bool = False
    pr_url: str | None = None
    fork_owner: str | None = None
    draft_path: Path | None = None
    draft_is_new: bool = False
    quality_passed: bool | None = None


def _risk_badge(title: str, body: str) -> tuple[str | None, str | None, str | None]:
    """`(risk, effort, impact)` via the same LLM call T2.5's Scout uses, or all `None` if that
    call fails (a missing risk badge is a display gap, not a reason to withhold the rest of
    the bundle — see module docstring)."""
    scores = _score(title, body, "verified-candidate")
    if scores is None:
        return None, None, None
    return scores["risk"], scores["effort"], scores["impact"]


def _readiness(store: Store, repo: str, number: int) -> tuple[dict, dict, dict, list[dict]] | None:
    """`(item, verify_run, self_review_run, all_runs)` if (`repo`, `number`) is gate-ready, else
    `None` (skip, logged) — the exact readiness conditions `assemble_bundle` requires, factored
    out so a caller that doesn't need the full bundle (:func:`verified_diff`, for
    `pr_quality.py`) can check readiness and read the verify run without also paying for
    `assemble_bundle`'s own risk-badge LLM call (:func:`_risk_badge`) or repro/self-review-vote
    assembly it would never use. `all_runs` is returned too so `assemble_bundle` doesn't re-scan
    the store a second time for the repro lookup."""
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

    return item, verify_run, self_review_run, all_runs


def assemble_bundle(store: Store, repo: str, number: int) -> EvidenceBundle | None:
    """Gather the full evidence bundle for (`repo`, `number`), or `None` if it isn't ready for
    the gate yet (see module docstring for the exact readiness conditions — all skip, not
    raise)."""
    ready = _readiness(store, repo, number)
    if ready is None:
        return None
    item, verify_run, self_review_run, all_runs = ready

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
        verify_recorded_at=verify_run.get("recorded_at") or "",
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


def verified_diff(store: Store, repo: str, number: int) -> tuple[str, str] | None:
    """`(diff, verify_recorded_at)` for (`repo`, `number`) if it's gate-ready, else `None` — the
    one thing a caller needs to judge or compare a diff against (`pr_quality.py`), without
    paying for `assemble_bundle`'s own risk-badge LLM call or repro/self-review-vote assembly."""
    ready = _readiness(store, repo, number)
    if ready is None:
        return None
    _item, verify_run, _self_review_run, _all_runs = ready
    return verify_run.get("patch") or "", verify_run.get("recorded_at") or ""


def _pr_draft_path(repo: str, number: int, *, drafts_dir: Path | None = None) -> Path:
    """Where :func:`_write_pr_draft` puts (and a human can find) one candidate's draft — matches
    :func:`~src.engineer._branch_name`'s own ``repo.replace("/", "-")`` convention so the two are
    trivially correlated by eye. `drafts_dir` left `None` (default) re-reads the module-level
    `_PR_DRAFTS_DIR` attribute (itself `config.DATA_DIR`'s value at `gate.py` import time) at
    call time rather than binding it into this function's own signature at `def`-time —
    specifically so tests can `monkeypatch.setattr(gate, "_PR_DRAFTS_DIR", ...)`. Every caller
    that already knows a specific data dir (the CLI's own `--data-dir`) should still pass its own
    explicitly, though: a caller pointed at a test/alternate store would otherwise have its
    drafts land in the real shared `config.DATA_DIR` regardless (this was a real bug caught by
    this module's own Demo run)."""
    return (drafts_dir or _PR_DRAFTS_DIR) / f"{repo.replace('/', '-')}-{number}.md"


def _write_pr_draft(bundle: EvidenceBundle, body: str, *, drafts_dir: Path | None = None) -> Path:
    """Write `body` (`bundle.format()`, rendered once by the caller — see `_finalize`) to disk
    and return the path — the artifact a human reviews (at their own pace, in their own editor)
    before ever running `--submit`. Content is identical to what the CLI already prints; raising
    the bar on the *content* itself (structured problem / root-cause / repro sections, not a raw
    evidence dump) is M3.5's T3.9, not this fix."""
    path = _pr_draft_path(bundle.repo, bundle.number, drafts_dir=drafts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _create_draft_pr(
    bundle: EvidenceBundle, body: str, *, fork_owner: str | None = None
) -> str | None:
    """`gh pr create --draft` for `bundle`'s branch, with `body` (`bundle.format()`, rendered
    once by the caller — see `_finalize`) as the PR body — the PR URL `gh` prints on success, or
    `None` if the call failed (logged, not raised: the evidence bundle was still validly
    assembled and approved, only the PR creation step itself failed).

    `fork_owner`, if given, is prefixed onto `--head` as `{fork_owner}:{branch}` — `gh`'s
    required form when the branch lives on a fork rather than `bundle.repo` itself (T3.6's own
    real run discovered this the hard way: `--head <branch>` alone makes `gh` look for that
    branch *inside* `bundle.repo`, failing with "No commits between main and <branch>" / "Head
    ref must be a branch" for a fork-hosted branch that's never existed there). Left `None`
    (default) to preserve prior behavior for a same-repo head."""
    head = f"{fork_owner}:{bundle.branch}" if fork_owner else bundle.branch
    cmd = [
        "gh",
        "pr",
        "create",
        "--draft",
        "--repo",
        bundle.repo,
        "--head",
        head,
        "--title",
        f"[vllm-forager] Fix for {bundle.repo}#{bundle.number}",
        "--body",
        body,
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
    submit: bool = False,
    fork_owner: str | None = None,
    pr_drafts_dir: Path | None = None,
    now: datetime | None = None,
    pr_body: str | None = None,
    quality_passed: bool | None = None,
) -> GateResult:
    """Approve-or-not `bundle`, submit-or-not the approved draft, record the `stage="gate"`
    outcome, and return the result. Shared by :func:`run_gate` (assemble+decide in one call) and
    `main` (which prints `bundle` in between assembling and calling this, so the human sees it
    before any decision is acted on).

    Opening a real PR (:func:`_create_draft_pr`) requires **four** things, not two: `approve is
    True`, `submit is True`, the candidate's draft file already existing *before* this call (i.e.
    written by an earlier, separate call), AND `quality_passed is True` — see module docstring
    for why the third and fourth conditions exist. `GateResult.draft_is_new`/`quality_passed`
    report which of these is almost always why `submit=True` didn't actually submit.

    `pr_body`, if given (T3.9's composed narrative), is written/submitted instead of
    `bundle.format()`'s raw evidence-dump rendering — the draft a human reviews and the PR that
    gets opened are always the same document either way, just a better one when `pr_body` exists.

    `fork_owner` is passed straight through to :func:`_create_draft_pr` — see its own docstring
    — and persisted in the `stage="gate"` record too, so the KB's own audit trail for a
    cross-repo PR shows which fork (if any) the head branch was addressed against.
    `pr_drafts_dir` defaults to `config.DATA_DIR`'s own location — a caller reading from an
    alternate store (the CLI's `--data-dir`) should pass its own, see `_pr_draft_path`'s own
    docstring for why."""
    approved = approve is True
    draft_is_new = False
    draft_path = None
    body = None
    if approved:
        body = pr_body if pr_body is not None else bundle.format()
        target_path = _pr_draft_path(repo, number, drafts_dir=pr_drafts_dir)
        draft_is_new = not target_path.exists()
        draft_path = _write_pr_draft(bundle, body, drafts_dir=pr_drafts_dir)
    submitted = approved and submit is True and not draft_is_new and quality_passed is True
    pr_url = None
    if submitted:
        assert body is not None  # submitted implies approved implies body was set above
        pr_url = _create_draft_pr(bundle, body, fork_owner=fork_owner)

    when = now or datetime.now(timezone.utc)
    record_run_best_effort(
        store,
        {
            "repo": repo,
            "number": number,
            "stage": "gate",
            "approved": approved,
            "submitted": submitted,
            "pr_url": pr_url,
            "fork_owner": fork_owner,
            "draft_path": str(draft_path) if draft_path else None,
            "draft_is_new": draft_is_new,
            "quality_passed": quality_passed,
            "recorded_at": when.strftime(TS_FORMAT),
        },
        stage="gate",
        repo=repo,
        number=number,
    )
    return GateResult(
        repo=repo,
        number=number,
        bundle=bundle,
        approved=approved,
        submitted=submitted,
        pr_url=pr_url,
        fork_owner=fork_owner,
        draft_path=draft_path,
        draft_is_new=draft_is_new,
        quality_passed=quality_passed,
    )


def run_gate(
    store: Store,
    repo: str,
    number: int,
    *,
    approve: bool = False,
    submit: bool = False,
    fork_owner: str | None = None,
    pr_drafts_dir: Path | None = None,
    now: datetime | None = None,
    pr_body: str | None = None,
    quality_passed: bool | None = None,
) -> GateResult | None:
    """Assemble the evidence bundle for (`repo`, `number`); if `approve` is the literal `True`,
    write a local PR-draft file; **only if `approve` and `submit` are both the literal `True`,
    the draft already existed from an earlier, separate call, AND `quality_passed` is the literal
    `True`**, additionally open a real PR against `repo`. Records the outcome as a `stage="gate"`
    run regardless (see module docstring for why submission needs conditions beyond the two
    flags).

    `pr_body`/`quality_passed` are plain values, not looked up here — this function doesn't
    import `pr_author`/`pr_quality` (see module docstring for why); a caller (`main()`) that
    wants T3.9/T3.10 in the loop reads their latest KB records itself and passes the results in.

    `fork_owner`, if given (e.g. `"HAN-oQo"`), tells `gh` the branch lives on that fork rather
    than `repo` itself — see :func:`_create_draft_pr`'s own docstring for why this is required
    for any real cross-repo PR. Left `None` (default) for a same-repo head. `pr_drafts_dir`
    defaults to `config.DATA_DIR`'s own location — pass your own if `store` reads from
    somewhere else, see `_pr_draft_path`'s own docstring for why.

    Returns:
        `None` if the candidate isn't ready for the gate (see :func:`assemble_bundle`; skipped,
        not raised). Otherwise a :class:`GateResult` — check `submitted` (not just `approve`/
        `submit`, the arguments) for whether submission was actually attempted, and `pr_url` for
        whether it actually succeeded; `draft_is_new`/`quality_passed` explain a `submitted=False`
        despite `submit=True`.
    """
    bundle = assemble_bundle(store, repo, number)
    if bundle is None:
        return None
    return _finalize(
        store,
        repo,
        number,
        bundle,
        approve=approve,
        submit=submit,
        fork_owner=fork_owner,
        pr_drafts_dir=pr_drafts_dir,
        now=now,
        pr_body=pr_body,
        quality_passed=quality_passed,
    )


def _current_narrative(store: Store, repo: str, number: int) -> tuple[str | None, bool | None]:
    """`(pr_body, quality_passed)` for (`repo`, `number`), read directly from the latest
    persisted `stage="pr_author"`/`stage="pr_quality"` KB records — never by calling into
    `pr_author.py`/`pr_quality.py` themselves (see module docstring for why: a circular import,
    and re-running composition here could produce a body different from the one already written
    to a draft file). `pr_body` is `None` if no `pr_author` run exists yet (caller falls back to
    `bundle.format()`). `quality_passed` is `None` (treated as "not passed" by `_finalize`) if no
    `pr_author` run exists, no `pr_quality` run exists, or the latest `pr_quality` run's
    `pr_author_recorded_at` doesn't match the `pr_author` run being used — a verdict for a
    superseded narrative must never be read as covering the current one."""
    pr_author_run = latest_run(store.list_runs(repo=repo, number=number, stage="pr_author"))
    if pr_author_run is None:
        return None, None
    pr_body = pr_author_run.get("body") or None

    pr_quality_run = latest_run(store.list_runs(repo=repo, number=number, stage="pr_quality"))
    if pr_quality_run is None:
        return pr_body, None
    if pr_quality_run.get("pr_author_recorded_at") != pr_author_run.get("recorded_at"):
        return pr_body, None
    return pr_body, bool(pr_quality_run.get("passes"))


def _parse_candidate(raw: str) -> tuple[str, int]:
    repo, _, number = raw.rpartition("#")
    if not repo or not number.isdigit():
        raise argparse.ArgumentTypeError(f"--candidate must be 'owner/repo#number', got {raw!r}")
    return repo, int(number)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the evidence bundle for one candidate, then act per `--approve`/
    `--submit` (see module docstring for why these are two separate, both-required flags — a
    real PR against an upstream repo is the one action that must never follow from a single
    flag). The bundle is always printed *before* either branch runs (CLAUDE.md's "seeing the
    bundle" requirement means the human's own terminal must show it before any decision is
    acted on, even when both flags are passed on the very first invocation).
    """
    ap = argparse.ArgumentParser(
        prog="python -m src.gate",
        description=(
            "Print a candidate's evidence bundle; --approve writes a local PR-draft file; "
            "--submit (only together with --approve) opens a real PR against --candidate's repo."
        ),
    )
    ap.add_argument("--candidate", required=True, type=_parse_candidate, help="owner/repo#number")
    ap.add_argument(
        "--approve", action="store_true", help="write a local PR-draft file (default: print only)"
    )
    ap.add_argument(
        "--submit",
        action="store_true",
        help=(
            "open a real PR against the candidate's repo -- only takes effect together with "
            "--approve; this is the one flag combination that ever calls `gh pr create` "
            "against an upstream repo, see module docstring's incident note"
        ),
    )
    ap.add_argument(
        "--fork-owner",
        default=None,
        help=(
            "GitHub owner of the fork the candidate branch actually lives on (e.g. "
            "'HAN-oQo') -- required for --repo to be a real upstream repo like "
            "vllm-project/vllm rather than the fork itself; see _create_draft_pr's docstring."
        ),
    )
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

    store, resolved_data_dir = resolve_store(args.data_dir)
    bundle = assemble_bundle(store, repo, number)
    if bundle is None:
        print(f"{repo}#{number} is not ready for the gate (see stderr for why).")
        return 1

    print(bundle.format())

    pr_body, quality_passed = _current_narrative(store, repo, number)
    if pr_body is None:
        print(
            "\nNo composed PR narrative yet (T3.9) -- the draft/PR body above will use the raw "
            "evidence bundle. Run `python -m src.pr_author --candidate "
            f"{repo}#{number}` first for a maintainer-grade body."
        )
    else:
        print(f"\nPR-quality gate (T3.10): {'PASSED' if quality_passed else 'NOT PASSED'}")
        if not quality_passed:
            print(
                f"Run `python -m src.pr_quality --candidate {repo}#{number}` to judge the "
                "current narrative -- --submit stays inert until it passes."
            )

    result = _finalize(
        store,
        repo,
        number,
        bundle,
        approve=args.approve,
        submit=args.submit,
        fork_owner=args.fork_owner,
        pr_drafts_dir=resolved_data_dir / _PR_DRAFTS_SUBDIR,
        pr_body=pr_body,
        quality_passed=quality_passed,
    )
    if result.submitted:
        print(f"\nOpened PR: {result.pr_url}" if result.pr_url else "\ngh pr create failed.")
    elif result.approved and args.submit and result.draft_is_new:
        print(f"\nWrote PR draft (first time): {result.draft_path}")
        print(
            "NOT submitted: this is the first time this candidate's draft was written, so "
            "there's been no chance to actually review it yet. Read the file, then re-run "
            "this same command (--approve --submit) again to open the real PR."
        )
    elif result.approved and args.submit and not result.quality_passed:
        print(f"\nWrote PR draft: {result.draft_path}")
        print(
            "NOT submitted: no passing T3.10 PR-quality verdict for the current narrative -- "
            "see the message above."
        )
    elif result.approved:
        print(f"\nWrote PR draft: {result.draft_path}")
        print("(pass --submit too, once you've reviewed it, to open a real PR)")
    else:
        print("\n(not approved -- pass --approve to write a PR draft)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
