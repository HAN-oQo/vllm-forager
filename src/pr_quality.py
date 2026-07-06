"""PR-quality gate (T3.10): "would a maintainer accept this?" — an adversarial ensemble scores
T3.9's composed PR narrative before the human-confirm step.

Part of M3.5, the same incident this milestone responds to: PR #47645 (withdrawn) wasn't just
opened against the wrong repo — its body was thin, generic, and made claims (a passing test) the
reader had to take on faith. T3.9 fixes the generation side (splicing repro/verify evidence in
verbatim, escaping the model's own prose against faking a section). This module is the
*independent check* on the result: like `self_review.py` asks "does this patch actually fix the
bug" with `N` adversarial critiques and a supermajority vote, this module asks "would a real
maintainer accept this PR narrative on sight" the same way, over the same voting mechanics
(`self_review.DEFAULT_VOTES`/`SUPERMAJORITY_THRESHOLD` are reused directly, not re-declared, per
this codebase's own "reuse the earlier sibling stage's decision knobs" convention — see
`gate.py`'s reuse of `scout._score`).

`run_pr_quality` always judges the candidate's **most recent** `stage="pr_author"` run, not
merely "some prior composed body" — `PRQualityResult.pr_author_recorded_at` records which one,
for KB traceability/display. The actual staleness *guard* is a separate check: `pr_author.py`
(T3.9) now records which verify run's diff it composed against
(`PRAuthorResult.verify_recorded_at`), and this module compares that against the *current*
verify run's `recorded_at` (`gate.verified_diff`'s second return value) before judging — a
candidate re-verified (e.g. a regression fix) after its narrative was composed is never judged
against a diff its own narrative doesn't actually describe; this module skips (`None`, logged)
instead, mirroring `self_review.py`'s identical "never substitute a stale pairing" rule for its
own `verify_recorded_at` check. If no `pr_author` run exists yet, or the candidate is no longer
gate-ready (`gate.verified_diff` returns `None` — e.g. a regression demoted the verify run),
this module also skips, the same per-item failure isolation every M3 stage applies.

This module deliberately calls `gate.verified_diff`, not `gate.assemble_bundle` — the latter
also computes a risk/effort/impact badge via a real LLM call (`gate._risk_badge`) that this
module has no use for (it never reads risk/effort/impact); `verified_diff` shares the exact same
gate-readiness check (`gate._readiness`) without paying for that extra call.

Each of the `n` independent judges sees the composed title/body, the target repo's contribution
profile (T3.8 — title convention, DCO requirement, CONTRIBUTING/template excerpts, reusing
`pr_author._profile_context` directly rather than a second copy), and the verified diff, and is
asked for a single yes/no ("would you accept this PR narrative as a maintainer, without seeing
anything else") plus a one-line reason — deliberately the same minimal `(acceptable, reason)`
shape as `self_review.Critique`'s `(looks_correct, reason)`, not a fixed checklist of named issue
categories: the DEVPLAN's own worked examples ("empty Repro block", "perf claim without
numbers", "raw diff dumped in body") are illustrative failure modes a judge might *name* in
`reason`, not a schema this module enforces structurally — enforcing a fixed taxonomy would make
the judge blind to a real quality problem that doesn't fit one of the pre-named boxes. The
prompt also explicitly tells judges that an unfilled Reproduction section / unticked checklist
item is a legitimate, honest state for a candidate with no captured repro run (`gate.py` never
requires one) — not a completeness defect to penalize, matching T3.9's own conditional-checklist
design (PR #68) rather than contradicting it.

Known limitations, not fixed here:
- Like every sibling ensemble stage, a single judge call failing (`llm.LLMError`, malformed
  reply) is excluded from the vote entirely, not counted as a reject; if every call fails,
  `total_votes == 0` and `passes` is `False` (fails safe, same rule `self_review.py` documents).
- The DEVPLAN's "sends it back to T3.9" isn't automated here — there is no M4 orchestrator yet to
  re-invoke `pr_author.run_pr_author` on a `passes=False` verdict (the same "not wired into the
  pipeline yet" gap T3.7/T3.9 already left for their own downstream steps). T3.10.5 wired
  `gate.py`'s `--submit` to require a passing `pr_quality` run for the *current* narrative (it
  reads this module's persisted KB record, not a live call — see `gate.py`'s own docstring), but
  nothing automatically re-invokes T3.9 on failure; a human still has to run `python -m
  src.pr_author` again after fixing whatever the judges flagged.
- Judges see `RepoProfile` context but this module doesn't itself re-derive
  `profile.title_pattern`/`requires_dco` compliance in code (unlike, say, a regex check) — it
  relies entirely on the judges noticing a mismatch, the same "advisory context, not a coded
  enforcement" tradeoff T3.9's own docstring makes for the same profile fields.
- The single `(acceptable, reason)` vote collapses all five judged axes (clarity, completeness,
  evidence, focused diff, convention match) into one boolean with a free-text reason — sufficient
  for a pass/fail gate, but not machine-parseable per-axis signal for a future automated
  re-routing to T3.9 (which would need to know *what* to fix, not just that something failed).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import gate
from .agents.forecaster import TS_FORMAT
from .pr_author import _DIFF_CONTEXT_CHARS, _profile_context, _template_sections
from .pr_profile import RepoProfile, get_profile
from .self_review import DEFAULT_VOTES, SUPERMAJORITY_THRESHOLD
from .stages import complete_or_none, latest_run, record_run_best_effort
from .store import resolve_store
from .store.base import Store

_VOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "acceptable": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["acceptable", "reason"],
}

# Matching self_review._critique_prompt's own title/body truncation convention -- a large,
# evidence-rich T3.9 body (verbatim repro/verify log tails) must not push the judge prompt past
# a provider's context/request-size limit, which would otherwise silently fail every judge call
# and read as a unanimous rejection (zero votes -> passes=False) rather than a size problem.
_TITLE_CONTEXT_CHARS = 500
_BODY_CONTEXT_CHARS = 12000


class PRQualityError(RuntimeError):
    """`n` (the number of judges to request) was less than 1."""


@dataclasses.dataclass(frozen=True)
class QualityVote:
    """One judge's verdict."""

    acceptable: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class PRQualityResult:
    """The ensemble's verdict on one composed PR narrative.

    `passes` is `total_votes > 0 and approve_count / total_votes >= threshold and not
    missing_template_headers` -- zero votes (every judge call failed) never passes, the same
    fail-safe rule `self_review.py` applies, and neither does a body missing one of the target
    repo's own required section headers (T3.10.7), regardless of how the ensemble voted: found
    by hand during T3.6's first real attempt that judges alone don't reliably catch this (a body
    using entirely different headers than `vllm-project/vllm`'s own template passed 5/5, twice).
    `pr_author_recorded_at` traces exactly which `stage="pr_author"` run was judged (see module
    docstring).
    """

    repo: str
    number: int
    votes: tuple[QualityVote, ...]
    approve_count: int
    total_votes: int
    missing_template_headers: tuple[str, ...]
    passes: bool
    pr_author_recorded_at: str
    recorded_at: str

    def to_run_record(self) -> dict:
        record = dataclasses.asdict(self)
        record["stage"] = "pr_quality"
        record["votes"] = [dataclasses.asdict(v) for v in self.votes]
        return record


def _missing_template_headers(body: str, profile: RepoProfile | None) -> tuple[str, ...]:
    """Which of `profile.pr_template`'s own purpose/test-plan/test-result headers (T3.10.7,
    `pr_author._template_sections`) are absent from `body` as a `## {header}` line -- empty if
    there's no template to check against (nothing this module can enforce), or every required
    header is present. A deterministic, code-level check rather than relying on the judges to
    notice (see this module's own docstring: they didn't, twice, on a real off-template body)."""
    sections_map = _template_sections(profile)
    if sections_map is None:
        return ()
    return tuple(
        header
        for header in (sections_map.purpose, sections_map.test_plan, sections_map.test_result)
        if f"## {header}" not in body
    )


def _quality_prompt(
    title: str, body: str, diff: str, profile: RepoProfile | None, required_headers: tuple[str, ...]
) -> str:
    headers_note = (
        f"This repo's own PR template requires these exact section headers: "
        f"{', '.join(required_headers)}. Check the body actually uses them (a body organized "
        f"under different section names is a real defect, not a style nit)."
        if required_headers
        else "This repo has no PR template with a detectable required section structure."
    )
    return (
        "You are a strict vLLM/ROCm maintainer deciding, at a glance, whether to accept the "
        "following pull request narrative -- not whether the underlying code is correct (assume "
        "it already passed review for that), but whether the PR *itself* reads as acceptance-"
        "ready. Judge on: clarity (is it understandable without extra context), completeness "
        "(problem/root-cause/fix/verification/limitations are all meaningfully filled in, not "
        "vague or boilerplate placeholders), claims-backed-by-evidence (any repro or performance "
        "claim cites something concrete -- an actual command, log line, or number -- not just an "
        "assertion), a focused diff (the change matches the stated problem, not an unrelated or "
        "overly broad edit), and whether the title/DCO/section headers follow the repo's own "
        f"observed conventions below. {headers_note} Note: a candidate can legitimately have no "
        "captured pre-fix reproduction (its Reproduction section says so explicitly, and its "
        "Checklist leaves that item unticked) -- this is an honest, allowed state, not a "
        "completeness defect; do not penalize it as vague or incomplete. Reply with `acceptable` "
        "(boolean) and `reason` (a short, specific explanation -- if rejecting, name the single "
        "biggest problem, e.g. 'the performance claim has no numbers', 'the diff touches "
        "unrelated files').\n\n"
        f"Title: {title[:_TITLE_CONTEXT_CHARS]}\n\nBody:\n{body[:_BODY_CONTEXT_CHARS]}\n\n"
        f"Diff:\n{diff[:_DIFF_CONTEXT_CHARS]}\n\n"
        f"Repo contribution norms:\n{_profile_context(profile)}"
    )


def _judge(
    title: str,
    body: str,
    diff: str,
    profile: RepoProfile | None,
    required_headers: tuple[str, ...] = (),
) -> QualityVote | None:
    """One independent judge's vote, or `None` if the call failed or the reply didn't shape into
    a usable vote (excluded from the tally, not counted as a reject -- see module docstring)."""
    reply = complete_or_none(
        _quality_prompt(title, body, diff, profile, required_headers),
        _VOTE_SCHEMA,
        stage="pr_quality",
        subject=title,
    )
    if reply is None:
        return None
    acceptable = reply.get("acceptable")
    reason = reply.get("reason")
    if not isinstance(acceptable, bool) or not isinstance(reason, str) or not reason.strip():
        return None
    return QualityVote(acceptable=acceptable, reason=reason.strip())


def run_pr_quality(
    store: Store,
    repo: str,
    number: int,
    *,
    n: int = DEFAULT_VOTES,
    threshold: float = SUPERMAJORITY_THRESHOLD,
    profile: RepoProfile | None = None,
    now: datetime | None = None,
) -> PRQualityResult | None:
    """Judge `repo`#`number`'s most recent composed PR narrative (`stage="pr_author"`) with `n`
    independent judges, record the ensemble's verdict to `store`, and return it.

    Returns `None` if there's no `pr_author` run for (`repo`, `number`) yet, the candidate is no
    longer gate-ready (`gate.verified_diff` returns `None`), or the `pr_author` run was composed
    against a verify run that's no longer the current one (see module docstring) -- all three
    skip, logged, not raise. `profile` defaults to `pr_profile.get_profile(repo)` when not given.

    Raises:
        PRQualityError: `n` is less than 1.
    """
    if n < 1:
        raise PRQualityError(f"n must be >= 1, got {n}")

    pr_author_run = latest_run(store.list_runs(repo=repo, number=number, stage="pr_author"))
    if pr_author_run is None:
        print(f"pr_quality: no pr_author run for {repo}#{number}", file=sys.stderr)
        return None

    verified = gate.verified_diff(store, repo, number)
    if verified is None:
        print(
            f"pr_quality: {repo}#{number} is not gate-ready (see stderr for why)", file=sys.stderr
        )
        return None
    diff, current_verify_recorded_at = verified

    composed_verify_recorded_at = pr_author_run.get("verify_recorded_at") or ""
    if composed_verify_recorded_at != current_verify_recorded_at:
        print(
            f"pr_quality: pr_author run for {repo}#{number} was composed against a different "
            "verify run than the current one -- skipping stale narrative/diff pairing",
            file=sys.stderr,
        )
        return None

    resolved_profile = profile if profile is not None else get_profile(repo)
    title = pr_author_run.get("title") or ""
    body = pr_author_run.get("body") or ""
    missing_headers = _missing_template_headers(body, resolved_profile)
    sections_map = _template_sections(resolved_profile)
    required_headers = (
        (sections_map.purpose, sections_map.test_plan, sections_map.test_result)
        if sections_map is not None
        else ()
    )

    votes = tuple(
        v
        for v in (_judge(title, body, diff, resolved_profile, required_headers) for _ in range(n))
        if v is not None
    )
    approve_count = sum(1 for v in votes if v.acceptable)
    total_votes = len(votes)
    passes = total_votes > 0 and (approve_count / total_votes) >= threshold and not missing_headers

    when = now or datetime.now(timezone.utc)
    result = PRQualityResult(
        repo=repo,
        number=number,
        votes=votes,
        approve_count=approve_count,
        total_votes=total_votes,
        missing_template_headers=missing_headers,
        passes=passes,
        pr_author_recorded_at=pr_author_run.get("recorded_at") or "",
        recorded_at=when.strftime(TS_FORMAT),
    )
    record_run_best_effort(
        store, result.to_run_record(), stage="pr_quality", repo=repo, number=number
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: judge `--candidate`'s most recently composed PR narrative and print the
    verdict. Exit code is `0` only when the ensemble passed -- so this is scriptable as a gate
    (`python -m src.pr_quality --candidate ... && python -m src.gate --candidate ... --submit`),
    though `gate.py` itself reads the persisted verdict directly rather than relying on this
    process's exit code (see `gate.py`'s own docstring)."""
    ap = argparse.ArgumentParser(
        prog="python -m src.pr_quality",
        description="Judge a gate-ready candidate's most recently composed PR narrative (T3.9).",
    )
    ap.add_argument(
        "--candidate", required=True, type=gate._parse_candidate, help="owner/repo#number"
    )
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    repo, number = args.candidate

    store, _ = resolve_store(args.data_dir)
    result = run_pr_quality(store, repo, number)
    if result is None:
        print(f"{repo}#{number} is not ready for quality judging (see stderr for why).")
        return 1

    verdict = "PASSED" if result.passes else "FAILED"
    print(f"{verdict}: {result.approve_count}/{result.total_votes} acceptable")
    if result.missing_template_headers:
        print(
            "  ❌ missing required section header(s) from the repo's own PR template: "
            + ", ".join(f"## {h}" for h in result.missing_template_headers)
        )
    for vote in result.votes:
        print(f"  - {'✅' if vote.acceptable else '❌'} {vote.reason}")
    return 0 if result.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
