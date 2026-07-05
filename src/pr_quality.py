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
merely "some prior composed body" — `PRQualityResult.pr_author_recorded_at` records exactly
which one, mirroring `self_review.SelfReviewResult.verify_recorded_at`'s identical purpose: a
candidate re-composed after this module already passed it once must not have that older verdict
silently read as still current. If no `pr_author` run exists yet, or the candidate is no longer
gate-ready (`gate.assemble_bundle` returns `None` — e.g. a regression demoted the verify run),
this module skips (`None`, logged), the same per-item failure isolation every M3 stage applies.

Each of the `n` independent judges sees the composed title/body, the target repo's contribution
profile (T3.8 — title convention, DCO requirement, CONTRIBUTING/template excerpts), and the
verified diff (T3.5's evidence bundle), and is asked for a single yes/no ("would you accept this
PR narrative as a maintainer, without seeing anything else") plus a one-line reason — deliberately
the same minimal `(acceptable, reason)` shape as `self_review.Critique`'s `(looks_correct,
reason)`, not a fixed checklist of named issue categories: the DEVPLAN's own worked examples
("empty Repro block", "perf claim without numbers", "raw diff dumped in body") are illustrative
failure modes a judge might *name* in `reason`, not a schema this module enforces structurally —
enforcing a fixed taxonomy would make the judge blind to a real quality problem that doesn't fit
one of the pre-named boxes.

Known limitations, not fixed here:
- Like every sibling ensemble stage, a single judge call failing (`llm.LLMError`, malformed
  reply) is excluded from the vote entirely, not counted as a reject; if every call fails,
  `total_votes == 0` and `passes` is `False` (fails safe, same rule `self_review.py` documents).
- The DEVPLAN's "sends it back to T3.9" isn't automated here — there is no M4 orchestrator yet to
  re-invoke `pr_author.run_pr_author` on a `passes=False` verdict (the same "not wired into the
  pipeline yet" gap T3.7/T3.9 already left for their own downstream steps). `gate.py`'s
  approve/submit path also doesn't yet require a passing `pr_quality` run before proceeding —
  wiring that gate is left to M4's orchestrator too.
- Judges see `RepoProfile` context but this module doesn't itself re-derive
  `profile.title_pattern`/`requires_dco` compliance in code (unlike, say, a regex check) — it
  relies entirely on the judges noticing a mismatch, the same "advisory context, not a coded
  enforcement" tradeoff T3.9's own docstring makes for the same profile fields.
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timezone

from . import gate
from .agents.forecaster import TS_FORMAT
from .pr_profile import RepoProfile, get_profile
from .self_review import DEFAULT_VOTES, SUPERMAJORITY_THRESHOLD
from .stages import complete_or_none, latest_run, record_run_best_effort
from .store.base import Store

_VOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "acceptable": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["acceptable", "reason"],
}

# Truncation window matching pr_author.py's own _DIFF_CONTEXT_CHARS -- a judge needs enough of
# the diff to assess "focused" scope, not the full patch.
_DIFF_CONTEXT_CHARS = 4000
_PROFILE_EXCERPT_CHARS = 1000


@dataclasses.dataclass(frozen=True)
class QualityVote:
    """One judge's verdict."""

    acceptable: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class PRQualityResult:
    """The ensemble's verdict on one composed PR narrative.

    `passes` is `total_votes > 0 and approve_count / total_votes >= threshold` -- zero votes
    (every judge call failed) never passes, the same fail-safe rule `self_review.py` applies.
    `pr_author_recorded_at` traces exactly which `stage="pr_author"` run was judged (see module
    docstring).
    """

    repo: str
    number: int
    votes: tuple[QualityVote, ...]
    approve_count: int
    total_votes: int
    passes: bool
    pr_author_recorded_at: str
    recorded_at: str

    def to_run_record(self) -> dict:
        record = dataclasses.asdict(self)
        record["stage"] = "pr_quality"
        record["votes"] = [dataclasses.asdict(v) for v in self.votes]
        return record


def _profile_context(profile: RepoProfile | None) -> str:
    """Same rendering as `pr_author._profile_context` -- a judge needs the identical contribution
    norms T3.9's composer was given, to catch a mismatch either missed."""
    if profile is None:
        return "No contribution-norms profile is available for this repo."
    lines = [
        f"DCO/Signed-off-by required: {profile.requires_dco}",
        f"Observed title convention: {profile.title_pattern or '(no consistent pattern detected)'}",
    ]
    if profile.title_examples:
        lines.append("Recent merged PR titles: " + "; ".join(profile.title_examples))
    if profile.contributing:
        lines.append(f"CONTRIBUTING excerpt: {profile.contributing[:_PROFILE_EXCERPT_CHARS]}")
    if profile.pr_template:
        lines.append(f"PR template excerpt: {profile.pr_template[:_PROFILE_EXCERPT_CHARS]}")
    return "\n".join(lines)


def _quality_prompt(title: str, body: str, diff: str, profile: RepoProfile | None) -> str:
    return (
        "You are a strict vLLM/ROCm maintainer deciding, at a glance, whether to accept the "
        "following pull request narrative -- not whether the underlying code is correct (assume "
        "it already passed review for that), but whether the PR *itself* reads as acceptance-"
        "ready. Judge on: clarity (is it understandable without extra context), completeness "
        "(problem/root-cause/fix/repro/verification/limitations are all meaningfully filled in, "
        "not vague or boilerplate placeholders), claims-backed-by-evidence (any repro or "
        "performance claim cites something concrete -- an actual command, log line, or number -- "
        "not just an assertion), a focused diff (the change matches the stated problem, not an "
        "unrelated or overly broad edit), and whether the title/DCO follow the repo's own "
        "observed conventions below. Reply with `acceptable` (boolean) and `reason` (a short, "
        "specific explanation -- if rejecting, name the single biggest problem, e.g. 'the Repro "
        "section has no captured output', 'the performance claim has no numbers', 'the diff "
        "touches unrelated files').\n\n"
        f"Title: {title}\n\nBody:\n{body}\n\n"
        f"Diff:\n{diff[:_DIFF_CONTEXT_CHARS]}\n\n"
        f"Repo contribution norms:\n{_profile_context(profile)}"
    )


def _judge(title: str, body: str, diff: str, profile: RepoProfile | None) -> QualityVote | None:
    """One independent judge's vote, or `None` if the call failed or the reply didn't shape into
    a usable vote (excluded from the tally, not counted as a reject -- see module docstring)."""
    reply = complete_or_none(
        _quality_prompt(title, body, diff, profile),
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

    Returns `None` if there's no `pr_author` run for (`repo`, `number`) yet, or the candidate is
    no longer gate-ready (`gate.assemble_bundle` returns `None`) -- both skip, logged, not raise.
    `profile` defaults to `pr_profile.get_profile(repo)` when not given.
    """
    pr_author_run = latest_run(store.list_runs(repo=repo, number=number, stage="pr_author"))
    if pr_author_run is None:
        print(f"pr_quality: no pr_author run for {repo}#{number}", file=sys.stderr)
        return None

    bundle = gate.assemble_bundle(store, repo, number)
    if bundle is None:
        return None

    resolved_profile = profile if profile is not None else get_profile(repo)
    title = pr_author_run.get("title") or ""
    body = pr_author_run.get("body") or ""

    votes = tuple(
        v
        for v in (_judge(title, body, bundle.diff, resolved_profile) for _ in range(n))
        if v is not None
    )
    approve_count = sum(1 for v in votes if v.acceptable)
    total_votes = len(votes)
    passes = total_votes > 0 and (approve_count / total_votes) >= threshold

    when = now or datetime.now(timezone.utc)
    result = PRQualityResult(
        repo=repo,
        number=number,
        votes=votes,
        approve_count=approve_count,
        total_votes=total_votes,
        passes=passes,
        pr_author_recorded_at=pr_author_run.get("recorded_at") or "",
        recorded_at=when.strftime(TS_FORMAT),
    )
    record_run_best_effort(
        store, result.to_run_record(), stage="pr_quality", repo=repo, number=number
    )
    return result
