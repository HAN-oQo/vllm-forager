"""Ensemble self-review gate (T3.4): N independent adversarial critiques of a verified patch,
before the human gate. *(Borrowed from The AI Scientist's ensemble reviewer.)*

A passing MI250 test can still hide a bad patch — reward-hacking (a fix that makes the specific
repro command pass without addressing the reported bug), an unrelated side effect, or a
change too narrow/too broad to be a real fix. T3.3's `verified=True` only proves "the failing
signal flipped"; it says nothing about whether the patch is *actually correct*. This module asks
an LLM to critique the same patch `N` times, independently, and only advances the candidate to
T3.5's human gate on a **supermajority** "looks correct" — the DEVPLAN's own worked example (5
critiques, 4 approve → advance; a 3–2 split → hold) is *not* a plain >50% majority (3/5 is
already a majority) but a real supermajority threshold, :data:`SUPERMAJORITY_THRESHOLD` (2/3):
4/5 ≈ 0.80 clears it, 3/5 = 0.60 doesn't. It's exposed as `run_self_review`'s own `threshold`
parameter (default `SUPERMAJORITY_THRESHOLD`), matching the same override-the-module-constant
convention :func:`~src.novelty.check_novelty`/:func:`~src.agents.grader.compute_metrics` already
use for their own decision thresholds.

Like :mod:`src.engineer`, "no verified T3.3 run for this candidate" and "no KB item record"
both skip (`None`, logged) — the same per-item failure isolation every sibling agent applies.
A single critique call failing (`llm.LLMError`, malformed reply) doesn't abort the whole
ensemble either: it's excluded from the vote entirely (neither an approve nor a reject), so one
flaky call doesn't silently tip a real 4-1 into a recorded 3-1 that reads as a rejection instead
of a missing data point. If *every* critique fails, `total_votes == 0` and the candidate holds
(fails safe: no votes cast is never treated as "unanimous approval").

Self-review always targets the candidate's **most recent** T3.3 run, not merely "some prior
`verified=True` run" — a candidate re-verified after a regression (a newer run with
`verified=False`) must not have its self-review computed against an older, now-superseded
success; that would record an `advance` verdict for a patch state that's no longer current. If
the most recent run isn't `verified=True`, this module skips (there is nothing to self-review
yet) rather than falling back to an older success. `SelfReviewResult.verify_recorded_at` records
exactly which T3.3 run's `patch`/`log` were reviewed, mirroring
:attr:`~src.engineer.EngineerResult.baseline_recorded_at`'s identical traceability purpose — so
T3.5 assembling an evidence bundle can confirm the self-review it's showing actually corresponds
to the verify attempt (patch + MI250 logs) it's showing alongside it, not a stale one from a
retried candidate.
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timezone

from .agents.forecaster import TS_FORMAT
from .stages import complete_or_none, get_item_or_skip, latest_run, record_run_best_effort
from .store.base import Store

_CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "looks_correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["looks_correct", "reason"],
}

DEFAULT_VOTES = 5

# A proportional supermajority, not a plain >50% majority -- see module docstring for why the
# DEVPLAN's own worked example (4/5 advances, 3/5 holds) requires this rather than a bare
# majority (3/5 = 0.6 is already a majority by the plain definition).
SUPERMAJORITY_THRESHOLD = 2 / 3

# Tail of a verify run's log shown to each critique -- the pass/fail summary a critic actually
# needs is conventionally at the *end* of test output, matching every other log-truncation
# convention in this codebase (runner.py's own ssh-255 message, Store.MAX_RUN_LOG_CHARS).
_LOG_CONTEXT_CHARS = 2000


class SelfReviewError(RuntimeError):
    """`n` (the number of critiques to request) was less than 1."""


@dataclasses.dataclass(frozen=True)
class Critique:
    """One adversarial critique's vote."""

    looks_correct: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class SelfReviewResult:
    """The ensemble's verdict on one verified patch.

    `advance` is `total_votes > 0 and approve_count / total_votes >= threshold` — zero votes
    (every critique call failed) never advances (see module docstring: fails safe).
    `verify_recorded_at` traces exactly which T3.3 run these critiques reviewed (see module
    docstring).
    """

    repo: str
    number: int
    critiques: tuple[Critique, ...]
    approve_count: int
    total_votes: int
    advance: bool
    verify_recorded_at: str
    recorded_at: str

    def to_run_record(self) -> dict:
        """This result as a plain dict, shaped for :meth:`~src.store.base.Store.record_run`."""
        record = dataclasses.asdict(self)
        record["stage"] = "self_review"
        record["critiques"] = [dataclasses.asdict(c) for c in self.critiques]
        return record


def _critique_prompt(title: str, body: str, patch: str, log: str) -> str:
    return (
        "A patch was applied to fix a vLLM/ROCm bug and the previously-failing test now "
        "passes. Adversarially review whether the patch actually, correctly fixes the "
        "*reported* bug — watch for reward-hacking (a change that only makes this specific "
        "test pass without addressing the real issue), unrelated side effects, or a fix too "
        "narrow/broad in scope. Reply with `looks_correct` (boolean) and `reason` (a short "
        "explanation).\n\n"
        f"Title: {title[:500]}\n\nBody: {(body or '')[:4000]}\n\n"
        f"Patch:\n{patch[:4000]}\n\n"
        f"Test output after the patch:\n{(log or '')[-_LOG_CONTEXT_CHARS:]}"
    )


def _critique(title: str, body: str, patch: str, log: str) -> Critique | None:
    """One independent adversarial critique, or `None` if the call failed or the reply didn't
    shape into a usable vote (see module docstring: excluded from the tally, not a reject)."""
    reply = complete_or_none(
        _critique_prompt(title, body, patch, log),
        _CRITIQUE_SCHEMA,
        stage="self_review",
        subject=title,
    )
    if reply is None:
        return None
    looks_correct = reply.get("looks_correct")
    reason = reply.get("reason")
    if not isinstance(looks_correct, bool) or not isinstance(reason, str) or not reason.strip():
        return None
    return Critique(looks_correct=looks_correct, reason=reason.strip())


def run_self_review(
    store: Store,
    repo: str,
    number: int,
    *,
    n: int = DEFAULT_VOTES,
    threshold: float = SUPERMAJORITY_THRESHOLD,
    now: datetime | None = None,
) -> SelfReviewResult | None:
    """Run `n` independent adversarial critiques of the candidate's most recent T3.3 run for
    (`repo`, `number`), record the ensemble's verdict to `store`, and return it.

    Returns:
        `None` if there's no T3.3 run for (`repo`, `number`) at all, the most recent one isn't
        `verified=True` (see module docstring — a stale older success is never substituted), or
        `store` has no item record for it (all three skip, not raise).

    Raises:
        SelfReviewError: `n` is less than 1.
    """
    if n < 1:
        raise SelfReviewError(f"n must be >= 1, got {n}")

    verify_run = latest_run(store.list_runs(repo=repo, number=number, stage="verify"))
    if verify_run is None:
        print(f"self_review: no verify run for {repo}#{number}", file=sys.stderr)
        return None
    if not verify_run.get("verified"):
        print(
            f"self_review: most recent verify run for {repo}#{number} is not verified",
            file=sys.stderr,
        )
        return None

    item = get_item_or_skip(store, repo, number, stage="self_review")
    if item is None:
        return None

    patch = verify_run.get("patch") or ""
    log = verify_run.get("log") or ""
    title = item.get("title") or ""
    body = item.get("body") or ""

    critiques = tuple(
        c for c in (_critique(title, body, patch, log) for _ in range(n)) if c is not None
    )
    approve_count = sum(1 for c in critiques if c.looks_correct)
    total_votes = len(critiques)
    advance = total_votes > 0 and (approve_count / total_votes) >= threshold

    when = now or datetime.now(timezone.utc)
    result = SelfReviewResult(
        repo=repo,
        number=number,
        critiques=critiques,
        approve_count=approve_count,
        total_votes=total_votes,
        advance=advance,
        verify_recorded_at=verify_run.get("recorded_at") or "",
        recorded_at=when.strftime(TS_FORMAT),
    )
    record_run_best_effort(
        store, result.to_run_record(), stage="self_review", repo=repo, number=number
    )
    return result
