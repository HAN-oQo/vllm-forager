"""Review pane (T5.6): the dashboard's approve/hold surface for the M3 human gate
(:mod:`src.gate`) -- diff + risk badge + MI250 logs in one place, and the "approve" / "hold"
buttons wired straight to the same gate that already exists.

This is one of only **two** write paths the whole dashboard has (CLAUDE.md: "Read-only except
two human decisions: candidate selection (T5.10) and patch review (T5.6)") -- kept in its own
module rather than folded into :mod:`dashboard.api` (T5.1), whose own docstring commits to
being strictly read-only, so the one deliberately mutating surface stays visually and
structurally separate from every panel that isn't.

**`submit` is never exposed here, and is hardcoded `False`.** Opening a real PR against an
upstream repo is a categorically different, more dangerous action than "I approve this patch"
-- see :mod:`src.gate`'s own module docstring for the exact incident (a real, public draft PR
against ``vllm-project/vllm``, since withdrawn) that distinction exists to prevent. A human who
wants to actually submit a patch still has to run ``python -m src.gate --approve --submit``
from the CLI, a separate, deliberate step this dashboard action never triggers -- removing the
whole class of risk at this call site, on top of (not instead of) `src.gate._finalize`'s own
five-condition guard against an accidental submission.

:func:`review_bundle` costs exactly one real ``llm.complete`` call (:func:`~src.gate._risk_badge`,
the same scoring T2.5's Scout does) -- unlike :func:`~dashboard.api.candidates`'s own
``compute=False`` guard (built to stop a page load from silently re-scoring an entire *list* of
candidates), this reads exactly **one** named candidate a human has explicitly opened to review
right now, the same bounded, deliberate cost :mod:`src.gate`'s own CLI already pays every time a
human runs it -- not the "scored on every page view" hazard T4.8-T4.11 exist to cap.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from src import gate
from src.store.base import Store


def review_bundle(store: Store, repo: str, number: int) -> dict | None:
    """The full evidence bundle for one gate-ready candidate -- diff, risk badge, repro/verify
    logs, self-review votes -- as a plain dict, or `None` if the candidate isn't gate-ready yet
    (mirrors :func:`~src.gate.assemble_bundle`'s own skip-not-raise contract: no verify run, the
    latest verify isn't verified, no matching self-review, or that self-review didn't advance).
    """
    bundle = gate.assemble_bundle(store, repo, number)
    return asdict(bundle) if bundle is not None else None


def review_decision(
    store: Store, repo: str, number: int, *, approve: bool, now: datetime | None = None
) -> dict | None:
    """Record a human's approve/hold decision for one gate-ready candidate, or `None` if the
    candidate isn't gate-ready (mirrors :func:`~src.gate.run_gate`'s own contract).

    `approve=True` ("approve") writes a local PR-draft file -- never opens a real PR, `submit`
    is always `False` here (see this module's own docstring). `approve=False` ("hold") writes
    nothing but still records the decision as its own `stage="gate"` run, the same audit trail
    every other outcome gets -- a human's explicit "not yet" is as worth keeping evidence of as
    an approval (this codebase's own evidence principle), not silently discarded because
    nothing was written to disk for it.
    """
    result = gate.run_gate(store, repo, number, approve=approve, submit=False, now=now)
    if result is None:
        return None
    return {**asdict(result), "draft_path": str(result.draft_path) if result.draft_path else None}
