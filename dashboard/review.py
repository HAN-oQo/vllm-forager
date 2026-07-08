"""Review pane (T5.6): the dashboard's approve/hold surface for the M3 human gate
(:mod:`src.gate`) -- diff + risk badge + MI250 logs in one place, and the "approve" / "hold"
buttons wired straight to the same gate that already exists.

This is one of only **two** write paths the whole dashboard has (per the M5 milestone's own
"Expected output" framing in ``docs/DEVPLAN.md``: "Read-only except two human decisions:
candidate selection (T5.10) and patch review (T5.6)") -- kept in its own module rather than
folded into :mod:`dashboard.api` (T5.1), whose own docstring commits to being strictly
read-only, so the one deliberately mutating surface stays visually and structurally separate
from every panel that isn't.

**`submit` is never exposed here, and is hardcoded `False`.** Opening a real PR against an
upstream repo is a categorically different, more dangerous action than "I approve this patch"
-- see :mod:`src.gate`'s own module docstring for the exact incident (a real, public draft PR
against ``vllm-project/vllm``, since withdrawn) that distinction exists to prevent. A human who
wants to actually submit a patch still has to run ``python -m src.gate --approve --submit``
from the CLI, a separate, deliberate step this dashboard action never triggers -- removing the
whole class of risk at this call site, on top of (not instead of) `src.gate._finalize`'s own
five-condition guard against an accidental submission. **SAFETY:** if a future change ever adds
a `submit` parameter to :func:`review_decision`, it must still default to `False` and never be
forwarded as anything but the literal `False` unless this module's whole safety framing (and
CLAUDE.md's own "fork-first, never spray public" rule) is revisited first -- this is the one
line in the whole dashboard that stands between a button click and a real, public PR.

:func:`review_bundle` costs exactly one real ``llm.complete`` call (:func:`~src.gate._risk_badge`,
the same scoring T2.5's Scout does) -- unlike :func:`~dashboard.api.candidates`'s own
``compute=False`` guard (built to stop a page load from silently re-scoring an entire *list* of
candidates), this reads exactly **one** named candidate a human has explicitly opened to review
right now, the same bounded, deliberate cost :mod:`src.gate`'s own CLI already pays every time a
human runs it -- not the "scored on every page view" hazard T4.8-T4.11 exist to cap.
:func:`review_decision` pays this same cost a second time (it re-assembles the bundle rather
than reusing whatever :func:`review_bundle` last returned) -- deliberate, not an oversight: the
gap between a human opening the review pane and clicking approve/hold is exactly the kind of
window `src.gate`'s own `draft_is_new` content check exists to guard against (a re-verified
candidate, or a re-composed T3.9 narrative, landing in between); deciding from a cached,
possibly-stale bundle would let a human approve content that's no longer what `store` actually
has, which no downstream check (`draft_is_new` compares against what's *on disk*, not against
"what changed since this page loaded") would catch.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src import gate
from src.store.base import Store

from .api import asdict_with


def review_bundle(store: Store, repo: str, number: int) -> dict | None:
    """The full evidence bundle for one gate-ready candidate -- diff, risk badge, repro/verify
    logs, self-review votes -- as a plain dict, or `None` if the candidate isn't gate-ready yet
    (mirrors :func:`~src.gate.assemble_bundle`'s own skip-not-raise contract: no verify run, the
    latest verify isn't verified, no matching self-review, or that self-review didn't advance).
    """
    bundle = gate.assemble_bundle(store, repo, number)
    return asdict(bundle) if bundle is not None else None


def _gate_result_dict(result: gate.GateResult) -> dict:
    """`result` as a plain dict, `draft_path` included as a `str` -- `asdict()` alone would
    leave it as a `Path` object, not JSON-serializable."""
    return asdict_with(result, draft_path=str(result.draft_path) if result.draft_path else None)


def review_decision(
    store: Store,
    repo: str,
    number: int,
    *,
    approve: bool,
    now: datetime | None = None,
    pr_drafts_dir: Path | None = None,
) -> dict | None:
    """Record a human's approve/hold decision for one gate-ready candidate, or `None` if the
    candidate isn't gate-ready (mirrors :func:`~src.gate.assemble_bundle`'s own contract).

    `approve=True` ("approve") writes a local PR-draft file -- never opens a real PR, `submit`
    is always `False` here (see this module's own docstring). `approve=False` ("hold") writes
    nothing but still records the decision as its own `stage="gate"` run, the same audit trail
    every other outcome gets -- a human's explicit "not yet" is as worth keeping evidence of as
    an approval (this codebase's own evidence principle), not silently discarded because
    nothing was written to disk for it.

    Mirrors :func:`~src.gate.main`'s own CLI flow, not just :func:`~src.gate.run_gate`'s bare
    defaults (a code-review finding, confirmed independently by two finder angles, on an
    earlier version of this function that went through `run_gate` directly):

    - Reads any existing composed T3.9 narrative + T3.10 quality verdict via
      :func:`~src.gate._current_narrative` (pure KB reads, no LLM call) before finalizing, so an
      approval through the dashboard uses the same maintainer-grade body a CLI approval would --
      the earlier version always fell back to the raw evidence-dump rendering, which a later CLI
      `--submit` would then reject as `draft_is_new` (a content mismatch it never actually
      caused, only revealed after the fact), silently defeating this todo's own "approve/hold a
      patch without leaving the dashboard" promise that a dashboard approval is a real step
      toward the eventual submitted document -- and, independently, silently overwrote an
      already-composed on-disk draft with the generic dump if one already existed.
    - Accepts `pr_drafts_dir` and forwards it as-is, rather than always falling through to
      `src.gate`'s own module-level default (`config.DATA_DIR`-based) regardless of what data
      dir `store` actually reads from. `src.gate.py`'s own module docstring already names this
      exact bug class ("a real bug caught by this module's own Demo run") for the CLI; a caller
      resolving a different `--data-dir` should pass the same directory `gate.main` itself
      resolves (`resolved_data_dir / gate._PR_DRAFTS_SUBDIR`) once a dashboard route threads
      `--data-dir` through this far (not yet wired -- no `dashboard/server.py` route calls this
      module at all yet, matching T5.1-T5.5's own "data layer first" precedent); omitting it
      keeps today's behavior (the CLI's own default) unchanged.
    """
    bundle = gate.assemble_bundle(store, repo, number)
    if bundle is None:
        return None
    pr_title, pr_body, quality_passed = gate._current_narrative(
        store, repo, number, current_verify_recorded_at=bundle.verify_recorded_at
    )
    result = gate._finalize(
        store,
        repo,
        number,
        bundle,
        approve=approve,
        submit=False,
        pr_drafts_dir=pr_drafts_dir,
        now=now,
        pr_body=pr_body,
        pr_title=pr_title,
        quality_passed=quality_passed,
    )
    return _gate_result_dict(result)
