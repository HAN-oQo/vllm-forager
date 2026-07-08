"""Orchestrator (T4.1): the conductor that turns individual agents into one always-on pipeline.

Every tick does two things, in order:

1. **Pin the active policy version** (:func:`~src.policy.get_active`) once, at the top of the
   tick — so every stage that runs *this* tick sees the same version, even if a concurrent
   ``policy_update`` proposal (T2.2) activates a new one mid-tick. (T4.1 only pins and reports
   this version; threading it into each stage's own scoring/prompting is each of those
   modules' own future integration, not done here.)
2. **Route to each stage strictly by its own cadence** — per the DEVPLAN spec:

   - **data plane** (collect) — a fixed interval, :data:`~src.config.COLLECT_INTERVAL_HOURS`
     (daily by default).
   - **intel plane** (classify + forecast + report) — weekly (:data:`_INTEL_INTERVAL_HOURS`).
   - **contribution plane** (scout → engineer → gate) — **triggered**, not cadence-based: it
     runs only when a :class:`Stage`'s own ``trigger`` callback says there's something worth
     acting on this tick, never on a fixed schedule (the DEVPLAN's own worked example: "skips
     contribution unless a candidate triggers it").

Real agent entry points (``collector.main``, ``analyze.main``, ...) are never imported and
called directly inside :func:`run_tick` — they're passed in as :class:`Stage`\\ s, the same
"explicit dependency, not a hidden global" shape this codebase already uses for `Store`. This
is what lets :mod:`tests.test_orchestrator` exercise cadence/trigger routing with fake clocks
and fake agents, with no real network/LLM call anywhere in the offline test suite.

Known limitations, not fixed here (later M4 todos):
- :func:`run_tick` (T4.5) now records a ``"started"`` event before a stage runs, and a stage's
  own `run` body can call :func:`emit_heartbeat` any number of times while it's still working —
  but none of :mod:`src.orchestrator`'s real stages (``_collect``/``_intel``/``_contribution``)
  actually call it: they're thin wrappers around CLI ``main()`` calls with no natural place to
  plug in progress reporting, the same disclosed gap as T4.4's ``items: None``. This is a real
  gap against CLAUDE.md's own Guardrails wording ("every stage emits started→heartbeat→
  finished/failed with its current step + intermediate output") — today only ``started`` and a
  terminal event are real for the production pipeline; the ``heartbeat``/intermediate-output
  part of that sentence is infrastructure-complete (:func:`emit_heartbeat`, exercised end to end
  by a fake stage in :mod:`tests.test_liveness`) but has zero real callers yet. Practically, this
  means :func:`~src.liveness.latest_status`'s staleness check can't distinguish "genuinely hung"
  from "just a slow real stage" for the actual production pipeline until that wiring lands.
  Wiring real heartbeats into ``collector``/``analyst``/``forecaster``/``scout`` (per-page,
  per-item, or per-substep) is a further refinement, not this todo's — and `stale_after_s` for
  the real pipeline should stay generous until then (T3.19's own precedent: a real ``intel``
  tick can run for hours).
- Every run-event timestamp this module writes (``"started"``, :func:`emit_heartbeat`'s
  ``"heartbeat"``, and the terminal ``"ok"``/``"failed"`` event) samples the real wall clock
  fresh, right when it's written — **not** `run_tick`'s own `now` parameter, which is used only
  for cadence math (`_is_due`) and the last-run cursor `_is_due` itself reads back. An earlier
  version of this code stamped the terminal event with the tick's `now` (captured once, before
  any stage ran) instead — for a multi-stage tick where an earlier stage takes real time to run
  (`_real_stages()`'s own collect → intel → contribution shape, with intel able to run for hours
  per T3.19), that made a later stage's terminal event record a timestamp *earlier* than that
  same stage's own `"started"` event: a real inversion, not just a tie, that made
  :func:`~src.liveness.latest_status` misclassify a stage that had already finished as
  `"running"`/`"stalled"`. Caught by this todo's own code review before it reached a demo; fixed
  by always sampling the real clock for every run-event timestamp, keeping `now` scoped strictly
  to scheduling decisions.
- ``main()`` takes a host-local, non-blocking lock (:func:`~src.locking.run_lock`, T4.3) around
  a tick, so two overlapping invocations never race — but it's process-level only, not KB-level
  idempotency for a *half-finished* tick's own writes (a stage that partially wrote before a
  crash isn't rolled back or resumed safely); see :mod:`src.locking`'s own docstring.
- The CLI's real "contribution" trigger (:func:`_has_open_rocm_or_good_first_issue`) is a
  cheap, label/keyword-only proxy for "is there something worth Scout's full (LLM-scoring)
  pass" — not a call into :func:`~src.agents.scout.discover_from_store` itself, which would
  defeat the point of a cheap trigger check by doing the expensive work just to decide whether
  to do the expensive work. A better trigger heuristic is a future refinement, not this todo's.
  It also only checks the two label/keyword-matched sources, not Scout's third (parity-gap)
  source, so a real parity-gap-only candidate with no matching open issue never triggers a run.
- :func:`run_tick` (T4.4) now records a ``status: "failed"`` run event for a stage that raises
  — but still re-raises afterward, so a failing stage still aborts the whole tick, including
  stages later in `stages` that don't depend on it (unlike every LLM-calling agent module in
  this codebase, e.g. :mod:`src.agents.analyst`/:mod:`src.agents.scout`, which isolate
  per-item failures and keep going). Full per-stage isolation — continuing to the next stage
  after recording a failure — is a further refinement, not this todo's; ``main()`` only catches
  this at the CLI boundary (see below), so a direct :func:`run_tick` caller still sees the
  exception, now alongside a KB record of what failed.
- Real stages' run events always carry ``items: None`` — ``collector.main``/``analyze.main``/
  ``forecast.main``/``report.main``/``candidates.main`` only *print* their counts, they don't
  return them, so getting a real count would mean bypassing each CLI's ``main()`` (and, for
  "contribution", re-implementing its console output to avoid losing it) rather than just
  wrapping it. `Stage.run`'s ``int | None`` return **is** wired all the way into the recorded
  run event — see :mod:`tests.test_events` — real stages just don't populate it yet.
- ``_real_stages()``'s real "intel"/"contribution" wrappers pass a fixed, conservative
  ``--per-domain-limit`` into ``analyze.main``/``candidates.main`` (see
  :data:`_INTEL_PER_DOMAIN_LIMIT`/:data:`_CONTRIBUTION_PER_DOMAIN_LIMIT`) rather than running
  either unbounded — T3.19 already hit the unbounded-cost failure mode this guards against (a
  54,841-item KB triggering a multi-hour, one-LLM-call-per-item classification pass). A fixed
  per-tick cap means a large backlog drains slowly across many ticks rather than all at once;
  a real backlog-draining strategy is a future refinement, not this todo's.
- Run events carry no run/invocation-correlation id and no tick-level envelope tying a tick's
  own events together — a hard-killed process (OOM, ``SIGKILL``) between two stages leaves the
  KB with events for only the stages that finished, with no marker that a further stage was
  queued and never even got a ``"failed"`` event. T4.5's ``started``/``heartbeat``/``finished``
  lifecycle will need a correlating id to stitch multiple records for one stage-invocation back
  together — not added here since nothing yet reads for it.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial

from . import config, cost
from .agents.scout import _candidate_source
from .locking import LockHeld, run_lock
from .policy import PolicyError, get_active
from .store import get_store
from .store.base import Store

# Intel plane (classify + forecast + report) cadence: weekly. Data plane's own cadence is
# config.COLLECT_INTERVAL_HOURS (already named/tunable there); this one is orchestrator-only
# so far, no other module reads it.
_INTEL_INTERVAL_HOURS = 24.0 * 7

# Bounds for the real "intel"/"contribution" stages (see module docstring's Known limitations).
# Same order of magnitude as T3.19's own dry-run values (docs/DEVPLAN.md T3.19 evidence).
_INTEL_PER_DOMAIN_LIMIT = 15
_CONTRIBUTION_PER_DOMAIN_LIMIT = 5

_LAST_RUN_KEY_PREFIX = "orchestrator:last_run:"
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class OrchestratorError(RuntimeError):
    """A misconfigured :class:`Stage`: neither a cadence nor a trigger to route by."""


@dataclass(frozen=True)
class Stage:
    """One pipeline stage the orchestrator can route a tick to.

    `name` keys this stage's last-run cursor in `Store` state — distinct stages must use
    distinct names, or they'll silently share (and clobber) one cursor.

    `interval_hours` is `None` for a **triggered** stage (contribution): it never runs on a
    cadence, only when `trigger(store)` returns `True` this tick. Exactly one of
    `interval_hours`/`trigger` must be set — :func:`run_tick` raises :class:`OrchestratorError`
    otherwise, rather than silently never running a misconfigured stage.

    `run` is the stage's actual work, an explicit callable — never imported and invoked
    directly inside this module — so a test can inject a fake stage without patching a real
    agent module's globals. Its return value (an item count, or `None` if not known/applicable)
    is folded into the run event :func:`run_tick` records for this stage (T4.4).
    """

    name: str
    run: Callable[[Store], int | None]
    interval_hours: float | None = None
    trigger: Callable[[Store], bool] | None = None

    def __post_init__(self) -> None:
        if self.interval_hours is None and self.trigger is None:
            raise OrchestratorError(
                f"stage {self.name!r} has neither interval_hours nor trigger to route by"
            )


@dataclass(frozen=True)
class TickResult:
    """Which stages ran this tick, and why the others didn't — returned so a caller (the CLI,
    a test) can assert on routing without re-deriving it from `Store` state."""

    policy_version: int
    ran: tuple[str, ...]
    skipped: dict[str, str]


def _last_run_key(stage_name: str) -> str:
    return f"{_LAST_RUN_KEY_PREFIX}{stage_name}"


def parse_last_run(raw: str) -> datetime | None:
    """Parse a stored last-run cursor, or `None` if it's not in the expected format.

    A malformed cursor (a hand-edited state file, a future format change, a partial write) is
    treated as "no cursor" by :func:`_is_due` rather than raised — the same recoverable-corrupt-
    timestamp convention as :func:`~src.pr_followup._parse_ts`, so one bad state value can't
    crash an otherwise-healthy tick.

    Public (not `_parse_last_run`) since this is also the one place a stored run-event
    timestamp gets parsed back for :mod:`src.liveness`/:mod:`dashboard.health` — a code-review
    finding that a private helper had quietly grown two external consumers reaching past its
    leading underscore, the same inconsistency the `_PLANES` -> `PLANES` promotion in
    :mod:`dashboard.pipeline_diagram` (T5.8) had just fixed one function away.
    """
    try:
        return datetime.strptime(raw, _TS_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _best_effort(write: Callable[[], None], *, what: str) -> None:
    """Call `write()`, logging (not raising) on failure -- a KB-write hiccup must never crash a
    tick after a stage's real work already happened. Used for both the run-event write and the
    last-run cursor write in :func:`run_tick`: recording the event but then crashing on the
    cursor write (or vice versa) would leave the KB's run history and its own cadence state
    disagreeing about whether the stage "really" completed this tick.

    Deliberately a small local helper rather than reusing `src.stages.record_run_best_effort`:
    that one requires `repo`/`number` (it's scoped to M3's per-item pipeline stages), which an
    orchestrator-level, whole-stage run event has neither of.
    """
    try:
        write()
    except Exception as exc:
        print(f"orchestrator: failed to {what}: {exc}", file=sys.stderr)


def emit_heartbeat(
    store: Store, *, stage: str, step: str, output_tail: str = "", policy_version: int | None = None
) -> None:
    """Record a `"heartbeat"` run event for `stage` (T4.5): its current step + a rolling tail
    of intermediate output, while it's still working. Never raises -- a heartbeat is inherently
    best-effort, the same as every other write `run_tick` makes.

    Callable directly by a stage's own `run` body -- it already receives `store` as its one
    argument, so no change to `Stage.run`'s signature was needed to support this. `run_tick`
    itself has no visibility into a stage's internal progress (`stage.run(store)` is one opaque
    call from its point of view); a stage that wants heartbeats has to emit them itself, at
    whatever points in its own work make sense.

    Uses the real wall clock (`datetime.now(timezone.utc)`), not a tick's own `now` (see
    :func:`run_tick`) -- staleness detection (:func:`~src.liveness.latest_status`) is about real
    elapsed time since the last heartbeat, which can be meaningfully later than when the tick
    itself started for a long-running stage.

    `policy_version` defaults to a fresh `get_active(store).version` lookup (two full state
    reads on `JsonlStore` -- `get_active` itself plus the version lookup it does internally) if
    not given explicitly; a caller that already knows the tick's own pinned version (`run_tick`
    computes it once at the top of every tick) can pass it directly to skip that lookup
    entirely -- nothing does yet, since `Stage.run` only receives `store`, the same reason T4.1
    never threaded policy into a stage's own scoring/prompting either.
    """
    try:
        version = policy_version if policy_version is not None else get_active(store).version
    except Exception as exc:
        # A PolicyError (no policy bootstrapped yet) or any other lookup failure must degrade
        # the same way a record_run failure does -- log and return, never raise, since a stage
        # calling this mid-work has already done real work that must not be lost over a
        # heartbeat's own bookkeeping failing.
        print(f"orchestrator: failed to record heartbeat for {stage!r}: {exc}", file=sys.stderr)
        return
    _best_effort(
        partial(
            store.record_run,
            {
                "stage": stage,
                "status": "heartbeat",
                "step": step,
                "output_tail": output_tail,
                "policy_version": version,
                "recorded_at": datetime.now(timezone.utc).strftime(_TS_FORMAT),
            },
        ),
        what=f"record heartbeat for {stage!r}",
    )


def _is_due(store: Store, stage: Stage, now: datetime) -> tuple[bool, str]:
    """Whether `stage` should run this tick, and a short reason either way."""
    if stage.interval_hours is None:
        assert stage.trigger is not None  # guaranteed by Stage.__post_init__
        return (True, "triggered") if stage.trigger(store) else (False, "no trigger")

    last = store.get_state(_last_run_key(stage.name))
    parsed = parse_last_run(last) if last is not None else None
    if parsed is None:
        return True, "never run"
    elapsed_hours = (now - parsed).total_seconds() / 3600
    if elapsed_hours >= stage.interval_hours:
        return True, "due"
    return False, f"not due for {stage.interval_hours - elapsed_hours:.1f}h"


def run_tick(store: Store, stages: list[Stage], *, now: datetime | None = None) -> TickResult:
    """One orchestrator tick: pin the active policy version, then run every `stages` entry
    that's due (a cadence stage) or triggered (a triggered stage) — in `stages`' own order.

    `now` defaults to the real UTC clock; tests pass a fixed value for determinism. A naive
    (tzinfo-less) `now` is treated as UTC rather than raising a `TypeError` once compared
    against a stored (always UTC-aware) cursor — the common `datetime.utcnow()` idiom is naive.
    `now` is used only for cadence math (`_is_due`) and the last-run cursor it reads back --
    every run-event timestamp below always samples the real wall clock instead (see the module
    docstring's Known limitations for why that distinction matters).

    Each stage that's due/triggered gets a `"started"` run event (T4.5) right before it runs,
    then exactly one terminal run event after it finishes (T4.4, `store.record_run`,
    best-effort): `{stage, status: "ok"|"failed", items, dur_s, policy_version, recorded_at}`
    (`error` too, on failure). A stage that raises still gets its `"failed"` event recorded
    before the exception propagates -- see the module docstring's Known limitations for why
    this doesn't (yet) isolate later stages from an earlier failure. A stage's own `run` body
    can additionally call :func:`emit_heartbeat` any number of times in between, to report
    progress on a long-running stage -- `run_tick` itself never calls it.

    Raises:
        PolicyError: no policy has been created yet — mirrors :func:`~src.policy.get_active`'s
            own contract; an orchestrator tick with nothing to pin isn't a state this function
            recovers from silently.
    """
    now = now if now is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    policy = get_active(store)  # pinned once; not re-fetched per stage within this tick

    ran: list[str] = []
    skipped: dict[str, str] = {}
    for stage in stages:
        due, reason = _is_due(store, stage, now)
        if not due:
            skipped[stage.name] = reason
            continue

        started = time.monotonic()
        # The tick's own `now` is used ONLY for the cadence cursor below -- `_is_due` compares
        # a future tick's `now` against this same value, so they must share one clock model
        # (real in production, fixed in a test). Every *run-event* timestamp in this loop
        # (`started`, `emit_heartbeat`, and the terminal event below) instead always samples
        # the real wall clock fresh, right when it's written -- staleness detection is about
        # real elapsed time, and stamping a stage's terminal event with the *tick's* `now`
        # (captured once, before any stage ran) used to make later stages in a multi-stage tick
        # record a terminal timestamp *earlier* than their own "started" event whenever an
        # earlier stage took real time to run (exactly `_real_stages()`'s own collect -> intel
        # -> contribution shape, with intel able to run for hours per T3.19) -- a real
        # production bug, caught by this PR's own review before it ever reached a demo, not
        # just a test artifact.
        cursor_at = now.strftime(_TS_FORMAT)
        _best_effort(
            partial(
                store.record_run,
                {
                    "stage": stage.name,
                    "status": "started",
                    "policy_version": policy.version,
                    "recorded_at": datetime.now(timezone.utc).strftime(_TS_FORMAT),
                },
            ),
            what=f"record started event for {stage.name!r}",
        )
        record = {
            "stage": stage.name,
            "policy_version": policy.version,
        }
        try:
            items = stage.run(store)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["dur_s"] = round(time.monotonic() - started, 3)
            record["recorded_at"] = datetime.now(timezone.utc).strftime(_TS_FORMAT)
            _best_effort(
                partial(store.record_run, record),
                what=f"record run event for {stage.name!r}",
            )
            raise

        record["status"] = "ok"
        record["items"] = items
        record["dur_s"] = round(time.monotonic() - started, 3)
        record["recorded_at"] = datetime.now(timezone.utc).strftime(_TS_FORMAT)
        _best_effort(
            partial(store.record_run, record),
            what=f"record run event for {stage.name!r}",
        )
        if stage.interval_hours is not None:
            _best_effort(
                partial(store.set_state, _last_run_key(stage.name), cursor_at),
                what=f"advance last-run cursor for {stage.name!r}",
            )
        ran.append(stage.name)

    return TickResult(policy_version=policy.version, ran=tuple(ran), skipped=skipped)


def _has_open_rocm_or_good_first_issue(store: Store) -> bool:
    """Cheap (no LLM) proxy trigger for the contribution stage: is there at least one open
    issue that would match one of Scout's two label/keyword-matched sources? See module
    docstring for why this checks cheaply rather than calling `scout.discover_from_store` just
    to decide whether to call it, and for the parity-gap blind spot this proxy doesn't cover.
    """
    return any(
        _candidate_source(item) is not None for item in store.query(type="issue", state="open")
    )


def _real_stages() -> list[Stage]:
    """The real pipeline wired up for `python -m src.orchestrator --once`."""
    from . import analyze, candidates, collector, forecast, report

    def _collect(_store: Store) -> None:
        collector.main([])

    def _intel(store: Store) -> None:
        # Bounded (see module docstring's Known limitations) -- an unbounded analyze.main([])
        # is exactly the T3.19 multi-hour/54,841-item hazard, reproduced on every "never run
        # yet" first tick against a KB with any sizeable pending backlog. T4.8: each sub-call
        # is wrapped in cost.agent_context so its llm.complete calls get a cost record tagged
        # by agent, sharing one run_id across all three (one "intel" tick invocation).
        run_id = uuid.uuid4().hex[:12]
        with cost.agent_context(store, agent="analyst", loop="intel", run_id=run_id):
            analyze.main(["--per-domain-limit", str(_INTEL_PER_DOMAIN_LIMIT)])
        with cost.agent_context(store, agent="forecaster", loop="intel", run_id=run_id):
            forecast.main([])
        with cost.agent_context(store, agent="reporter", loop="intel", run_id=run_id):
            report.main([])

    def _contribution(store: Store) -> None:
        # Scout only, for now — printing the ranked queue for a human to act on. Wiring the
        # rest of the contribution plane (engineer -> gate) into an orchestrated tick is a
        # later M4 todo, not this one. Bounded for the same reason as `_intel` above.
        run_id = uuid.uuid4().hex[:12]
        with cost.agent_context(store, agent="scout", loop="contribution", run_id=run_id):
            candidates.main(["--per-domain-limit", str(_CONTRIBUTION_PER_DOMAIN_LIMIT)])

    # Each real CLI (collector.main, analyze.main, ...) takes its own `argv`, not a `Store` --
    # it resolves its own store internally the same way this CLI's `main()` does. `Stage.run`'s
    # own `Store` parameter (see its docstring) is still unused by `_collect` for that reason;
    # `_intel`/`_contribution` now use theirs only to attribute T4.8 cost records to this tick
    # (`llm.cost_context`/`cost.record_cost`) -- the CLIs themselves still share state with this
    # tick via the KB, not a passed-in reference.
    return [
        Stage("collect", _collect, interval_hours=config.COLLECT_INTERVAL_HOURS),
        Stage("intel", _intel, interval_hours=_INTEL_INTERVAL_HOURS),
        Stage("contribution", _contribution, trigger=_has_open_rocm_or_good_first_issue),
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run one orchestrator tick against the real KB and real agents.

    No ``--data-dir`` override (unlike most CLIs in this repo): each real stage
    (``collector.main``, ``analyze.main``, ...) resolves its *own* store independently, and
    not every one of them accepts a ``--data-dir`` override (``collector.py`` doesn't) — so an
    override here could silently make the orchestrator's own routing decisions (cadence,
    policy) disagree with which KB the stages it runs actually touch. Always uses
    :func:`~src.store.get_store` (env ``STORE``/``FORAGER_DATA_DIR``), the same ambient
    configuration every stage already independently resolves to.
    """
    ap = argparse.ArgumentParser(
        prog="python -m src.orchestrator",
        description="Run one orchestrator tick: pin the active policy, route to due/triggered "
        "stages.",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Run a single tick and exit (the only mode T4.1 implements; always-on scheduling "
        "is T4.2's tmux/cron runbook, not a loop inside this process).",
    )
    args = ap.parse_args(argv)

    if not args.once:
        print(
            "orchestrator: only --once is supported so far (T4.2 adds scheduling)", file=sys.stderr
        )
        return 2

    store = get_store()
    try:
        with run_lock("orchestrator"):
            result = run_tick(store, _real_stages())
    except LockHeld as e:
        # Another tick is already mid-flight (a scheduled run overlapping a manual one, or a
        # slow run still going when the next cron fire happens, T4.3) -- back off cleanly
        # rather than racing it; no stage ever ran, so there's nothing to double-write.
        print(f"orchestrator: {e} -- skipping this tick", file=sys.stderr)
        return 0
    except PolicyError as e:
        # No policy has been bootstrapped yet (`policy.create_policy()` never called against
        # this KB) -- a clean, actionable CLI message instead of a raw traceback out of
        # policy.get_active(), since this is the always-on entry point T4.2 will run unattended.
        print(f"orchestrator: {e}", file=sys.stderr)
        return 1
    print(f"policy@{result.policy_version} — ran: {list(result.ran)}, skipped: {result.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
