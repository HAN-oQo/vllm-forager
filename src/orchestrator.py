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
- No run events (T4.4) or liveness heartbeats (T4.5) are written yet — :func:`run_tick` only
  returns an in-memory :class:`TickResult`; persisting a per-stage audit trail to the KB is
  T4.4/T4.5's job, layered onto this same routing skeleton.
- No overlap locking (T4.3) — nothing here stops two ticks from running concurrently.
- The CLI's real "contribution" trigger (:func:`_has_open_rocm_or_good_first_issue`) is a
  cheap, label/keyword-only proxy for "is there something worth Scout's full (LLM-scoring)
  pass" — not a call into :func:`~src.agents.scout.discover_from_store` itself, which would
  defeat the point of a cheap trigger check by doing the expensive work just to decide whether
  to do the expensive work. A better trigger heuristic is a future refinement, not this todo's.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from . import config
from .policy import get_active
from .store import get_store
from .store.base import Store

# Intel plane (classify + forecast + report) cadence: weekly. Data plane's own cadence is
# config.COLLECT_INTERVAL_HOURS (already named/tunable there); this one is orchestrator-only
# so far, no other module reads it.
_INTEL_INTERVAL_HOURS = 24.0 * 7

_LAST_RUN_KEY_PREFIX = "orchestrator:last_run:"


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
    agent module's globals.
    """

    name: str
    run: Callable[[Store], None]
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


def _parse_last_run(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _is_due(store: Store, stage: Stage, now: datetime) -> tuple[bool, str]:
    """Whether `stage` should run this tick, and a short reason either way."""
    if stage.interval_hours is None:
        assert stage.trigger is not None  # guaranteed by Stage.__post_init__
        return (True, "triggered") if stage.trigger(store) else (False, "no trigger")

    last = store.get_state(_last_run_key(stage.name))
    if last is None:
        return True, "never run"
    elapsed_hours = (now - _parse_last_run(last)).total_seconds() / 3600
    if elapsed_hours >= stage.interval_hours:
        return True, "due"
    return False, f"not due for {stage.interval_hours - elapsed_hours:.1f}h"


def run_tick(store: Store, stages: list[Stage], *, now: datetime | None = None) -> TickResult:
    """One orchestrator tick: pin the active policy version, then run every `stages` entry
    that's due (a cadence stage) or triggered (a triggered stage) — in `stages`' own order.

    `now` defaults to the real UTC clock; tests pass a fixed value for determinism.

    Raises:
        PolicyError: no policy has been created yet — mirrors :func:`~src.policy.get_active`'s
            own contract; an orchestrator tick with nothing to pin isn't a state this function
            recovers from silently.
    """
    now = now if now is not None else datetime.now(timezone.utc)
    policy = get_active(store)  # pinned once; not re-fetched per stage within this tick

    ran: list[str] = []
    skipped: dict[str, str] = {}
    for stage in stages:
        due, reason = _is_due(store, stage, now)
        if not due:
            skipped[stage.name] = reason
            continue
        stage.run(store)
        if stage.interval_hours is not None:
            store.set_state(_last_run_key(stage.name), now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        ran.append(stage.name)

    return TickResult(policy_version=policy.version, ran=tuple(ran), skipped=skipped)


def _has_open_rocm_or_good_first_issue(store: Store) -> bool:
    """Cheap (no LLM) proxy trigger for the contribution stage: is there at least one open
    issue that would match one of Scout's two GitHub-signal sources? See module docstring for
    why this checks cheaply rather than calling `scout.discover_from_store` just to decide
    whether to call it.
    """
    from .agents.scout import _candidate_source  # local import: avoids a module-load cycle

    return any(
        item.get("type") == "issue"
        and item.get("state") == "open"
        and _candidate_source(item) is not None
        for item in store.query()
    )


def _real_stages() -> list[Stage]:
    """The real pipeline wired up for `python -m src.orchestrator --once`."""
    from . import analyze, candidates, collector, forecast, report

    def _collect(_store: Store) -> None:
        collector.main([])

    def _intel(_store: Store) -> None:
        analyze.main([])
        forecast.main([])
        report.main([])

    def _contribution(_store: Store) -> None:
        # Scout only, for now — printing the ranked queue for a human to act on. Wiring the
        # rest of the contribution plane (engineer -> gate) into an orchestrated tick is a
        # later M4 todo, not this one.
        candidates.main([])

    # Each real CLI (collector.main, analyze.main, ...) takes its own `argv`, not a `Store` --
    # it resolves its own store internally the same way this CLI's `main()` does. The `_store`
    # parameter every wrapper above ignores is `Stage.run`'s own contract (see its docstring),
    # not something these CLIs need — they share state with this tick via the KB, not a
    # passed-in reference.
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
    result = run_tick(store, _real_stages())
    print(f"policy@{result.policy_version} — ran: {list(result.ran)}, skipped: {result.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
