"""Repro harness (T3.2): reproduce a candidate's issue on MI250, capture the failing signal.

Given a candidate — an issue already in the KB describing a ROCm bug — this module asks an LLM
to synthesize a shell command that attempts to reproduce it, runs that command on an MI250 host
via :func:`~src.runner.run`, and records the outcome as a run in the KB's `runs` collection
(:meth:`~src.store.base.Store.record_run`, T3.2's own reason for that collection existing).

This is the "before" half of T3.3's fail→patch→pass proof: a candidate with no baseline failing
signal has nothing for a later patch to flip. `ReproResult.reproduced` (`exit_code != 0`) is
that signal — the DEVPLAN's own framing ("capture the failing assertion/log as the baseline
signal") treats a non-zero exit as the evidence, not a judgment about whether the *specific*
assertion matches the reported bug; that finer read is left to a human/T3.4's ensemble review,
not this module.

Neither "no KB record for this candidate" nor "the LLM couldn't synthesize a usable repro
command" raises — both skip and return `None`, the by-now established per-item failure isolation
every sibling agent (grader/curator/scout/novelty) applies: one candidate this module can't
handle shouldn't abort a caller iterating over many. A :class:`~src.runner.RunnerError` from
:func:`~src.runner.run` itself is a different kind of failure (ssh unreachable, a genuine hang)
and is deliberately **not** caught here — an infra problem isn't "no signal to record," it's
something the caller needs to see and retry, not silently misread as "candidate doesn't
reproduce."

Known limitation, not fixed here: the synthesized repro command is exactly as good as the LLM's
read of the issue body — it may run something that doesn't actually exercise the reported bug
at all (a `reproduced=True` from an unrelated failure, or `reproduced=False` from a command that
never touched the real code path). Nothing in this first cut verifies the command's relevance;
that's a human/T3.4 concern downstream, not something this harness itself can judge from the
issue text alone.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from . import llm, runner
from .agents.forecaster import TS_FORMAT
from .store.base import Store

_REPRO_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}},
    "required": ["command"],
}


@dataclass(frozen=True)
class ReproResult:
    """One repro attempt's outcome — the same fields :meth:`to_run_record` shapes for
    :meth:`~src.store.base.Store.record_run`.

    `reproduced` is `exit_code != 0` (see module docstring for what that does and doesn't
    prove).
    """

    repo: str
    number: int
    host: str
    command: str
    exit_code: int
    log: str
    reproduced: bool
    recorded_at: str

    def to_run_record(self) -> dict:
        """This result as a plain dict, shaped for :meth:`~src.store.base.Store.record_run`."""
        return {
            "repo": self.repo,
            "number": self.number,
            "stage": "repro",
            "host": self.host,
            "command": self.command,
            "exit_code": self.exit_code,
            "log": self.log,
            "reproduced": self.reproduced,
            "recorded_at": self.recorded_at,
        }


def _repro_prompt(title: str, body: str) -> str:
    return (
        "This is a vLLM/ROCm GitHub issue describing a bug. Write ONE shell command that "
        "attempts to reproduce it on a ROCm (gfx90a) machine with vLLM already checked out in "
        "the current directory — e.g. a minimal pytest/python invocation exercising the "
        "described failure. Reply with `command`.\n\n"
        f"Title: {title}\n\nBody: {(body or '')[:4000]}"
    )


def synthesize_repro_command(title: str, body: str) -> str | None:
    """A shell command an LLM believes reproduces `title`/`body`'s bug, or `None` if the call
    failed or the reply didn't shape into a usable command (see module docstring: skip,
    logged, not raised)."""
    try:
        reply = llm.complete(_repro_prompt(title, body), json_schema=_REPRO_SCHEMA)
    except llm.LLMError as exc:
        print(f"repro: command synthesis failed for {title!r}: {exc}", file=sys.stderr)
        return None
    if not isinstance(reply, dict):
        return None
    command = reply.get("command")
    return command.strip() if isinstance(command, str) and command.strip() else None


def run_repro(
    store: Store,
    repo: str,
    number: int,
    host: str,
    *,
    now: datetime | None = None,
) -> ReproResult | None:
    """Synthesize and run a repro command for (`repo`, `number`) on `host`, record the result
    to `store`, and return it.

    Returns:
        `None` if `store` has no record for (`repo`, `number`), or no repro command could be
        synthesized (see module docstring — both cases are skipped, not raised).

    Raises:
        runner.RunnerError: propagated uncaught from :func:`~src.runner.run` — see module
            docstring for why an infra failure isn't treated the same as "no signal."
    """
    item = store.get_item(repo, number)
    if item is None:
        print(f"repro: no KB record for {repo}#{number}", file=sys.stderr)
        return None

    command = synthesize_repro_command(item.get("title") or "", item.get("body") or "")
    if command is None:
        return None

    result = runner.run(host, command)
    when = now or datetime.now(timezone.utc)
    repro_result = ReproResult(
        repo=repo,
        number=number,
        host=host,
        command=command,
        exit_code=result.exit_code,
        log=result.log,
        reproduced=result.exit_code != 0,
        recorded_at=when.strftime(TS_FORMAT),
    )
    store.record_run(repro_result.to_run_record())
    return repro_result
