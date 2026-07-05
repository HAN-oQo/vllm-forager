"""Repro harness (T3.2): reproduce a candidate's issue on MI250, capture the failing signal.

Given a candidate — an issue already in the KB describing a ROCm bug — this module asks an LLM
to synthesize a shell command that attempts to reproduce it, runs that command on an MI250 host
via :func:`~src.runner.run`, and records the outcome as a run in the KB's `runs` collection
(:meth:`~src.store.base.Store.record_run`, T3.2's own reason for that collection existing).

**Security note — read before wiring this to a real MI250 host:** :func:`synthesize_repro_command`
turns *untrusted, externally-authored* GitHub issue text into a shell command that
:func:`run_repro` then executes verbatim on real hardware, unreviewed — exactly the risk
``runner.run``'s own docstring warns against ("callers must not compose it from untrusted
external input without their own escaping"). This module does that anyway, because reproducing
a *reported* bug inherently means running something derived from what a reporter wrote — there
is no way to do T3.2's job at all without it. Nothing here sandboxes, allowlists, or reviews the
synthesized command before it runs; a crafted issue body (prompt-injection style) could steer the
LLM into emitting a destructive command. Treat any host `run_repro` is pointed at as disposable,
non-production hardware, and do not point it at anything else, until a real sandboxing/review
layer exists — that's future work, not something this first cut mitigates.

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
reproduce." A `store.record_run` failure, by contrast, IS caught — the repro already happened on
real hardware by that point; losing the in-memory :class:`ReproResult` too, on top of a mere
persistence hiccup, would force redoing the (possibly expensive) MI250 run just to recover an
outcome this process already has.

Known limitation, not fixed here: the synthesized repro command is exactly as good as the LLM's
read of the issue body — it may run something that doesn't actually exercise the reported bug
at all (a `reproduced=True` from an unrelated failure, or `reproduced=False` from a command that
never touched the real code path). Nothing in this first cut verifies the command's relevance;
that's a human/T3.4 concern downstream, not something this harness itself can judge from the
issue text alone.
"""

from __future__ import annotations

import dataclasses
import shlex
from datetime import datetime, timezone

from . import runner
from .agents.forecaster import TS_FORMAT
from .stages import complete_or_none, get_item_or_skip, record_run_best_effort
from .store.base import MAX_RUN_LOG_CHARS, Store

_REPRO_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}},
    "required": ["command"],
}

# A repro is meant to be a quick reproduction check, not a full ROCm build -- much shorter than
# runner.DEFAULT_TIMEOUT_S (an hour, sized for that build). Still overridable per call.
DEFAULT_REPRO_TIMEOUT_S = 600.0


@dataclasses.dataclass(frozen=True)
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
        """This result as a plain dict, shaped for :meth:`~src.store.base.Store.record_run` —
        `log` is truncated to its last :data:`~src.store.base.MAX_RUN_LOG_CHARS` characters for
        the persisted copy only (see that constant's own docstring); this `ReproResult` itself
        keeps the full text."""
        record = dataclasses.asdict(self)
        record["stage"] = "repro"
        record["log"] = record["log"][-MAX_RUN_LOG_CHARS:]
        return record


def _repro_prompt(title: str, body: str, repo_dir: str | None = None) -> str:
    cwd_hint = (
        f"Your command will already be run with `{repo_dir}` (vLLM's checkout) as its working "
        "directory."
        if repo_dir
        else "vLLM is already checked out in the current directory."
    )
    return (
        "This is a vLLM/ROCm GitHub issue describing a bug. Write ONE shell command that "
        f"attempts to reproduce it on a ROCm (gfx90a) machine — {cwd_hint} A minimal "
        "pytest/python invocation exercising the described failure is enough. Reply with "
        "`command`.\n\n"
        f"Title: {title[:500]}\n\nBody: {(body or '')[:4000]}"
    )


def synthesize_repro_command(title: str, body: str, repo_dir: str | None = None) -> str | None:
    """A shell command an LLM believes reproduces `title`/`body`'s bug, or `None` if the call
    failed or the reply didn't shape into a usable command (see module docstring: skip,
    logged, not raised).

    `repo_dir`, if given, is mentioned to the LLM purely as *context* (so it doesn't compose a
    redundant `cd` of its own) — :func:`run_repro` is what actually guarantees the command lands
    there, deterministically, the same way :func:`~src.engineer._verify_command` does; asking the
    LLM to compose the `cd` itself would make that guarantee only as reliable as the reply's own
    compliance, which nothing here would catch if it silently omitted or malformed it."""
    reply = complete_or_none(
        _repro_prompt(title, body, repo_dir), _REPRO_SCHEMA, stage="repro", subject=title
    )
    if reply is None:
        return None
    command = reply.get("command")
    return command.strip() if isinstance(command, str) and command.strip() else None


def run_repro(
    store: Store,
    repo: str,
    number: int,
    host: str,
    *,
    repo_dir: str | None = None,
    timeout: float = DEFAULT_REPRO_TIMEOUT_S,
    now: datetime | None = None,
) -> ReproResult | None:
    """Synthesize and run a repro command for (`repo`, `number`) on `host`, record the result
    to `store`, and return it.

    `repo_dir`, if given, is the checkout's actual absolute path on `host` — passed to
    :func:`synthesize_repro_command` as context, then deterministically prepended as a `cd` to
    whatever command comes back (mirroring :func:`~src.engineer._verify_command`'s own fix for
    the identical problem: plain `ssh host command` lands in the ssh session's default
    directory, not necessarily the checkout, per :mod:`~src.runner`'s own documented contract).
    Left `None` (default) to preserve prior behavior: assume the ssh session's own default
    directory is already the checkout.

    Returns:
        `None` if `store` has no record for (`repo`, `number`), or no repro command could be
        synthesized (see module docstring — both cases are skipped, not raised).

    Raises:
        runner.RunnerError: propagated uncaught from :func:`~src.runner.run` — see module
            docstring for why an infra failure isn't treated the same as "no signal."
    """
    item = get_item_or_skip(store, repo, number, stage="repro")
    if item is None:
        return None

    command = synthesize_repro_command(item.get("title") or "", item.get("body") or "", repo_dir)
    if command is None:
        return None
    if repo_dir:
        command = f"cd {shlex.quote(repo_dir)} && {command}"

    result = runner.run(host, command, timeout=timeout)
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
    record_run_best_effort(
        store, repro_result.to_run_record(), stage="repro", repo=repo, number=number
    )
    return repro_result
