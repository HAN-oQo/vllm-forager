"""Engineer patch loop (T3.3): LLM-generated patch → rebuild/test on MI250 → confirm the T3.2
failing signal flips.

The core contribution act. Given a candidate T3.2 already reproduced (a `stage="repro"` run in
the KB with `reproduced=True`), an LLM synthesizes a patch (a unified diff) from the issue text
and the failing log, applies it on a fresh branch on `host`, and re-runs the *exact same repro
command* that originally failed. MI250 is the empirical oracle here (CLAUDE.md): `verified` is
never a judgment call, only `exit_code == 0` on that re-run — the same failing test that
justified T3.2's `reproduced=True` now passing, nothing more. `verified=False` means exactly
what the DEVPLAN says it should: **no PR** — this module produces a verdict, not a PR; T3.4's
ensemble review and T3.5's human gate are what a `verified=True` result feeds into next, and
neither runs from here.

**Security note:** the same one from :mod:`src.repro`, extended — the composed remote command
now embeds an LLM-synthesized *patch* (a diff) via a shell heredoc, not just a repro command, so
whatever untrusted-input risk :mod:`src.repro`'s module docstring describes applies here too,
compounded (a malicious diff is a more direct code-injection vector than a malicious test
command). Same mitigation as there: none at the code level in this first cut — treat any host
this points at as disposable, non-production hardware.

Like :mod:`src.repro`, "no reproduced baseline for this candidate" and "the LLM couldn't
synthesize a usable patch" both skip (return `None`, logged), not raise — the same per-item
failure isolation every sibling agent applies. A :class:`~src.runner.RunnerError` still
propagates uncaught (an infra failure isn't a verification verdict), and a `store.record_run`
failure is caught so an already-completed MI250 verify attempt is never discarded — both
mirroring :mod:`src.repro`'s own choices for the identical reasons.

Known limitation, not fixed here: :func:`~src.store.base.Store.list_runs`'s ordering isn't
guaranteed across every backend (see its own docstring) — picking "the baseline" when a
candidate has more than one prior `stage="repro"` run uses whatever order `list_runs` returns,
best-effort on :class:`~src.store.jsonl_store.JsonlStore`, not guaranteed on
:class:`~src.store.firestore_store.FirestoreStore`. `EngineerResult.baseline_recorded_at`
records which one was actually used, so this is at least traceable, not silently ambiguous.
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timezone

from . import llm, runner
from .agents.forecaster import TS_FORMAT
from .store.base import MAX_RUN_LOG_CHARS, Store

_PATCH_SCHEMA = {
    "type": "object",
    "properties": {"patch": {"type": "string"}},
    "required": ["patch"],
}

# Rebuild + re-run a repro command is more work than a bare repro check, but still not a full
# from-scratch ROCm build -- longer than repro.py's own default, shorter than runner.py's.
DEFAULT_ENGINEER_TIMEOUT_S = 1800.0


@dataclasses.dataclass(frozen=True)
class EngineerResult:
    """One patch-verify attempt's outcome.

    `verified` is `exit_code == 0` on the re-run of the exact command that produced the T3.2
    baseline's failing signal — never a judgment about patch quality beyond that (see module
    docstring). `baseline_recorded_at` traces which prior repro run this compared against (see
    module docstring's own note on `list_runs` ordering).
    """

    repo: str
    number: int
    host: str
    branch: str
    patch: str
    command: str
    exit_code: int
    log: str
    verified: bool
    baseline_recorded_at: str
    recorded_at: str

    def to_run_record(self) -> dict:
        """This result as a plain dict, shaped for :meth:`~src.store.base.Store.record_run` —
        `log` is truncated the same way, and for the same reason, as
        :meth:`~src.repro.ReproResult.to_run_record`."""
        record = dataclasses.asdict(self)
        record["stage"] = "verify"
        record["log"] = record["log"][-MAX_RUN_LOG_CHARS:]
        return record


def _branch_name(number: int) -> str:
    """A deterministic branch name for candidate `number` — `git checkout -B` (not `-b`) is
    used to apply it, so retrying the same candidate resets the branch rather than failing on
    "branch already exists"."""
    return f"forager/candidate-{number}"


def _patch_prompt(title: str, body: str, failing_log: str) -> str:
    return (
        "This is a vLLM/ROCm GitHub issue and the failing output from reproducing it. Write a "
        "minimal unified diff (`git apply`-compatible patch) that fixes the bug. Reply with "
        "`patch` (the diff text).\n\n"
        f"Title: {title[:500]}\n\nBody: {(body or '')[:4000]}\n\n"
        f"Failing output:\n{(failing_log or '')[:4000]}"
    )


def synthesize_patch(title: str, body: str, failing_log: str) -> str | None:
    """A unified diff an LLM believes fixes `title`/`body`'s bug (informed by `failing_log`,
    the T3.2 baseline's captured output), or `None` if the call failed or the reply didn't
    shape into a usable patch (see module docstring: skip, logged, not raised)."""
    try:
        reply = llm.complete(_patch_prompt(title, body, failing_log), json_schema=_PATCH_SCHEMA)
    except llm.LLMError as exc:
        print(f"engineer: patch synthesis failed for {title!r}: {exc}", file=sys.stderr)
        return None
    if not isinstance(reply, dict):
        return None
    patch = reply.get("patch")
    return patch.strip() if isinstance(patch, str) and patch.strip() else None


def _verify_command(branch: str, patch: str, repro_command: str) -> str:
    """One remote shell command: create/reset `branch`, apply `patch` via a heredoc, commit it,
    then re-run `repro_command` — the exact command T3.2 originally captured a failing signal
    from."""
    return (
        f"git checkout -B {branch} && "
        f"git apply <<'FORAGER_PATCH_EOF'\n{patch}\nFORAGER_PATCH_EOF\n"
        f"&& git commit -am 'forager: candidate {branch} patch' "
        f"&& {repro_command}"
    )


def run_engineer(
    store: Store,
    repo: str,
    number: int,
    host: str,
    *,
    timeout: float = DEFAULT_ENGINEER_TIMEOUT_S,
    now: datetime | None = None,
) -> EngineerResult | None:
    """Synthesize a patch for (`repo`, `number`), apply and rebuild/test it on `host`, record
    the result to `store`, and return it.

    Returns:
        `None` if there's no reproduced T3.2 baseline run for (`repo`, `number`), `store` has
        no item record for it, or no patch could be synthesized (all three skip, not raise).

    Raises:
        runner.RunnerError: propagated uncaught from :func:`~src.runner.run` — an infra failure
            isn't a verification verdict (see module docstring).
    """
    baseline_runs = [
        r for r in store.list_runs(repo=repo, number=number, stage="repro") if r.get("reproduced")
    ]
    if not baseline_runs:
        print(f"engineer: no reproduced baseline for {repo}#{number}", file=sys.stderr)
        return None
    baseline = baseline_runs[-1]

    item = store.get_item(repo, number)
    if item is None:
        print(f"engineer: no KB record for {repo}#{number}", file=sys.stderr)
        return None

    patch = synthesize_patch(
        item.get("title") or "", item.get("body") or "", baseline.get("log") or ""
    )
    if patch is None:
        return None

    branch = _branch_name(number)
    command = _verify_command(branch, patch, baseline.get("command") or "")
    result = runner.run(host, command, timeout=timeout)
    when = now or datetime.now(timezone.utc)
    engineer_result = EngineerResult(
        repo=repo,
        number=number,
        host=host,
        branch=branch,
        patch=patch,
        command=command,
        exit_code=result.exit_code,
        log=result.log,
        verified=result.exit_code == 0,
        baseline_recorded_at=baseline.get("recorded_at") or "",
        recorded_at=when.strftime(TS_FORMAT),
    )
    try:
        store.record_run(engineer_result.to_run_record())
    except Exception as exc:
        # See src.repro's identical choice: never lose an already-run result over a mere
        # persistence hiccup.
        print(f"engineer: failed to record run for {repo}#{number}: {exc}", file=sys.stderr)
    return engineer_result
