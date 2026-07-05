"""Engineer patch loop (T3.3): LLM-generated patch → rebuild/test on MI250 → confirm the T3.2
failing signal flips.

The core contribution act. Given a candidate T3.2 already reproduced (a `stage="repro"` run in
the KB with `reproduced=True`), an LLM synthesizes a patch (a unified diff) from the issue text
and the failing log, applies it on a fresh candidate branch on `host`, and re-runs the *exact
same repro command* that originally failed. MI250 is the empirical oracle here (CLAUDE.md):
`verified` is never a judgment call, only `exit_code == 0` on that re-run — the same failing
test that justified T3.2's `reproduced=True` now passing, nothing more. `verified=False` means
exactly what the DEVPLAN says it should: **no PR** — this module produces a verdict, not a PR;
T3.4's ensemble review and T3.5's human gate are what a `verified=True` result feeds into next,
and neither runs from here.

**Security note:** the same one from :mod:`src.repro`, extended — the composed remote script
now embeds an LLM-synthesized *patch* (a diff) via a shell heredoc, not just a repro command, so
whatever untrusted-input risk :mod:`src.repro`'s module docstring describes applies here too,
compounded: a patch line that happens to exactly match the heredoc's own closing delimiter
(:data:`_PATCH_HEREDOC_DELIMITER`) would prematurely terminate it, and everything after that
line in the patch is then parsed and *executed* as shell input rather than treated as diff
content — a more direct code-injection vector than :mod:`src.repro`'s single flat command. Same
mitigation as there: none at the code level in this first cut — treat any host this points at as
disposable, non-production hardware.

Like :mod:`src.repro`, "no reproduced baseline for this candidate" and "the LLM couldn't
synthesize a usable patch" both skip (return `None`, logged), not raise — the same per-item
failure isolation every sibling agent applies. A :class:`~src.runner.RunnerError` still
propagates uncaught (an infra failure isn't a verification verdict), and a `store.record_run`
failure is caught so an already-completed MI250 verify attempt is never discarded — both
mirroring :mod:`src.repro`'s own choices for the identical reasons.

Known limitations, not fixed here:
- Applying the patch, committing it, and re-running the repro command all happen as one script
  under one `runner.run` call — a `verified=False` doesn't distinguish "the diff didn't even
  apply" from "it applied but the bug is still there," and a hang anywhere in the script shares
  one `timeout` budget. Splitting these into separately-attributable steps is a real improvement
  T3.4's ensemble review would benefit from, but isn't needed to satisfy T3.3's own acceptance
  criterion (a binary verified signal) and would add real complexity (multiple `runner.run`
  round-trips, more MI250 wall-clock) this first cut doesn't take on.
- :func:`~src.store.base.Store.list_runs`'s ordering isn't guaranteed across every backend (see
  its own docstring) — picking "the baseline" among more than one prior `stage="repro"` run
  uses `max(..., key=recorded_at)`, which is only as reliable as every recorded run actually
  carrying a real `recorded_at`. `EngineerResult.baseline_recorded_at` records which one was
  actually used, so this is at least traceable, not silently ambiguous.
- `base_ref` (what a candidate branch resets to before applying a patch) defaults to `"main"`,
  an assumption about the checkout's default branch name that may not hold for every real
  deployment — callers should override it to match whatever branch the MI250 checkout actually
  tracks.
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timezone

from . import llm, runner
from .agents.forecaster import TS_FORMAT
from .stages import get_item_or_skip, record_run_best_effort
from .store.base import MAX_RUN_LOG_CHARS, Store

_PATCH_SCHEMA = {
    "type": "object",
    "properties": {"patch": {"type": "string"}},
    "required": ["patch"],
}

# Rebuild + re-run a repro command is more work than a bare repro check, but still not a full
# from-scratch ROCm build -- longer than repro.py's own default, shorter than runner.py's.
DEFAULT_ENGINEER_TIMEOUT_S = 1800.0

# What a fresh candidate branch resets to before a patch is applied -- see module docstring's
# own "Known limitations" note on this being an assumption, not a discovered value.
DEFAULT_BASE_REF = "main"

# A patch line that happens to equal this exactly would prematurely close the heredoc below
# (see module docstring's Security note) -- long and namespaced specifically to make an
# accidental collision with real diff content very unlikely, though not impossible.
_PATCH_HEREDOC_DELIMITER = "FORAGER_PATCH_EOF"


@dataclasses.dataclass(frozen=True)
class EngineerResult:
    """One patch-verify attempt's outcome.

    `verified` is `exit_code == 0` on the re-run of the exact command that produced the T3.2
    baseline's failing signal — never a judgment about patch quality beyond that (see module
    docstring). `baseline_recorded_at` traces which prior repro run this compared against (see
    module docstring's own note on `list_runs` ordering). `command` (the full composed remote
    script, patch text included) is kept here for debugging but deliberately dropped from the
    persisted KB record by :meth:`to_run_record` — it would otherwise duplicate `patch`'s own
    (unbounded) text in every stored run.
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
        :meth:`~src.repro.ReproResult.to_run_record`; `command` is dropped entirely (see this
        class's own docstring)."""
        record = dataclasses.asdict(self)
        record["stage"] = "verify"
        record["log"] = record["log"][-MAX_RUN_LOG_CHARS:]
        del record["command"]
        return record


def _branch_name(repo: str, number: int) -> str:
    """A deterministic branch name for this `repo`/`number` candidate — includes `repo` (its
    own ``/`` replaced, matching :mod:`~src.store.jsonl_store`'s own file-naming convention for
    the same reason) since two different tracked repos can share the same issue/PR number, and
    a name collision would make one candidate's branch silently double as another's. `git
    checkout -B` (not `-b`) is used to apply it, so retrying the same candidate resets the
    branch rather than failing on "branch already exists" — see :func:`_verify_command` for why
    an explicit `base_ref` is what actually makes that reset land on a clean base."""
    return f"forager/{repo.replace('/', '-')}-{number}"


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


def _verify_command(branch: str, patch: str, repro_command: str, *, base_ref: str) -> str:
    """One remote shell **script** (not a single `&&`-chain — see below): reset `branch` to a
    clean `base_ref`, apply `patch` via a heredoc, commit it, then re-run `repro_command` — the
    exact command T3.2 originally captured a failing signal from.

    Uses ``set -e`` plus one statement per line, not `cmd1 && cmd2 <<'EOF' ... EOF && cmd3`: a
    heredoc's closing delimiter must be alone on its own line, which ends the *enclosing*
    command as far as the shell's grammar is concerned — a `&&` on the very next line has
    nothing to its left to attach to and is a syntax error (verified directly: `bash -c "echo
    one && cat <<'E'\\nx\\nE\\n&& echo two"` fails with "syntax error near unexpected token
    '&&'"). `set -e` reproduces the intended short-circuit-on-failure behavior correctly instead.

    `git checkout -B branch base_ref` (an explicit `base_ref`, not the checkout's current HEAD)
    is what makes a retry actually reset to a clean base rather than stacking a new patch on top
    of whatever a prior attempt already committed to `branch`; `git clean -fd` afterward removes
    any untracked leftovers (e.g. a new file a prior attempt's patch added) `checkout` alone
    wouldn't touch. `git add -A` (not `git commit -am`) is used so a patch that adds a new file
    is actually staged — `-a` only stages modifications/deletions to already-tracked files, so a
    patch consisting solely of a new file would otherwise make `git commit` fail with nothing to
    commit, aborting before `repro_command` ever runs and misreporting a correct patch as
    `verified=False`.
    """
    return "\n".join(
        [
            "set -e",
            f"git checkout -B {branch} {base_ref}",
            "git clean -fd",
            f"git apply <<'{_PATCH_HEREDOC_DELIMITER}'",
            patch,
            _PATCH_HEREDOC_DELIMITER,
            "git add -A",
            f"git commit -m 'forager: candidate {branch} patch'",
            repro_command,
        ]
    )


def run_engineer(
    store: Store,
    repo: str,
    number: int,
    host: str,
    *,
    base_ref: str = DEFAULT_BASE_REF,
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
    baseline = max(baseline_runs, key=lambda r: r.get("recorded_at") or "")

    item = get_item_or_skip(store, repo, number, stage="engineer")
    if item is None:
        return None

    patch = synthesize_patch(
        item.get("title") or "", item.get("body") or "", baseline.get("log") or ""
    )
    if patch is None:
        return None

    branch = _branch_name(repo, number)
    command = _verify_command(branch, patch, baseline.get("command") or "", base_ref=base_ref)
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
    record_run_best_effort(
        store, engineer_result.to_run_record(), stage="engineer", repo=repo, number=number
    )
    return engineer_result
