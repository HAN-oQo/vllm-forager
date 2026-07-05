"""Remote runner (T3.1): execute a command on an MI250 host over ssh, streaming logs.

M3's Contribution plane runs entirely on hardware this (CPU) agent doesn't have — every repro,
build, and verify step from T3.2 onward is really "run this command on ``mi250-05x`` and look at
what came back." This module is the one place that knows how to do that: compose the ssh/scp
invocation, stream the remote command's output as it arrives (so a long build/test run is
visible while it's happening, not just after), and fetch back whatever artifact files the
caller asked for.

A non-zero exit code from the remote command is a normal, expected result — a failing repro
*is* the signal T3.2 wants captured — so :func:`run` never raises over it. The one exception is
ssh's own reserved exit code, ``255``: per ``ssh(1)``, ssh exits ``255`` when *it* couldn't
deliver the command at all (connection refused, auth failure, host key mismatch), as opposed to
forwarding the remote command's real exit status — treating that identically to a genuine
remote-command failure would let a transport hiccup masquerade as "the patch is still broken" to
a caller like T3.3, undermining CLAUDE.md's "MI250 is the empirical verification oracle"
guarantee. :func:`run` raises :class:`RunnerError` for it instead (the same heuristic tools like
Ansible/Fabric rely on) — accepting the rare, documented false-positive of a remote command that
itself happens to exit 255.

:class:`RunnerError` is otherwise reserved for cases where the command couldn't even be
attempted or observed (``ssh``/``scp`` missing from `PATH`, or the remote side never finishing
within `timeout`). Fetching one requested artifact failing (the file never materialized, `scp`
itself failed) doesn't abort the whole call either — it's logged and skipped, the same per-item
failure isolation every other agent in this codebase applies, so one missing log file doesn't
erase the exit code and every other artifact that *did* come back.

`timeout` is enforced by a background :class:`threading.Timer` that kills the process, not by
``Popen.wait(timeout=...)`` alone — reading a real pipe with ``for line in proc.stdout:`` blocks
until the remote process closes it (normally only on exit), so a `wait(timeout=...)` reached only
*after* that loop would never fire for a command that hangs while its stdout stays open. The
timer runs for the whole call, independent of whether the streaming loop is currently blocked on
a read or has already finished.

Real ssh/scp are never exercised by the default test run — see ``tests/test_runner.py``'s own
live smoke test, gated behind ``@pytest.mark.integration`` and a ``MI250_HOST`` env var, per
CLAUDE.md's "anything hitting live ... MI250 ... is @pytest.mark.integration" convention.

Known limitations, not fixed here (this is T3.1's first cut — no real caller exists yet to size
these against; see T3.2/T3.3):
- No remote working-directory concept — a caller that needs `command` to run inside a specific
  checkout (T3.2/T3.3 will) composes that itself, e.g. ``f"cd {repo_dir} && {command}"``.
- stdout/stderr are merged (``stderr=STDOUT``) so lines print in the order they actually
  interleaved — unlike :func:`~src.llm._run_claude_cli`, which keeps them separate specifically
  to extract a legible error message, `run()` cannot cleanly isolate the remote command's own
  stderr from its stdout. Kept simple for v1 (splitting them without deadlocking a caller reading
  one pipe while the other fills requires a second reader thread); the merged `log` still
  contains everything.
- No ssh connection reuse (``ControlMaster``) across calls, and no retry on a transient network
  drop — each `run()`/artifact fetch pays a fresh handshake.
- `log` is accumulated fully in memory for the call's lifetime — fine for a build/test log, not
  bounded for an arbitrarily long-running command.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 3600.0  # an hour — a vLLM (ROCm) build/test run is not a quick command.

# Artifacts are log/text files, not the build itself -- bounding each fetch independently of
# (and much shorter than) `timeout` keeps a caller's total wall-clock budget from silently
# becoming `len(artifacts) * timeout` (see module docstring on artifact fetch failure isolation).
_ARTIFACT_FETCH_TIMEOUT_S = 60.0

_SSH_TRANSPORT_FAILURE_EXIT_CODE = 255  # ssh(1): reserved for ssh's own connection/auth failures.


class RunnerError(RuntimeError):
    """``ssh``/``scp`` wasn't found on `PATH`, the remote command didn't finish within `timeout`,
    or ssh itself failed to reach `host` (exit ``255``) — never raised for the remote command's
    own non-255 exit code (see module docstring)."""


@dataclass(frozen=True)
class RunResult:
    """The outcome of one :func:`run` call.

    `log` is the command's combined stdout+stderr, in the order lines actually arrived (ssh
    itself interleaves the two streams over one channel, so there is no meaningful way to keep
    them separate). `artifacts` is the (possibly shorter than requested — see module docstring)
    list of local paths successfully fetched back.
    """

    exit_code: int
    log: str
    artifacts: tuple[str, ...] = ()


def run(
    host: str,
    command: str,
    *,
    artifacts: list[str] | None = None,
    local_dir: str | Path = ".",
    timeout: float = DEFAULT_TIMEOUT_S,
) -> RunResult:
    """Run `command` on `host` over ssh, streaming its output to this process's own stdout as
    it arrives, then fetch any `artifacts` (remote file paths) back into `local_dir` via scp.

    Artifacts are fetched regardless of `command`'s exit code — a failing repro's log/artifacts
    are exactly the evidence a caller like T3.2 wants, not just a successful run's.

    Args:
        host: an ssh-reachable alias/hostname (e.g. ``"mi250-051"``).
        command: the shell command to run on `host`, executed verbatim by the remote shell —
            callers must not compose it from untrusted external input without their own escaping.
        artifacts: remote file paths to fetch back after `command` finishes.
        local_dir: where fetched artifacts land (created if it doesn't exist), named to avoid
            collisions between two artifacts that share a basename in different remote
            directories.
        timeout: seconds to wait for `command` before killing it and raising.

    Raises:
        RunnerError: ``ssh`` isn't on `PATH`, `command` didn't finish within `timeout`, or ssh
            itself couldn't reach `host` (exit ``255``).
    """
    try:
        proc = subprocess.Popen(
            ["ssh", host, command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise RunnerError("`ssh` not found on PATH — install OpenSSH or add it to PATH") from exc

    if proc.stdout is None:  # unreachable given stdout=PIPE above; guards mypy + a bad mock
        raise RunnerError(f"no stdout pipe for the ssh process to {host!r}")

    timed_out = threading.Event()

    def _kill_on_timeout() -> None:
        timed_out.set()
        with contextlib.suppress(ProcessLookupError):  # already exited as the timer fired
            proc.kill()

    timer = threading.Timer(timeout, _kill_on_timeout)
    timer.start()
    try:
        lines = []
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)
        exit_code = proc.wait()
    finally:
        timer.cancel()

    if timed_out.is_set():
        raise RunnerError(f"command on {host!r} timed out after {timeout}s")

    log = "".join(lines)
    if exit_code == _SSH_TRANSPORT_FAILURE_EXIT_CODE:
        raise RunnerError(
            f"ssh to {host!r} failed (exit {_SSH_TRANSPORT_FAILURE_EXIT_CODE}) — connection/auth "
            f"error, not a remote command result: {log[-500:]}"
        )

    fetched = _fetch_artifacts(host, artifacts or [], local_dir)
    return RunResult(exit_code=exit_code, log=log, artifacts=tuple(fetched))


def _local_artifact_path(remote_path: str, local_dir: str | Path) -> Path:
    """A local path for `remote_path` under `local_dir` that can't collide with another
    artifact of the same basename from a different remote directory (flattens the whole
    remote path into the filename instead of keeping just its basename)."""
    flattened = remote_path.strip("/").replace("/", "__") or "artifact"
    return Path(local_dir) / flattened


def _fetch_artifacts(host: str, remote_paths: list[str], local_dir: str | Path) -> list[str]:
    """Best-effort scp of each `remote_paths` entry into `local_dir` — a failed fetch is logged
    and skipped (see module docstring), not raised."""
    if not remote_paths:
        return []
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    fetched = []
    for remote_path in remote_paths:
        local_path = _local_artifact_path(remote_path, local_dir)
        try:
            subprocess.run(
                ["scp", f"{host}:{remote_path}", str(local_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=_ARTIFACT_FETCH_TIMEOUT_S,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            print(f"runner: failed to fetch {remote_path!r} from {host}: {detail}", file=sys.stderr)
            continue
        fetched.append(str(local_path))
    return fetched
