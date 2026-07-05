"""Remote runner (T3.1): execute a command on an MI250 host over ssh, streaming logs.

M3's Contribution plane runs entirely on hardware this (CPU) agent doesn't have — every repro,
build, and verify step from T3.2 onward is really "run this command on ``mi250-05x`` and look at
what came back." This module is the one place that knows how to do that: compose the ssh/scp
invocation, stream the remote command's output as it arrives (so a long build/test run is
visible while it's happening, not just after), and fetch back whatever artifact files the
caller asked for.

A non-zero exit code from the remote command is a normal, expected result — a failing repro
*is* the signal T3.2 wants captured — so :func:`run` never raises over it; :class:`RunnerError`
is reserved for cases where the command couldn't even be attempted or observed (``ssh``/``scp``
missing from `PATH`, or the remote side never finishing within `timeout`). Fetching one
requested artifact failing (the file never materialized, `scp` itself failed) doesn't abort the
whole call either — it's logged and skipped, the same per-item failure isolation every other
agent in this codebase applies, so one missing log file doesn't erase the exit code and every
other artifact that *did* come back.

Real ssh/scp are never exercised by the default test run — see ``tests/test_runner.py``'s own
live smoke test, gated behind ``@pytest.mark.integration`` and a ``MI250_HOST`` env var, per
CLAUDE.md's "anything hitting live ... MI250 ... is @pytest.mark.integration" convention.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 3600.0  # an hour — a vLLM (ROCm) build/test run is not a quick command.


class RunnerError(RuntimeError):
    """``ssh``/``scp`` wasn't found on `PATH`, or the remote command didn't finish within
    `timeout` — never raised for the remote command's own non-zero exit (see module docstring)."""


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
        command: the shell command to run on `host`.
        artifacts: remote file paths to fetch back after `command` finishes.
        local_dir: where fetched artifacts land, named by their remote basename.
        timeout: seconds to wait for `command` before killing it and raising.

    Raises:
        RunnerError: ``ssh`` isn't on `PATH`, or `command` didn't finish within `timeout`.
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
        raise RunnerError("`ssh` not found on PATH") from exc

    lines = []
    assert proc.stdout is not None  # guaranteed by stdout=PIPE above
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)

    try:
        exit_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise RunnerError(f"command on {host!r} timed out after {timeout}s") from exc

    fetched = _fetch_artifacts(host, artifacts or [], local_dir, timeout=timeout)
    return RunResult(exit_code=exit_code, log="".join(lines), artifacts=tuple(fetched))


def _fetch_artifacts(
    host: str, remote_paths: list[str], local_dir: str | Path, *, timeout: float
) -> list[str]:
    """Best-effort scp of each `remote_paths` entry into `local_dir` — a failed fetch is logged
    and skipped (see module docstring), not raised."""
    fetched = []
    for remote_path in remote_paths:
        local_path = Path(local_dir) / Path(remote_path).name
        try:
            subprocess.run(
                ["scp", f"{host}:{remote_path}", str(local_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as exc:
            print(f"runner: failed to fetch {remote_path!r} from {host}: {exc}", file=sys.stderr)
            continue
        fetched.append(str(local_path))
    return fetched
