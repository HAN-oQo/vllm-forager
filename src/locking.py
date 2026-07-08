"""Run lock (T4.3): a non-blocking, per-host file lock so two orchestrator ticks never overlap.

T4.2 first added this guarantee as a shell-level `flock` inside `scripts/orchestrator.sh`, but
that only protected invocations that went through the wrapper script -- a direct
`python -m src.orchestrator --once` (e.g. from cron without the wrapper, or a human running it
by hand) had no such guard. This module gives the same "second run backs off, first keeps
going" guarantee inside :func:`src.orchestrator.main` itself, so it covers every invocation
path and is exercised by an offline pytest test (`tests/test_locking.py`) instead of only a
manual demo. `scripts/orchestrator.sh`'s own `flock` block was removed once this landed --
keeping both would have meant two independent locks racing to lock the *same* default path
(`config.DATA_DIR` == the repo's `data/` dir by default), where the shell's lock -- held for the
whole script's lifetime and inherited by the Python child process's file descriptor table --
would make this module's own lock attempt always fail, breaking every tick launched through the
wrapper script.

Locking is host-local (a file under `config.DATA_DIR`), not KB-level: it stops two *processes*
on the same host from racing, but says nothing about a Firestore-backed KB shared across hosts.
That's an acceptable scope for now -- ce-master is the only host that runs the orchestrator
(CLAUDE.md's Infrastructure section) -- and matches T4.3's own DEVPLAN example ("a second run
starts while the first is mid-flight -> it backs off").
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager

from . import config


class LockHeld(RuntimeError):
    """Another process already holds this lock -- the caller should back off, not retry/crash."""


@contextmanager
def run_lock(name: str) -> Iterator[None]:
    """Hold an exclusive, non-blocking lock named `name` for the duration of the `with` block.

    Raises :class:`LockHeld` immediately (never blocks) if another process already holds it.
    The lock file lives at ``config.DATA_DIR / f"{name}.lock"`` and is never deleted -- an
    `flock` is released when its file descriptor closes (process exit, or normal `with`-block
    exit via the `finally` below), so a stale leftover file from a crashed process is harmless:
    the *lock*, not the file's existence, is what matters.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / f"{name}.lock"
    fh = path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise LockHeld(f"lock {name!r} is already held ({path})") from None
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
