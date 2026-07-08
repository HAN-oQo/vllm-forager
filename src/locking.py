"""Run lock (T4.3): a non-blocking, per-host file lock so two orchestrator ticks never overlap.

T4.2 first added this guarantee as a shell-level `flock` inside `scripts/orchestrator.sh`, but
that only protected invocations that went through the wrapper script -- a direct
`python -m src.orchestrator --once` (e.g. from cron without the wrapper, or a human running it
by hand) had no such guard. This module gives the same "second run backs off, first keeps
going" guarantee inside :func:`src.orchestrator.main` itself, so it covers every invocation of
*that* entry point (the wrapper script, a bare CLI call, a future caller), and is exercised by
an offline pytest test (`tests/test_locking.py`) instead of only a manual demo.
`scripts/orchestrator.sh`'s own `flock` block was removed once this landed -- keeping both would
have meant two independent locks racing to lock the *same* default path (`config.DATA_DIR` ==
the repo's `data/` dir by default), where the shell's lock -- held for the whole script's
lifetime and inherited by the Python child process's file descriptor table -- would make this
module's own lock attempt always fail, breaking every tick launched through the wrapper script.

**Scope gap, not fixed here:** this only guards `orchestrator.main()`'s own invocations against
each other. It does **not** protect a *standalone* `python -m src.collector` /
`python -m src.analyze` / `python -m src.candidates` run (e.g. via `/collect-loop`, or
`scripts/collect.sh`'s own independent cron entry, both still directly runnable per
`docs/RUNBOOK.md`) from racing an orchestrator tick that's internally calling the same agent
`main()`s under this lock -- those CLIs take no lock of their own. Locking every individual
entry point is a broader, project-wide idempotency effort DEVPLAN's own T4.3 "e.g." doesn't
describe (it's specifically about *one orchestrated run* overlapping another), and is left as a
future refinement, not this todo's.

Locking is host-local (a file under `config.DATA_DIR`), not KB-level: it stops two *processes*
on the same host from racing, but says nothing about a Firestore-backed KB shared across hosts.
That's an acceptable scope for now -- ce-master is the only host that runs the orchestrator
(CLAUDE.md's Infrastructure section) -- and matches T4.3's own DEVPLAN example ("a second run
starts while the first is mid-flight -> it backs off").

Also not fixed here: no staleness/timeout detection. `flock` only releases on file-descriptor
close (process exit or a normal `with`-block exit) -- a genuinely hung-but-alive process (e.g. a
wedged network/LLM call inside a stage) holds the lock indefinitely, and every subsequent tick
then backs off via :class:`LockHeld` exactly as it would for healthy contention, with no way to
tell the two apart from outside. Surfacing that distinction is T4.4/T4.5's job (run events +
heartbeats), not this lock's.
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
    Any other OS-level failure (e.g. `config.DATA_DIR` unwritable, a filesystem that can't honor
    advisory locks) propagates as its own real exception instead of being misreported as
    "already held" -- catching only :class:`BlockingIOError` (Python's own errno-mapped
    exception for `EAGAIN`/`EWOULDBLOCK`, PEP 3151) rather than a bare `OSError` is what makes
    that distinction; a caller silently treating a real filesystem problem as routine lock
    contention would look "healthy" (a clean backoff) tick after tick, forever.

    The lock file lives at ``config.DATA_DIR / f"{name}.lock"`` and is never deleted -- an
    `flock` is released when its file descriptor closes (process exit, or normal `with`-block
    exit), so a stale leftover file from a crashed process is harmless: the *lock*, not the
    file's existence, is what matters.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / f"{name}.lock"
    fh = path.open("a")  # never truncate -- the file's content is never read, only its fd/inode
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise LockHeld(f"lock {name!r} is already held ({path})") from None
    try:
        yield
    finally:
        fh.close()  # releases the flock too -- no separate LOCK_UN needed
