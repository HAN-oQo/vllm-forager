"""Build/version stamp (T5.15): the git commit the running process was started from.

A long-running dashboard process holds the ``dashboard/render.py`` it imported **at startup**
— Python never hot-reloads it, so after a merge the live page can silently keep serving stale
code with no way to tell from the page itself (observed 2026-07: a process served the
pre-tree flat renderer for two days after the tree-view design landed, until someone noticed
and manually restarted it). :data:`GIT_SHA` exists so that question is answerable by a glance
at the page footer, not by shelling into the host and comparing SHAs by hand.

Computed **once, at import time** — a module-level constant, not a function a caller invokes
per request. Recomputing it on every page load would defeat the entire point: it must reflect
the code this process actually loaded at startup, not whatever happens to be on disk right
now (which is exactly the case that differs after an un-restarted ``git pull``).
"""

from __future__ import annotations

import subprocess

from . import config

_GIT_SHA_TIMEOUT_S = 5


def _git_sha() -> str:
    """The running process's own checkout's short git commit SHA, or ``"unknown"`` if it
    can't be determined (not a git checkout, `git` not on `PATH`, a shallow-clone quirk) —
    degrades rather than raising, since a non-essential diagnostic must never crash a
    dashboard import."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_SHA_TIMEOUT_S,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() or "unknown"


GIT_SHA = _git_sha()
