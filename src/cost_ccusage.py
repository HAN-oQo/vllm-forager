"""Claude Code session cost ingest (T4.9).

The `claude -p` sessions this project's own dev-loop/collect-loop/triage self-heal run
**bypass** :func:`src.llm.complete` entirely — they *are* the LLM call, not something calling
into it — so T4.8's ``llm.cost_context`` mechanism can never see them. `ccusage
<https://github.com/ryoppippi/ccusage>`_ (``npx ccusage@latest session --json``) reads Claude
Code's own local session transcripts and reports token/cost totals per session; this module
ingests that JSON into the same cost run-event stream T4.8 writes
(``store.record_run({"stage": "cost", ...})`` — see :mod:`src.cost`), one record per
``(session, model)`` breakdown, tagged by the session id (as ``run_id``) and, where the caller
knows it, which of this project's own loops the session belonged to.

Known limitations (disclosed, not fixed here):

- ``ccusage`` has no concept of which of *this project's* loops (dev-loop, collect-loop,
  triage self-heal, ``/code-review``, ...) a session belongs to — it only knows Claude Code's
  own session id. Every record ingested by one :func:`ingest` call gets the same
  caller-supplied ``loop`` (or ``None``); there is no per-session loop attribution without
  something upstream recording which loop invoked which session id, which nothing does today.
- ``ccusage`` aggregates every coding-CLI tool it finds local logs for (Codex, Gemini,
  opencode, ...), not just Claude Code — :func:`ingest` only ingests rows whose ``agent`` is
  ``"claude"`` (this project only ever invokes ``claude -p``), silently skipping any other
  tool's rows rather than misattributing them as this system's own spend.
- Re-running :func:`ingest` against overlapping ``ccusage --json`` output (e.g. a nightly cron
  re-querying a rolling window) re-records every session again — there is no idempotency key
  checked against what's already in the KB, so re-ingesting the same session twice double-counts
  it in :func:`src.cost.rollup`. A future caller should track "already-ingested session ids"
  itself (e.g. via ``store.get_state``/``set_state``) if this becomes a real operational
  problem; today's Test/e.g. only exercise a single ingest of a fixture.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import cost
from .store.base import Store

# Only ccusage's own recorded label for a real `claude -p` session -- see module docstring.
_CLAUDE_CODE_AGENT_LABEL = "claude"

# The `agent` tag this module's own cost records carry -- distinct from T4.8's internal agent
# names (analyst/forecaster/reporter/scout), since a Claude Code session isn't one of them.
_SESSION_AGENT_TAG = "claude_code_session"


def _normalize_timestamp(raw: str | None) -> str:
    """ccusage's `lastActivity` is ISO-8601 with milliseconds (``...T00:29:10.194Z``) --
    reformat to this codebase's own `recorded_at` convention so every record in the run-event
    stream sorts/parses the same way. Falls back to the raw string (rather than raising) if it
    doesn't parse -- `cost.rollup` already tolerates an unparseable `recorded_at` by bucketing
    it under `"unknown"`."""
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ingest(store: Store, ccusage_json: dict, *, loop: str | None = None) -> int:
    """Ingest `ccusage session --json`'s already-parsed output into the cost run-event stream,
    one record per ``(session, model)`` breakdown. Returns the number of records written.

    Each session's own `period` (its session id) becomes every one of its records' `run_id`,
    so a multi-model session's rows still stay correlated to one Claude Code invocation.
    """
    written = 0
    for session in ccusage_json.get("session") or []:
        if session.get("agent") != _CLAUDE_CODE_AGENT_LABEL:
            continue
        run_id = str(session.get("period") or "")
        recorded_at = _normalize_timestamp((session.get("metadata") or {}).get("lastActivity"))
        for breakdown in session.get("modelBreakdowns") or []:
            record = {
                "stage": "cost",
                "agent": _SESSION_AGENT_TAG,
                "run_id": run_id,
                "loop": loop,
                "provider": "claude_cli",
                "model": breakdown.get("modelName") or "unknown",
                "tokens_in": int(breakdown.get("inputTokens") or 0),
                "tokens_out": int(breakdown.get("outputTokens") or 0),
                "tokens_cache": int(breakdown.get("cacheCreationTokens") or 0)
                + int(breakdown.get("cacheReadTokens") or 0),
                "cost_usd": float(breakdown.get("cost") or 0.0),
                "recorded_at": recorded_at,
            }
            cost.write_cost_record(store, record)
            written += 1
    return written
