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
- **No idempotency, and this is not a hypothetical edge case**: ``ccusage session --json`` with
  no ``--since``/``--until`` returns a node's *entire* historical session list every time it's
  invoked (verified against a real run on this machine). The exact "a nightly `ccusage --json`
  per node" operating mode this todo's own DEVPLAN ``e.g.`` describes would re-ingest every
  session on every run, inflating :func:`src.cost.rollup`'s totals without bound from the very
  first repeat invocation — not an "overlapping window" corner case. A naive fix (skip a
  session id already seen) is *also wrong*: a session's reported totals are a cumulative
  snapshot that keeps growing across turns, so "skip if seen" would freeze its cost at whatever
  the first ingest happened to observe rather than update it. A correct fix needs either
  ``ccusage --since <high-water-mark-of-lastActivity>`` tracked via ``store.get_state``/
  ``set_state`` (matching ``src.orchestrator``'s own cursor pattern) on the *caller* side, or a
  KB-level upsert-by-``(run_id, model)`` instead of :func:`src.cost.write_cost_record`'s
  append-only write — neither exists yet. No scheduled caller (cron/script) invokes
  :func:`ingest` today, so this cannot yet fire in practice; it becomes live the moment
  something schedules repeat ingestion, which must implement one of the above first.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from . import cost
from .store.base import Store

# Only ccusage's own recorded label for a real `claude -p` session -- see module docstring.
_CLAUDE_CODE_AGENT_LABEL = "claude"

# The `agent` tag this module's own cost records carry -- distinct from T4.8's internal agent
# names (analyst/forecaster/reporter/scout), since a Claude Code session isn't one of them.
_SESSION_AGENT_TAG = "claude_code_session"


def _normalize_timestamp(raw: object) -> str:
    """ccusage's `lastActivity` is ISO-8601 with milliseconds (``...T00:29:10.194Z``) --
    reformat to this codebase's own `recorded_at` convention so every record in the run-event
    stream sorts/parses the same way. `raw` is treated as untrusted external input (parsed JSON
    from a separately-versioned CLI, per this module's own docstring): a non-string value is
    coerced via `str()` rather than crashing on a missing `.replace()` method, and a value with
    no UTC offset (no trailing `Z`) is treated as UTC -- mirroring `src.agents.scout._parse_ts`
    -- rather than shifted by `datetime.astimezone`'s "naive means local host time" default,
    which would silently vary this function's output by whatever timezone happens to run it.
    Falls back to the raw string on any parse failure (rather than raising) -- `cost.rollup`
    already tolerates an unparseable `recorded_at` by bucketing it under `"unknown"`."""
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return str(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime(cost.RECORDED_AT_FORMAT)


def _records_for_session(session: dict, *, loop: str | None) -> list[cost.CostRecord]:
    """One :class:`~src.cost.CostRecord` per ``modelBreakdowns`` entry in `session`, or `[]` if
    `session` isn't a real Claude Code session (see :func:`ingest`'s own agent filter)."""
    if session.get("agent") != _CLAUDE_CODE_AGENT_LABEL:
        return []
    run_id = str(session.get("period") or "")
    recorded_at = _normalize_timestamp((session.get("metadata") or {}).get("lastActivity"))
    records: list[cost.CostRecord] = []
    for breakdown in session.get("modelBreakdowns") or []:
        records.append(
            {
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
        )
    return records


def ingest(store: Store, ccusage_json: dict, *, loop: str | None = None) -> int:
    """Ingest `ccusage session --json`'s already-parsed output into the cost run-event stream,
    one record per ``(session, model)`` breakdown. Returns the number of records *attempted*
    (each write goes through :func:`src.cost.write_cost_record`, which is itself best-effort —
    a KB-write failure is logged, not raised, so this count can exceed what's actually queryable
    via ``store.list_runs(stage="cost")`` afterward if the KB was unwritable for part of a run).

    Each session's own `period` (its session id) becomes every one of its records' `run_id`, so
    a multi-model session's rows still stay correlated to one Claude Code invocation. A single
    malformed session (unexpected field shapes) is skipped with a message to stderr rather than
    aborting every other session in the same `ccusage_json` payload — matching this codebase's
    established per-item failure isolation (e.g. `src.agents.analyst`'s per-item classification).
    """
    sessions = ccusage_json.get("session")
    if sessions is None:
        if ccusage_json:
            print(
                "cost_ccusage: no 'session' key in ccusage JSON "
                f"(keys: {sorted(ccusage_json)}) -- did the ccusage schema change?",
                file=sys.stderr,
            )
        return 0

    attempted = 0
    for session in sessions:
        try:
            records = _records_for_session(session, loop=loop)
        except Exception as exc:
            print(f"cost_ccusage: skipping a malformed session: {exc}", file=sys.stderr)
            continue
        for record in records:
            cost.write_cost_record(store, record)
            attempted += 1
    return attempted
