"""Per-agent LLM cost capture (T4.8) — folded into the T4.4 run-event stream.

:func:`record_cost` writes one record per :func:`src.llm.complete`/``complete_detailed`` call
via ``store.record_run`` with ``stage="cost"`` — the same append-only collection T4.4's
started/heartbeat/ok/failed events already use (``store.list_runs(stage="cost")`` reads them
back), rather than a new KB collection: :class:`~src.store.base.Store` already promises "a run
record's shape beyond ``repo``/``number`` is caller-defined" (its own module docstring), so a
cost record is just another kind of run.

Attribution (``agent``/``run_id``/``loop``) can't be captured *inside* ``llm.py`` itself — it
has no idea which of the ~10 agent modules is calling it, or which orchestrator stage the call
belongs to. Rather than threading a store + these three fields through every one of those call
sites, callers wrap a unit of work in ``llm.cost_context(sink)`` (see ``src/llm.py``) with a
``sink`` built from :func:`record_cost`; every ``llm.complete``/``complete_detailed`` call made
anywhere in that dynamic scope — including deep inside a helper this module never imports — gets
recorded automatically. ``src/orchestrator.py``'s real ``_intel``/``_contribution`` stages are
wired this way.

:func:`write_cost_record` is the one write primitive both paths above eventually funnel through
— but it also has a second, direct caller: :mod:`src.cost_ccusage` (T4.9) writes ``stage="cost"``
records straight from a parsed ``ccusage session --json`` payload, entirely outside
``llm.cost_context``/:func:`record_cost` (a Claude Code CLI session bypasses ``llm.py``
entirely — see that module's own docstring). So not every ``stage="cost"`` record traces back
to an ``llm.complete`` call; some are ingested session summaries tagged
``agent="claude_code_session"``.

Known limitations (disclosed, not fixed here — matches this module's own size to what T4.8
asks for):

- A call that raises ``LLMError`` *before* a provider ever replies (missing credential, bad
  ``LLM_PROVIDER``, transport/CLI failure) never reaches ``llm.py``'s sink hook, so nothing was
  billed and there is genuinely nothing to record. A JSON-mode reply that fails to *parse*,
  though, is captured: ``llm.py`` fires the sink right after the provider call returns, before
  attempting to parse the reply — a malformed JSON reply (a realistic failure every real agent
  already handles per-item, e.g. ``src/agents/analyst.py``) still gets its already-spent cost
  recorded even though the call itself still raises ``LLMError`` to its caller.
- ``CallMeta`` doesn't track prompt-cache token counts (T0.7 never asked a provider for them);
  every record's ``tokens_cache`` is always ``0``, not an estimate.
- Cache-token pricing aside, every non-``local`` model's cost is exactly ``meta.cost_usd`` —
  ``llm.py`` already prices ``claude_cli`` (provider-reported, exact) and ``claude_api``
  (estimated from its own maintained table); duplicating that table here would just be a
  second copy to keep in sync. The *only* thing this module's own pricing adds is
  :data:`LOCAL_MODEL_PRICES_PER_MTOK_ENV` for ``local`` calls, which ``llm.py`` always prices
  at ``$0.0`` (no metered per-token charge for a self-hosted endpoint).
- Of ``llm.complete``/``complete_detailed`` calls specifically (as opposed to
  :mod:`src.cost_ccusage`'s separate ingest path — see above), only ones made *inside* an
  active ``llm.cost_context`` scope are recorded. Today that's exactly the calls
  ``src/orchestrator.py``'s real ``_intel``/``_contribution`` stages make (analyst, forecaster,
  reporter, scout) — modules invoked outside the orchestrator tick (e.g. ``agents/grader.py``,
  ``agents/novelty.py``, ``agents/curator.py``, ``rag_eval.py``) make their own
  ``llm.complete`` calls with no cost context installed, so those calls go completely
  unrecorded, silently, until something wraps them too.
- ``_cost_sink`` is a :class:`contextvars.ContextVar`, which propagates into ``asyncio`` tasks
  but *not* into a manually spawned :class:`threading.Thread` (only into an explicit
  ``contextvars.copy_context().run(...)``). No agent spawns threads today, so this is latent,
  not live — but a future thread-parallelized agent would need to carry the context across
  explicitly, or its calls would silently go unrecorded rather than raise.
- A ``stage="cost"`` record has no ``status`` field, unlike every other record in this same
  ``runs`` collection (``"started"``/``"heartbeat"``/``"ok"``/``"failed"`` from
  :mod:`src.orchestrator`, or a repro/verify/self-review outcome from :mod:`src.stages`). A
  future reader that enumerates *distinct* ``stage`` values from ``store.list_runs()`` (e.g. a
  T5.8 health panel) rather than the fixed ``collect``/``intel``/``contribution`` set from
  :func:`~src.orchestrator._real_stages` must skip or special-case ``"cost"`` — feeding it
  through :func:`~src.liveness.latest_status` as if it were a fourth orchestrator stage would
  read as permanently ``"stalled"`` (no ``status`` key to classify).
- :func:`rollup` aggregates whatever is already in the KB when it's called — there is no cron
  entry wiring it to run daily yet (T5.8's ops dashboard is the natural first real caller). Its
  per-``(date, agent, model)`` bucket's ``provider`` is whichever record is encountered first
  for that key; if the same model name were ever served through two different providers for the
  same agent on the same day, the bucket's ``provider`` field wouldn't reflect that split (the
  summed ``cost_usd``/token counts would still be correct either way).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import TypedDict

from . import llm
from .store.base import Store

# This codebase's shared `recorded_at` convention for the run-event stream (matches
# `src.orchestrator._TS_FORMAT`, which this module can't import directly -- orchestrator.py
# imports `cost`, so importing back would cycle). `src.cost_ccusage` (no such cycle) imports
# this constant instead of re-declaring its own copy of the literal.
RECORDED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class CostRecord(TypedDict):
    """The shape every ``stage="cost"`` record shares, whether built from an ``llm`` call's
    :class:`~src.llm.CallMeta` (:func:`record_cost`) or an ingested ``ccusage`` session/model
    row (:mod:`src.cost_ccusage`). Annotating both call sites' dict literals with this catches a
    typo'd/missing/mistyped key at mypy time instead of only surfacing as a malformed record
    silently accepted by :func:`write_cost_record`."""

    stage: str
    agent: str
    run_id: str
    loop: str | None
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    tokens_cache: int
    cost_usd: float
    recorded_at: str


# JSON object of {"served-model-name": [price_in_per_mtok, price_out_per_mtok], ...} in USD.
# Self-hosted `local` calls have no metered per-token charge, so `llm.py` always reports
# `cost_usd=0.0` for them (see its own `_run_local`) -- correct by default, since there is no
# real invoice to reconcile against. This table is how an operator who *wants* the roll-up to
# reflect an amortized/opportunity cost for a specific served model opts in, model by model; an
# unlisted local model still costs $0.0, exactly like before this env var is ever set.
LOCAL_MODEL_PRICES_PER_MTOK_ENV = "LOCAL_MODEL_PRICES_PER_MTOK"


def _local_price_table() -> dict[str, tuple[float, float]]:
    raw = os.getenv(LOCAL_MODEL_PRICES_PER_MTOK_ENV)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    table: dict[str, tuple[float, float]] = {}
    for model, prices in parsed.items():
        if not isinstance(prices, list) or len(prices) != 2:
            continue
        try:
            table[str(model)] = (float(prices[0]), float(prices[1]))
        except (TypeError, ValueError):
            continue
    return table


def _priced_cost_usd(meta: llm.CallMeta) -> float:
    """`meta.cost_usd`, except a `local` call whose model has an explicit entry in
    :data:`LOCAL_MODEL_PRICES_PER_MTOK_ENV` (see module docstring)."""
    if meta.provider != "local":
        return meta.cost_usd
    rate = _local_price_table().get(meta.model)
    if rate is None:
        return meta.cost_usd
    price_in, price_out = rate
    return meta.prompt_tokens / 1e6 * price_in + meta.completion_tokens / 1e6 * price_out


def write_cost_record(store: Store, record: CostRecord) -> None:
    """Append `record` (already shaped for the cost run-event stream) via ``store.record_run``,
    best-effort — a KB write hiccup must never crash a caller that already has its (billed)
    cost data. Shared by :func:`record_cost` (T4.8, one ``llm`` call) and
    :mod:`src.cost_ccusage` (T4.9, one ``ccusage`` session/model row) — both build their own
    ``record`` dict from a different source, then hand it to this one write primitive.
    """
    try:
        # `dict(record)`: `Store.record_run` takes a plain `dict` -- mypy treats a TypedDict as
        # a distinct, non-`dict`-compatible type for assignment purposes even though it's a
        # real dict at runtime, so this copy is purely to satisfy that, not a behavior change.
        store.record_run(dict(record))
    except Exception as exc:
        agent = record.get("agent")
        print(f"cost: failed to record cost for agent={agent!r}: {exc}", file=sys.stderr)


def record_cost(
    store: Store,
    *,
    agent: str,
    run_id: str,
    loop: str | None,
    meta: llm.CallMeta,
) -> None:
    """Persist one cost record for a completed ``llm`` call, best-effort — a KB write hiccup
    must never turn into a crash for the agent that already got its (billed) LLM reply.

    The whole body is guarded, not just :func:`write_cost_record`'s own ``store.record_run``
    call: ``llm.cost_context``'s contract is that ``sink`` (built from this function) must
    never raise — a bug in ``_priced_cost_usd`` or the record's construction must degrade the
    same way a KB-write failure does, not propagate back into ``llm.py`` and fail an
    otherwise-successful call.
    """
    try:
        record: CostRecord = {
            "stage": "cost",
            "agent": agent,
            "run_id": run_id,
            "loop": loop,
            "provider": meta.provider,
            "model": meta.model,
            "tokens_in": meta.prompt_tokens,
            "tokens_out": meta.completion_tokens,
            "tokens_cache": 0,  # see module docstring's Known limitations
            "cost_usd": _priced_cost_usd(meta),
            "recorded_at": datetime.now(timezone.utc).strftime(RECORDED_AT_FORMAT),
        }
    except Exception as exc:
        print(f"cost: failed to record cost for agent={agent!r}: {exc}", file=sys.stderr)
        return
    write_cost_record(store, record)


def agent_context(store: Store, *, agent: str, loop: str, run_id: str):
    """`llm.cost_context(sink)` pre-wired to :func:`record_cost` for one agent/loop/run_id —
    every `llm.complete`/`complete_detailed` call made inside the returned ``with`` block gets
    a cost record tagged accordingly, with no change needed at the call site itself. Pulled out
    here (not left as a closure in the one caller that needs it today,
    `src/orchestrator.py`'s `_intel`/`_contribution`) so a second caller — a later stage, or
    T4.9's session ingest — doesn't have to re-derive the same one-line wrapper.
    """
    return llm.cost_context(
        lambda meta: record_cost(store, agent=agent, run_id=run_id, loop=loop, meta=meta)
    )


def rollup(store: Store) -> list[dict]:
    """Aggregate every recorded cost by (day, agent, model): one dict per bucket with
    ``date``/``agent``/``model``/``provider``/``calls``/``tokens_in``/``tokens_out``/
    ``cost_usd``, sorted by ``date`` then ``agent`` then ``model``.

    A record with a missing/unparseable ``recorded_at`` buckets under the literal date
    ``"unknown"`` rather than being dropped — a KB write from a corrupted or partial line
    should still show up *somewhere* in a spend report, not vanish silently.
    """
    buckets: dict[tuple[str, str, str], dict] = {}
    for run in store.list_runs(stage="cost"):
        recorded_at = run.get("recorded_at") or ""
        date = recorded_at[:10] if len(recorded_at) >= 10 else "unknown"
        agent = run.get("agent") or "unknown"
        model = run.get("model") or "unknown"
        key = (date, agent, model)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = {
                "date": date,
                "agent": agent,
                "model": model,
                "provider": run.get("provider") or "unknown",
                "calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
            }
            buckets[key] = bucket
        bucket["calls"] += 1
        bucket["tokens_in"] += int(run.get("tokens_in") or 0)
        bucket["tokens_out"] += int(run.get("tokens_out") or 0)
        bucket["cost_usd"] += float(run.get("cost_usd") or 0.0)

    return sorted(buckets.values(), key=lambda b: (b["date"], b["agent"], b["model"]))
