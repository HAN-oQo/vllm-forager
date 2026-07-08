"""Tests for per-agent cost capture (T4.8) — offline & deterministic.

Per the DEVPLAN todo: mocked calls across agents produce correct per-agent / per-day sums via
:func:`src.cost.rollup`, and a local-vLLM model is priced via the custom table
(:data:`src.cost.LOCAL_MODEL_PRICES_PER_MTOK_ENV`) rather than the ``$0.0``
:mod:`src.llm` always reports for a self-hosted call.
"""

from __future__ import annotations

import json

import pytest

from src import cost, llm
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m4


@pytest.fixture
def store(tmp_path) -> JsonlStore:
    return JsonlStore(tmp_path)


def _meta(**overrides) -> llm.CallMeta:
    fields = {
        "provider": "claude_cli",
        "model": "claude-sonnet-5",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "latency_s": 1.2,
        "cost_usd": 0.0021,
    }
    fields.update(overrides)
    return llm.CallMeta(**fields)


# --------------------------------------------------------------------- record_cost


def test_record_cost_writes_a_well_formed_record(store: JsonlStore) -> None:
    cost.record_cost(store, agent="analyst", run_id="r-1", loop="intel", meta=_meta())

    runs = store.list_runs(stage="cost")
    assert len(runs) == 1
    record = runs[0]
    assert record["agent"] == "analyst"
    assert record["run_id"] == "r-1"
    assert record["loop"] == "intel"
    assert record["provider"] == "claude_cli"
    assert record["model"] == "claude-sonnet-5"
    assert record["tokens_in"] == 100
    assert record["tokens_out"] == 50
    assert record["tokens_cache"] == 0
    assert record["cost_usd"] == 0.0021
    assert record["recorded_at"]


def test_record_cost_survives_a_record_run_failure(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        store, "record_run", lambda _r: (_ for _ in ()).throw(RuntimeError("disk full"))
    )

    cost.record_cost(store, agent="analyst", run_id="r-1", loop="intel", meta=_meta())  # no raise

    assert "failed to record cost" in capsys.readouterr().err


def test_record_cost_prices_a_local_model_via_the_custom_table(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        cost.LOCAL_MODEL_PRICES_PER_MTOK_ENV,
        json.dumps({"my-local-model": [1.0, 2.0]}),
    )
    meta = _meta(
        provider="local",
        model="my-local-model",
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        cost_usd=0.0,  # llm.py always reports 0.0 for local -- see its own _run_local
    )

    cost.record_cost(store, agent="analyst", run_id="r-1", loop="intel", meta=meta)

    record = store.list_runs(stage="cost")[0]
    assert record["cost_usd"] == pytest.approx(1.0 + 1.0)  # 1M in @ $1 + 0.5M out @ $2/Mtok


def test_record_cost_unpriced_local_model_stays_free(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(cost.LOCAL_MODEL_PRICES_PER_MTOK_ENV, raising=False)
    meta = _meta(provider="local", model="unlisted-model", cost_usd=0.0)

    cost.record_cost(store, agent="analyst", run_id="r-1", loop="intel", meta=meta)

    assert store.list_runs(stage="cost")[0]["cost_usd"] == 0.0


def test_record_cost_ignores_a_malformed_local_price_table(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cost.LOCAL_MODEL_PRICES_PER_MTOK_ENV, "not json")
    meta = _meta(provider="local", model="unlisted-model", cost_usd=0.0)

    cost.record_cost(store, agent="analyst", run_id="r-1", loop="intel", meta=meta)  # no raise

    assert store.list_runs(stage="cost")[0]["cost_usd"] == 0.0


# --------------------------------------------------------------------- rollup


def test_rollup_sums_calls_and_cost_per_day_agent_and_model(store: JsonlStore) -> None:
    """Mocked calls across agents -> correct per-agent / per-day sums (DEVPLAN's own worked
    example). Written directly via store.record_run (not record_cost, which always stamps the
    real wall clock) so two records can land on different, fixed days."""
    store.record_run(
        {
            "stage": "cost",
            "agent": "analyst",
            "run_id": "r-1",
            "loop": "intel",
            "provider": "claude_cli",
            "model": "claude-sonnet-5",
            "tokens_in": 100,
            "tokens_out": 50,
            "tokens_cache": 0,
            "cost_usd": 0.01,
            "recorded_at": "2026-01-08T09:00:00Z",
        }
    )
    store.record_run(
        {
            "stage": "cost",
            "agent": "analyst",
            "run_id": "r-2",
            "loop": "intel",
            "provider": "claude_cli",
            "model": "claude-sonnet-5",
            "tokens_in": 200,
            "tokens_out": 100,
            "tokens_cache": 0,
            "cost_usd": 0.02,
            "recorded_at": "2026-01-08T15:00:00Z",  # same day as r-1
        }
    )
    store.record_run(
        {
            "stage": "cost",
            "agent": "scout",
            "run_id": "r-3",
            "loop": "contribution",
            "provider": "claude_cli",
            "model": "claude-sonnet-5",
            "tokens_in": 30,
            "tokens_out": 10,
            "tokens_cache": 0,
            "cost_usd": 0.003,
            "recorded_at": "2026-01-08T10:00:00Z",  # same day, different agent
        }
    )
    store.record_run(
        {
            "stage": "cost",
            "agent": "analyst",
            "run_id": "r-4",
            "loop": "intel",
            "provider": "claude_cli",
            "model": "claude-sonnet-5",
            "tokens_in": 40,
            "tokens_out": 20,
            "tokens_cache": 0,
            "cost_usd": 0.004,
            "recorded_at": "2026-01-09T09:00:00Z",  # next day
        }
    )
    # A non-"cost" run (e.g. a T4.4 orchestrator tick event) must never leak into the rollup.
    store.record_run({"stage": "collect", "status": "ok", "recorded_at": "2026-01-08T09:00:00Z"})

    buckets = cost.rollup(store)

    assert buckets == [
        {
            "date": "2026-01-08",
            "agent": "analyst",
            "model": "claude-sonnet-5",
            "provider": "claude_cli",
            "calls": 2,
            "tokens_in": 300,
            "tokens_out": 150,
            "cost_usd": pytest.approx(0.03),
        },
        {
            "date": "2026-01-08",
            "agent": "scout",
            "model": "claude-sonnet-5",
            "provider": "claude_cli",
            "calls": 1,
            "tokens_in": 30,
            "tokens_out": 10,
            "cost_usd": pytest.approx(0.003),
        },
        {
            "date": "2026-01-09",
            "agent": "analyst",
            "model": "claude-sonnet-5",
            "provider": "claude_cli",
            "calls": 1,
            "tokens_in": 40,
            "tokens_out": 20,
            "cost_usd": pytest.approx(0.004),
        },
    ]


def test_rollup_with_no_cost_records_is_empty(store: JsonlStore) -> None:
    assert cost.rollup(store) == []


def test_rollup_buckets_an_unparseable_recorded_at_as_unknown_date(store: JsonlStore) -> None:
    store.record_run(
        {
            "stage": "cost",
            "agent": "analyst",
            "model": "claude-sonnet-5",
            "cost_usd": 0.01,
            "recorded_at": "",
        }
    )

    buckets = cost.rollup(store)

    assert buckets[0]["date"] == "unknown"


# --------------------------------------------------------------------- llm.cost_context wiring


def test_cost_context_sink_receives_the_calls_meta_from_complete(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: llm.complete under llm.cost_context() -> cost.record_cost() -> a readable
    cost record, with no change needed at llm.complete's own call site."""
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "my-model")

    class _FakeResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
                "model": "my-model",
            }

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _FakeResp())

    with llm.cost_context(
        lambda meta: cost.record_cost(store, agent="analyst", run_id="r-1", loop="intel", meta=meta)
    ):
        llm.complete("hello", provider="local")

    runs = store.list_runs(stage="cost")
    assert len(runs) == 1
    assert runs[0]["agent"] == "analyst"
    assert runs[0]["tokens_in"] == 4
    assert runs[0]["tokens_out"] == 6


def test_cost_context_does_not_leak_outside_its_own_scope(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm:8000/v1")

    class _FakeResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "hi"}}]}

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _FakeResp())

    with llm.cost_context(
        lambda meta: cost.record_cost(store, agent="analyst", run_id="r-1", loop="intel", meta=meta)
    ):
        pass  # sink installed, but no call made inside the scope

    llm.complete("hello", provider="local")  # outside the scope -- must not be recorded

    assert store.list_runs(stage="cost") == []
