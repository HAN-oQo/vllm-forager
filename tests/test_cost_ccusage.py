"""Tests for the Claude Code session cost ingest (T4.9) — offline & deterministic.

Per the DEVPLAN todo: a fixture `ccusage session --json` payload (shaped exactly like a real
`npx ccusage@latest session --json` run) ingests into session cost rows with the expected
totals, offline (no `npx`/network call anywhere here).
"""

from __future__ import annotations

import pytest

from src import cost, cost_ccusage
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m4


@pytest.fixture
def store(tmp_path) -> JsonlStore:
    return JsonlStore(tmp_path)


# A trimmed, real-shaped fixture -- one single-model session, one multi-model session, and one
# non-Claude-Code session (a different coding CLI ccusage also tracks on a shared machine).
_FIXTURE = {
    "session": [
        {
            "agent": "claude",
            "period": "0002f01d-6bd9-4d33-a52d-a71d4e788a5e",
            "metadata": {"lastActivity": "2026-07-07T00:29:10.194Z"},
            "modelBreakdowns": [
                {
                    "modelName": "claude-sonnet-5",
                    "inputTokens": 1,
                    "outputTokens": 175,
                    "cacheCreationTokens": 8085,
                    "cacheReadTokens": 29439,
                    "cost": 0.0399798,
                }
            ],
        },
        {
            "agent": "claude",
            "period": "0d781670-afc1-425c-a140-e50970bf3c0b",
            "metadata": {"lastActivity": "2026-07-04T09:02:24.996Z"},
            "modelBreakdowns": [
                {
                    "modelName": "claude-opus-4-8",
                    "inputTokens": 126054,
                    "outputTokens": 394219,
                    "cacheCreationTokens": 1578870,
                    "cacheReadTokens": 67535573,
                    "cost": 57.537539,
                },
                {
                    "modelName": "claude-sonnet-5",
                    "inputTokens": 961,
                    "outputTokens": 346799,
                    "cacheCreationTokens": 1309860,
                    "cacheReadTokens": 191044619,
                    "cost": 45.9696998,
                },
            ],
        },
        {
            "agent": "codex",  # a different coding CLI -- must be skipped, not misattributed
            "period": "ffffffff-0000-0000-0000-000000000000",
            "metadata": {"lastActivity": "2026-07-06T00:00:00.000Z"},
            "modelBreakdowns": [
                {
                    "modelName": "gpt-5",
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "cacheCreationTokens": 0,
                    "cacheReadTokens": 0,
                    "cost": 0.5,
                }
            ],
        },
    ],
    "totals": {"totalCost": 103.547219},  # present in real output; ingest() never reads this
}


def test_ingest_writes_one_record_per_session_model_breakdown(store: JsonlStore) -> None:
    written = cost_ccusage.ingest(store, _FIXTURE, loop="dev-loop")

    assert written == 3  # 1 (session 1) + 2 (session 2) -- the codex session is skipped
    runs = store.list_runs(stage="cost")
    assert len(runs) == 3
    assert {r["model"] for r in runs} == {"claude-sonnet-5", "claude-opus-4-8"}
    assert all(r["agent"] == "claude_code_session" for r in runs)
    assert all(r["provider"] == "claude_cli" for r in runs)
    assert all(r["loop"] == "dev-loop" for r in runs)


def test_ingest_skips_non_claude_code_agent_rows(store: JsonlStore) -> None:
    cost_ccusage.ingest(store, _FIXTURE)

    models = {r["model"] for r in store.list_runs(stage="cost")}
    assert "gpt-5" not in models


def test_ingest_shares_one_run_id_across_a_sessions_model_breakdowns(store: JsonlStore) -> None:
    cost_ccusage.ingest(store, _FIXTURE)

    multi_model_runs = [
        r
        for r in store.list_runs(stage="cost")
        if r["run_id"] == "0d781670-afc1-425c-a140-e50970bf3c0b"
    ]
    assert len(multi_model_runs) == 2
    assert {r["model"] for r in multi_model_runs} == {"claude-opus-4-8", "claude-sonnet-5"}


def test_ingest_sums_cache_creation_and_read_tokens_into_tokens_cache(store: JsonlStore) -> None:
    cost_ccusage.ingest(store, _FIXTURE)

    record = next(
        r
        for r in store.list_runs(stage="cost")
        if r["run_id"] == "0002f01d-6bd9-4d33-a52d-a71d4e788a5e"
    )
    assert record["tokens_cache"] == 8085 + 29439
    assert record["tokens_in"] == 1
    assert record["tokens_out"] == 175
    assert record["cost_usd"] == pytest.approx(0.0399798)


def test_ingest_normalizes_the_iso_millis_timestamp(store: JsonlStore) -> None:
    cost_ccusage.ingest(store, _FIXTURE)

    record = next(
        r
        for r in store.list_runs(stage="cost")
        if r["run_id"] == "0002f01d-6bd9-4d33-a52d-a71d4e788a5e"
    )
    assert record["recorded_at"] == "2026-07-07T00:29:10Z"


def test_ingest_falls_back_to_the_raw_string_for_an_unparseable_timestamp(
    store: JsonlStore,
) -> None:
    fixture = {
        "session": [
            {
                "agent": "claude",
                "period": "r-1",
                "metadata": {"lastActivity": "not-a-timestamp"},
                "modelBreakdowns": [
                    {
                        "modelName": "claude-sonnet-5",
                        "inputTokens": 1,
                        "outputTokens": 1,
                        "cacheCreationTokens": 0,
                        "cacheReadTokens": 0,
                        "cost": 0.01,
                    }
                ],
            }
        ]
    }

    cost_ccusage.ingest(store, fixture)

    assert store.list_runs(stage="cost")[0]["recorded_at"] == "not-a-timestamp"


def test_ingest_with_no_sessions_writes_nothing(store: JsonlStore) -> None:
    assert cost_ccusage.ingest(store, {"session": []}) == 0
    assert store.list_runs(stage="cost") == []


def test_ingest_survives_a_record_run_failure(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        store, "record_run", lambda _r: (_ for _ in ()).throw(RuntimeError("disk full"))
    )

    written = cost_ccusage.ingest(store, _FIXTURE)  # must not raise

    assert written == 3  # ingest() still counts attempted writes even if each one failed
    assert "failed to record cost" in capsys.readouterr().err


def test_ingest_then_rollup_produces_the_expected_totals(store: JsonlStore) -> None:
    """DEVPLAN's own worked example: a fixture ccusage JSON -> session cost rows with the
    expected totals, read back through the same rollup() T4.8 already exercises."""
    cost_ccusage.ingest(store, _FIXTURE, loop="dev-loop")

    buckets = {(b["date"], b["agent"], b["model"]): b for b in cost.rollup(store)}

    sonnet_bucket = buckets[("2026-07-07", "claude_code_session", "claude-sonnet-5")]
    assert sonnet_bucket["calls"] == 1
    assert sonnet_bucket["cost_usd"] == pytest.approx(0.0399798)

    opus_bucket = buckets[("2026-07-04", "claude_code_session", "claude-opus-4-8")]
    assert opus_bucket["cost_usd"] == pytest.approx(57.537539)

    # The second session's sonnet-5 row lands on a different day than the first session's --
    # both must be summed separately, not merged into one (date, agent, model) bucket.
    second_sonnet_bucket = buckets[("2026-07-04", "claude_code_session", "claude-sonnet-5")]
    assert second_sonnet_bucket["cost_usd"] == pytest.approx(45.9696998)
