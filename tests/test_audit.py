"""Tests for the data-quality guardrail (T0.11) — offline & deterministic.

Per the DEVPLAN todo: synthetic local vs remote → count delta computed and gap ratio flagged
over threshold. Also covers the pure gap-ratio math, the persisted ``data_quality`` record
(written to a subdir so it isn't ingested as items), and the injected-remote glue. The live
GraphQL compare against a small repo is a separate ``@pytest.mark.integration`` smoke.
"""

import json

import pytest

from src import audit, config
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m0


def _item(repo, number, type="issue"):
    return {
        "repo": repo,
        "number": number,
        "type": type,
        "title": f"#{number}",
        "state": "open",
        "labels": [],
        "updated_at": "2025-01-01T00:00:00Z",
        "url": f"http://x/{repo}/{number}",
        "body": "",
    }


# ------------------------------------------------------------------- gap_ratio


def test_gap_ratio_contiguous_is_zero():
    assert audit.gap_ratio([1, 2, 3, 4, 5]) == 0.0


def test_gap_ratio_counts_missing_over_span():
    # range 1..10 = span 10; present {1,5,10} = 3 → 7 missing → 0.7
    assert audit.gap_ratio([1, 5, 10]) == pytest.approx(0.7)


def test_gap_ratio_ignores_dupes_and_order_and_short_input():
    assert audit.gap_ratio([3, 1, 2, 2, 1]) == 0.0  # unique 1..3, contiguous
    assert audit.gap_ratio([7]) == 0.0  # < 2 numbers → no range
    assert audit.gap_ratio([]) == 0.0


# -------------------------------------------------------------------- reconcile


def test_reconcile_ok_when_matched_and_contiguous():
    local = {"total": 3, "numbers": [1, 2, 3]}
    rec = audit.reconcile("o/r", local, {"total": 3})
    assert rec["count_delta"] == 0
    assert rec["gap_ratio"] == 0.0
    assert rec["flagged"] is False


def test_reconcile_flags_negative_delta():
    # local trails remote → probable silent loss → flagged even with no gaps.
    local = {"total": 8, "numbers": [1, 2, 3, 4, 5, 6, 7, 8]}
    rec = audit.reconcile("o/r", local, {"total": 10})
    assert rec["count_delta"] == -2
    assert rec["flagged"] is True


def test_reconcile_flags_gap_ratio_over_threshold():
    local = {"total": 3, "numbers": [1, 5, 10]}  # gap ratio 0.7
    rec = audit.reconcile("o/r", local, {"total": 3}, gap_threshold=0.05)
    assert rec["gap_ratio"] == pytest.approx(0.7)
    assert rec["flagged"] is True


def test_reconcile_does_not_flag_gap_under_threshold():
    local = {"total": 9, "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 10]}  # 1 missing in span 10 → 0.1
    rec = audit.reconcile("o/r", local, {"total": 9}, gap_threshold=0.2)
    assert rec["gap_ratio"] == pytest.approx(0.1)
    assert rec["flagged"] is False


def test_reconcile_flags_collection_errors():
    local = {"total": 5, "numbers": [1, 2, 3, 4, 5]}
    rec = audit.reconcile("o/r", local, {"total": 5}, errors=1)
    assert rec["error_count"] == 1
    assert rec["flagged"] is True


def test_reconcile_uses_config_threshold_by_default(monkeypatch):
    monkeypatch.setattr(config, "DATA_QUALITY_GAP_RATIO_THRESHOLD", 0.9)
    local = {"total": 3, "numbers": [1, 5, 10]}  # 0.7 ratio, under the raised 0.9 threshold
    assert audit.reconcile("o/r", local, {"total": 3})["flagged"] is False


# ------------------------------------------------------------------ local_counts


def test_local_counts_windowed_by_since(tmp_path):
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1) | {"updated_at": "2025-06-05T00:00:00Z"},  # in window
            _item("o/r", 2) | {"updated_at": "2025-06-06T00:00:00Z"},  # in window
            _item("o/r", 3) | {"updated_at": "2024-01-01T00:00:00Z"},  # before since → excluded
        ]
    )
    counts = audit.local_counts(store, "o/r", "2025-06-01T00:00:00Z")
    # total and numbers both come from the same windowed set (the 2024 item is out).
    assert counts == {"total": 2, "numbers": [1, 2]}


# ------------------------------------------------------- audit_repo (glue + sink)


def test_audit_repo_writes_data_quality_record(tmp_path):
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1), _item("o/r", 2), _item("o/r", 3)])

    # Inject a remote fetcher so no network is touched.
    def fake_remote(repo, since):
        assert repo == "o/r"
        return {"issues": 3, "prs": 0, "total": 5}  # remote says 5 → delta -2 → flagged

    rec = audit.audit_repo(
        store,
        "o/r",
        "2025-01-01T00:00:00Z",
        remote_fetcher=fake_remote,
        checked_at="2026-07-04T00:00:00Z",
    )
    assert rec["local_total"] == 3
    assert rec["remote_total"] == 5
    assert rec["count_delta"] == -2
    assert rec["flagged"] is True

    # Record persisted to the audit subdir (NOT the data-dir root, so query() ignores it).
    sink = tmp_path / "audit" / "data_quality.jsonl"
    assert sink.exists()
    written = [json.loads(line) for line in sink.read_text().splitlines() if line.strip()]
    assert len(written) == 1
    assert written[0]["repo"] == "o/r" and written[0]["checked_at"] == "2026-07-04T00:00:00Z"


def test_data_quality_sink_is_not_ingested_as_items(tmp_path):
    # The audit record must not pollute the item store: query() globs *.jsonl in the ROOT only.
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1)])
    audit.audit_repo(
        store,
        "o/r",
        "2025-01-01T00:00:00Z",
        remote_fetcher=lambda r, s: {"issues": 1, "prs": 0, "total": 1},
        checked_at="2026-07-04T00:00:00Z",
    )
    assert [it["number"] for it in store.query()] == [1]  # only the real item, not the record


def test_audit_repo_appends_across_runs(tmp_path):
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1)])
    fake = lambda r, s: {"issues": 1, "prs": 0, "total": 1}  # noqa: E731
    for stamp in ("2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"):
        audit.audit_repo(
            store, "o/r", "2025-01-01T00:00:00Z", remote_fetcher=fake, checked_at=stamp
        )
    sink = tmp_path / "audit" / "data_quality.jsonl"
    lines = [line for line in sink.read_text().splitlines() if line.strip()]
    assert len(lines) == 2  # one record appended per run (history retained)


# ---------------------------------------------------- record_stall (T0.12 follow-up)


def test_stall_recorded(tmp_path):
    store = JsonlStore(tmp_path)
    fake_remote = lambda repo, since: {"total": 1234}  # noqa: E731

    rec = audit.record_stall(
        store,
        "o/r",
        "2025-06-01T00:00:00Z",
        remote_fetcher=fake_remote,
        checked_at="2026-07-04T00:00:00Z",
    )
    assert rec == {
        "repo": "o/r",
        "reason": "cursor_stall",
        "window_since": "2025-06-01T00:00:00Z",
        "remote_total": 1234,
        "flagged": True,
    }

    # Persisted to the same audit sink as reconcile() records (subdir, so query() ignores it).
    sink = tmp_path / "audit" / "data_quality.jsonl"
    written = [json.loads(line) for line in sink.read_text().splitlines() if line.strip()]
    assert len(written) == 1
    assert written[0]["reason"] == "cursor_stall"
    assert written[0]["checked_at"] == "2026-07-04T00:00:00Z"


def test_stall_recorded_requires_data_dir_for_non_jsonl_store(tmp_path):
    class NoDirStore:
        pass  # no `data_dir` attribute — mimics a future non-JSONL Store

    with pytest.raises(TypeError):
        audit.record_stall(
            NoDirStore(),
            "o/r",
            "2025-06-01T00:00:00Z",
            remote_fetcher=lambda repo, since: {"total": 1},
            checked_at="2026-07-04T00:00:00Z",
        )


# ------------------------------------------------------------- live GraphQL smoke


@pytest.mark.integration
def test_remote_counts_live_graphql():
    # Small, stable repo; just assert the shape + non-negative count (needs GITHUB_TOKEN).
    counts = audit.remote_counts("llm-d/llm-d", "2026-06-01T00:00:00Z")
    assert set(counts) == {"total"}
    assert counts["total"] >= 0
