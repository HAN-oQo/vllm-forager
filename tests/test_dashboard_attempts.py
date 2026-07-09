"""Tests for the dashboard Attempts tab (T5.11) — offline & deterministic.

Per the DEVPLAN todo: seeded attempt reports → the tab lists them with outcome + renders all
3 sections; a verified, gate-ready candidate's view also carries the T5.6 review bundle so a
human can go straight from browsing to approving.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard import attempts
from src import gate
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5


def _store_with_verify(
    tmp_path: Path, *, repo: str = "o/r", number: int = 1, verified: bool, recorded_at: str
) -> JsonlStore:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            {
                "repo": repo,
                "number": number,
                "type": "issue",
                "title": "vLLM crashes on gfx90a with fp8",
                "body": "Running fp8 quant on MI250 raises an assertion.",
                "state": "open",
                "url": f"https://github.com/{repo}/issues/{number}",
            }
        ]
    )
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "verify",
            "branch": f"forager/{repo.replace('/', '-')}-{number}",
            "patch": "--- a/x.py\n+++ b/x.py\n",
            "command": "pytest test_fp8.py",
            "log": "1 passed\n" if verified else "AssertionError\n",
            "verified": verified,
            "recorded_at": recorded_at,
        }
    )
    return store


def _make_gate_ready(store: JsonlStore, *, repo: str = "o/r", number: int = 1) -> None:
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "self_review",
            "critiques": [{"looks_correct": True, "reason": "ok"}],
            "approve_count": 4,
            "total_votes": 5,
            "advance": True,
            "verify_recorded_at": "2025-12-31T00:00:00Z",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )


@pytest.fixture(autouse=True)
def _mock_risk_score(monkeypatch: pytest.MonkeyPatch):
    """Mirrors `tests/test_dashboard_review.py`'s own autouse fixture -- isolates every test
    from the real LLM call `gate._risk_badge` makes."""
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )


# --------------------------------------------------------------------- list_attempts


def test_list_attempts_returns_worked_candidates_newest_first(tmp_path: Path) -> None:
    store = _store_with_verify(
        tmp_path, repo="o/r", number=1, verified=True, recorded_at="2025-12-31T00:00:00Z"
    )
    store.upsert_items(
        [
            {
                "repo": "o/r2",
                "number": 2,
                "type": "issue",
                "title": "second issue",
                "body": "",
                "state": "open",
                "url": "https://github.com/o/r2/issues/2",
            }
        ]
    )
    store.record_run(
        {
            "repo": "o/r2",
            "number": 2,
            "stage": "verify",
            "branch": "forager/o-r2-2",
            "patch": "",
            "command": "pytest",
            "log": "still failing\n",
            "verified": False,
            "recorded_at": "2026-06-01T00:00:00Z",
        }
    )

    rows = attempts.list_attempts(store)

    assert [(r["repo"], r["number"]) for r in rows] == [("o/r2", 2), ("o/r", 1)]
    assert rows[0]["verified"] is False
    assert rows[0]["title"] == "second issue"
    assert rows[1]["verified"] is True


def test_list_attempts_empty_when_nothing_worked(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert attempts.list_attempts(store) == []


# --------------------------------------------------------------------- open_attempt


def test_open_attempt_returns_none_when_not_worked(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert attempts.open_attempt(store, "o/r", 1) is None


def test_open_attempt_returns_full_report_for_a_verified_candidate(tmp_path: Path) -> None:
    store = _store_with_verify(tmp_path, verified=True, recorded_at="2025-12-31T00:00:00Z")

    result = attempts.open_attempt(store, "o/r", 1)

    assert result is not None
    assert result["verified"] is True
    assert "vLLM crashes on gfx90a with fp8" in result["issue_overview"]
    assert "```bash" in result["reproduce"]
    assert "VERIFIED" in result["outcome"]


def test_open_attempt_still_reports_a_failed_candidate(tmp_path: Path) -> None:
    store = _store_with_verify(tmp_path, verified=False, recorded_at="2025-12-31T00:00:00Z")

    result = attempts.open_attempt(store, "o/r", 1)

    assert result is not None
    assert result["verified"] is False
    assert "FAILED" in result["outcome"]
    assert result["review_bundle"] is None


def test_open_attempt_review_bundle_is_none_when_verified_but_not_gate_ready(
    tmp_path: Path,
) -> None:
    store = _store_with_verify(tmp_path, verified=True, recorded_at="2025-12-31T00:00:00Z")

    result = attempts.open_attempt(store, "o/r", 1)

    assert result is not None
    assert result["rendered_html"] is None
    assert result["review_bundle"] is None


def test_open_attempt_includes_review_bundle_for_a_gate_ready_candidate(
    tmp_path: Path,
) -> None:
    store = _store_with_verify(tmp_path, verified=True, recorded_at="2025-12-31T00:00:00Z")
    _make_gate_ready(store)

    result = attempts.open_attempt(store, "o/r", 1)

    assert result is not None
    assert result["rendered_html"] is not None
    assert result["review_bundle"] is not None
    assert result["review_bundle"]["risk"] == "low"
    assert result["review_bundle"]["diff"] == "--- a/x.py\n+++ b/x.py\n"
