"""Tests for the dashboard review pane (T5.6) — offline & deterministic; the LLM risk-score
call is mocked (per the DEVPLAN todo: "gate mocked").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard import review
from src import gate
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5


def _store_ready_for_gate(tmp_path: Path, *, repo: str = "o/r", number: int = 1) -> JsonlStore:
    """Mirrors `tests/test_gate.py::_store_ready_for_gate`'s own fixture shape -- a candidate
    with a reproduced repro run, a verified verify run, and a matching, advancing self-review."""
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
            "stage": "repro",
            "command": "pytest test_fp8.py",
            "log": "AssertionError\n",
            "reproduced": True,
            "recorded_at": "2025-12-30T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "verify",
            "branch": f"forager/{repo.replace('/', '-')}-{number}",
            "patch": "--- a/x.py\n+++ b/x.py\n",
            "log": "1 passed\n",
            "verified": True,
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
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
    return store


@pytest.fixture(autouse=True)
def _mock_risk_score(monkeypatch: pytest.MonkeyPatch):
    """Mirrors `tests/test_gate.py`'s own autouse fixture -- isolates every test from the real
    LLM call `gate._risk_badge` makes (scout's own `_score`)."""
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )


@pytest.fixture(autouse=True)
def _isolate_pr_drafts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Mirrors `tests/test_gate.py`'s own autouse fixture -- an approve decision writes a real
    PR-draft file; redirect the default drafts dir away from the real shared `config.DATA_DIR`."""
    monkeypatch.setattr(gate, "_PR_DRAFTS_DIR", tmp_path / "pr_drafts")


# --------------------------------------------------------------------- review_bundle


def test_review_bundle_returns_the_full_bundle_as_a_dict(tmp_path: Path) -> None:
    store = _store_ready_for_gate(tmp_path)

    bundle = review.review_bundle(store, "o/r", 1)

    assert bundle is not None
    assert bundle["diff"] == "--- a/x.py\n+++ b/x.py\n"
    assert bundle["risk"] == "low"
    assert bundle["repro_log"] == "AssertionError\n"
    assert bundle["verify_log"] == "1 passed\n"


def test_review_bundle_returns_none_when_not_gate_ready(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert review.review_bundle(store, "o/r", 1) is None


# --------------------------------------------------------------------- review_decision


def test_approve_flips_candidate_status_and_writes_a_draft(tmp_path: Path) -> None:
    store = _store_ready_for_gate(tmp_path)

    result = review.review_decision(store, "o/r", 1, approve=True)

    assert result is not None
    assert result["approved"] is True
    assert result["submitted"] is False  # submit is never exposed here
    assert result["draft_path"] is not None
    assert isinstance(result["draft_path"], str)  # JSON-serializable, not a Path object

    runs = store.list_runs(repo="o/r", number=1, stage="gate")
    assert runs[-1]["approved"] is True


def test_hold_records_the_decision_without_writing_a_draft(tmp_path: Path) -> None:
    store = _store_ready_for_gate(tmp_path)

    result = review.review_decision(store, "o/r", 1, approve=False)

    assert result is not None
    assert result["approved"] is False
    assert result["draft_path"] is None

    runs = store.list_runs(repo="o/r", number=1, stage="gate")
    assert runs[-1]["approved"] is False


def test_review_decision_never_submits_even_if_a_prior_draft_exists(tmp_path: Path) -> None:
    """Regression: submit is hardcoded False in dashboard.review -- confirms a second approve
    call (which would satisfy gate.py's own "draft already exists" precondition for --submit)
    still never opens a real PR through this surface."""
    store = _store_ready_for_gate(tmp_path)
    review.review_decision(store, "o/r", 1, approve=True)

    result = review.review_decision(store, "o/r", 1, approve=True)

    assert result is not None
    assert result["submitted"] is False
    assert result["pr_url"] is None


def test_review_decision_returns_none_when_not_gate_ready(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    assert review.review_decision(store, "o/r", 1, approve=True) is None


def test_review_decision_uses_an_existing_composed_narrative(tmp_path: Path) -> None:
    """Regression: an earlier version always fell back to the raw bundle.format() dump, even
    when a maintainer-grade T3.9 narrative (passing T3.10 quality) already existed for the
    exact verify attempt being approved."""
    store = _store_ready_for_gate(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "[Bugfix] composed title",
            "body": "composed maintainer-grade body",
            "verify_recorded_at": "2025-12-31T00:00:00Z",
            "recorded_at": "2026-01-02T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_quality",
            "votes": [{"acceptable": True, "reason": "ok"}],
            "approve_count": 1,
            "total_votes": 1,
            "passes": True,
            "pr_author_recorded_at": "2026-01-02T00:00:00Z",
            "recorded_at": "2026-01-03T00:00:00Z",
        }
    )

    result = review.review_decision(store, "o/r", 1, approve=True)

    assert result is not None
    assert result["quality_passed"] is True
    draft_content = Path(result["draft_path"]).read_text()
    assert draft_content == "composed maintainer-grade body"


def test_review_decision_falls_back_to_the_raw_bundle_without_a_narrative(tmp_path: Path) -> None:
    store = _store_ready_for_gate(tmp_path)

    result = review.review_decision(store, "o/r", 1, approve=True)

    assert result is not None
    assert result["quality_passed"] is None
    draft_content = Path(result["draft_path"]).read_text()
    assert "vLLM crashes on gfx90a with fp8" in draft_content  # bundle.format()'s own rendering


def test_review_decision_honors_a_custom_pr_drafts_dir(tmp_path: Path) -> None:
    """Regression: an earlier version never threaded pr_drafts_dir through, so every draft
    landed under gate's own module-level default regardless of the caller's own data dir --
    the exact bug class gate.py's own module docstring says already bit the CLI once."""
    store = _store_ready_for_gate(tmp_path)
    custom_dir = tmp_path / "custom_drafts"

    result = review.review_decision(store, "o/r", 1, approve=True, pr_drafts_dir=custom_dir)

    assert result is not None
    assert result["draft_path"].startswith(str(custom_dir))
    assert Path(result["draft_path"]).exists()
