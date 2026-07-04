"""Tests for the Forecaster CLI (T1.5) — offline & deterministic.

``main()`` on a tmp store logs predictions for classified items and prints a count. Mirrors
``tests/test_analyze_cli.py``'s store-selection coverage.
"""

import pytest

from src import forecast, llm
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m1


def _item(repo: str, number: int, title: str, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": title,
        "state": "open",
        "labels": [],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "url": f"http://x/{repo}/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


def _reply() -> dict:
    return {
        "claim": "this will be merged",
        "resolution_rule": "resolved true if the PR is merged",
        "prob": 0.7,
        "due_date": "2099-01-01T00:00:00Z",
    }


def test_main_logs_predictions_for_classified_items(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "hipBLAS build fails", category="rocm-build")])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())

    rc = forecast.main(["--data-dir", str(tmp_path)])

    assert rc == 0
    assert capsys.readouterr().out.strip() == "logged 1 prediction(s)"


def test_main_no_classified_items_prints_zero(tmp_path, capsys) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "uncategorized")])
    rc = forecast.main(["--data-dir", str(tmp_path)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "logged 0 prediction(s)"
