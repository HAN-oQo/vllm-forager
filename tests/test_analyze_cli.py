"""Tests for the Analyst CLI (T1.4) — offline & deterministic.

``main()`` on a tmp store classifies pending items and prints a count. Mirrors
``tests/test_report_cli.py``'s store-selection coverage.
"""

import pytest

from src import analyze, llm
from src.store.jsonl_store import JsonlStore
from src.taxonomy import create_taxonomy

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


def test_main_classifies_pending_items(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, ["rocm-build"])
    store.upsert_items([_item("o/r", 1, "hipBLAS build fails on gfx90a")])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"category": "rocm-build"})

    rc = analyze.main(["--data-dir", str(tmp_path)])

    assert rc == 0
    assert capsys.readouterr().out.strip() == "classified 1 item(s)"
    assert store.query()[0]["category"] == "rocm-build"


def test_main_no_pending_items_prints_zero(tmp_path, capsys) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, ["rocm-build"])
    rc = analyze.main(["--data-dir", str(tmp_path)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "classified 0 item(s)"
