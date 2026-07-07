"""Tests for the Analyst CLI (T1.4) — offline & deterministic.

``main()`` on a tmp store classifies pending items and prints a count. Mirrors
``tests/test_report_cli.py``'s store-selection coverage.
"""

import pytest

from src import analyze, llm, parity
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


@pytest.fixture(autouse=True)
def _config_repos(monkeypatch: pytest.MonkeyPatch) -> None:
    # T3.15 roots every classified path at the item's own repo's config.REPOS domain -- pin a
    # fixture domain here rather than depending on production config.py's real content, so an
    # unrelated future edit to config.REPOS can't silently break this CLI test.
    monkeypatch.setattr(
        parity.config, "REPOS", [{"slug": "o/r", "role": "primary", "domain": "speech"}]
    )


def test_main_classifies_pending_items(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [["speech", "rocm-build"]])
    store.upsert_items([_item("o/r", 1, "hipBLAS build fails on gfx90a")])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": ["rocm-build"]})

    rc = analyze.main(["--data-dir", str(tmp_path)])

    assert rc == 0
    assert capsys.readouterr().out.strip() == "classified 1 item(s)"
    assert store.query()[0]["category"] == "speech > rocm-build"


def test_main_no_pending_items_prints_zero(tmp_path, capsys) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [["speech", "rocm-build"]])
    rc = analyze.main(["--data-dir", str(tmp_path)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "classified 0 item(s)"


def test_main_per_repo_limit_caps_classification(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """T3.19: `--per-repo-limit` bounds a dry run over a KB with many pending items."""
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [["speech", "rocm-build"]])
    store.upsert_items([_item("o/r", n, f"hipBLAS bug {n} on gfx90a") for n in range(1, 6)])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": ["rocm-build"]})

    rc = analyze.main(["--data-dir", str(tmp_path), "--per-repo-limit", "2"])

    assert rc == 0
    assert capsys.readouterr().out.strip() == "classified 2 item(s)"
