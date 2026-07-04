"""Tests for the Analyst agent (T1.4) — offline & deterministic.

Per the DEVPLAN todo: mock ``llm.complete`` → item updated with category + citation
(``url``) preserved. Also covers the unknown-category fallback, delta-only reclassification,
and the multi-item store-backed path.
"""

import pytest

from src import llm
from src.agents import analyst
from src.store.jsonl_store import JsonlStore
from src.taxonomy import Taxonomy, create_taxonomy

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


def test_classify_item_sets_category_and_preserves_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"category": "rocm-build"})
    taxonomy = Taxonomy(version=1, categories=("rocm-build", "performance"))
    item = _item("o/r", 1, "hipBLAS build fails on gfx90a")

    result = analyst.classify_item(item, taxonomy)

    assert result["category"] == "rocm-build"
    assert result["taxonomy_version"] == 1
    assert result["url"] == item["url"]  # evidence citation preserved
    assert result["title"] == item["title"]  # original fields untouched


def test_classify_item_unknown_category_falls_back_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"category": "not-a-real-category"})
    taxonomy = Taxonomy(version=1, categories=("rocm-build",))
    result = analyst.classify_item(_item("o/r", 1, "x"), taxonomy)
    assert result["category"] == analyst.OTHER


def test_classify_item_non_dict_reply_falls_back_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "not json")
    taxonomy = Taxonomy(version=1, categories=("rocm-build",))
    result = analyst.classify_item(_item("o/r", 1, "x"), taxonomy)
    assert result["category"] == analyst.OTHER


def test_analyze_store_classifies_only_pending_items(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, ["rocm-build", "performance"])
    store.upsert_items(
        [
            _item("o/r", 1, "hipBLAS build fails on gfx90a"),
            _item("o/r", 2, "already classified", category="performance", taxonomy_version=1),
        ]
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"category": "rocm-build"})

    classified = analyst.analyze_store(store)

    assert len(classified) == 1
    assert classified[0]["number"] == 1
    assert classified[0]["category"] == "rocm-build"
    # the store itself now reflects the new classification, and the pre-classified item
    # (#2) was left untouched by this run
    stored = {item["number"]: item for item in store.query()}
    assert stored[1]["category"] == "rocm-build"
    assert stored[2]["category"] == "performance"


def test_analyze_store_no_pending_items_returns_empty(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, ["rocm-build"])
    store.upsert_items([_item("o/r", 1, "x", category="rocm-build", taxonomy_version=1)])
    assert analyst.analyze_store(store) == []


def test_analyze_store_no_taxonomy_raises(tmp_path) -> None:
    from src.taxonomy import TaxonomyError

    store = JsonlStore(tmp_path)
    with pytest.raises(TaxonomyError, match="no taxonomy exists"):
        analyst.analyze_store(store)
