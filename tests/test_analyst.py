"""Tests for the Analyst agent (T1.4) — offline & deterministic.

Per the DEVPLAN todo: mock ``llm.complete`` → item updated with category + citation
(``url``) preserved. Also covers the unknown/case-insensitive category matching, the
empty-taxonomy short-circuit, per-item failure isolation, delta-only reclassification, and
the ordering of the taxonomy lookup vs. the pending-items check.
"""

import pytest

from src import llm
from src.agents import analyst
from src.store.jsonl_store import JsonlStore
from src.taxonomy import Taxonomy, TaxonomyError, create_taxonomy

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
    captured = {}

    def fake_complete(prompt: str, *, json_schema: dict) -> dict:
        captured["prompt"] = prompt
        captured["json_schema"] = json_schema
        return {"category": "rocm-build"}

    monkeypatch.setattr(llm, "complete", fake_complete)
    taxonomy = Taxonomy(version=1, categories=(("rocm-build",), ("performance",)))
    item = _item("o/r", 1, "hipBLAS build fails on gfx90a")

    result = analyst.classify_item(item, taxonomy)

    assert result["category"] == "rocm-build"
    assert result["taxonomy_version"] == 1
    assert result["url"] == item["url"]  # evidence citation preserved
    assert result["title"] == item["title"]  # original fields untouched
    # the schema/prompt actually reach llm.complete, not just decorative constants
    assert captured["json_schema"] == analyst._CATEGORY_SCHEMA
    assert "hipBLAS build fails on gfx90a" in captured["prompt"]
    assert "rocm-build, performance" in captured["prompt"]


def test_classify_item_flattens_a_multi_level_taxonomy_path_to_one_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1.5.1: a taxonomy category is now a path; classify_item still classifies into one
    flat label per item (via Taxonomy.labels) until T1.5.2 teaches it to classify per-level."""
    captured = {}

    def fake_complete(prompt: str, *, json_schema: dict) -> dict:
        captured["prompt"] = prompt
        return {"category": "ROCm/AMD > DeepSeek-V4 > performance"}

    monkeypatch.setattr(llm, "complete", fake_complete)
    taxonomy = Taxonomy(version=1, categories=(("ROCm/AMD", "DeepSeek-V4", "performance"),))

    result = analyst.classify_item(_item("o/r", 1, "MLA decode regression"), taxonomy)

    assert result["category"] == "ROCm/AMD > DeepSeek-V4 > performance"
    assert "ROCm/AMD > DeepSeek-V4 > performance" in captured["prompt"]


def test_prompt_truncates_long_body() -> None:
    long_body = "x" * (analyst._BODY_CHARS + 500)
    prompt = analyst._prompt(_item("o/r", 1, "t", body=long_body), ("a",))
    assert "x" * analyst._BODY_CHARS in prompt
    assert "x" * (analyst._BODY_CHARS + 1) not in prompt


def test_classify_item_unknown_category_falls_back_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"category": "not-a-real-category"})
    taxonomy = Taxonomy(version=1, categories=(("rocm-build",),))
    result = analyst.classify_item(_item("o/r", 1, "x"), taxonomy)
    assert result["category"] == analyst.OTHER


@pytest.mark.parametrize("variant", ["ROCm-Build", "rocm-build ", " ROCM-BUILD"])
def test_classify_item_category_match_is_case_and_whitespace_insensitive(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """Mirrors taxonomy.add_category's own casefold+strip normalization."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"category": variant})
    taxonomy = Taxonomy(version=1, categories=(("rocm-build",),))
    result = analyst.classify_item(_item("o/r", 1, "x"), taxonomy)
    # the taxonomy's own canonical spelling is used, not the model's raw casing
    assert result["category"] == "rocm-build"


def test_classify_item_non_dict_reply_falls_back_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "not json")
    taxonomy = Taxonomy(version=1, categories=(("rocm-build",),))
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


def test_analyze_store_skips_failing_item_and_persists_the_rest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: one item's LLMError used to discard the whole batch's progress."""
    store = JsonlStore(tmp_path)
    create_taxonomy(store, ["rocm-build"])
    store.upsert_items(
        [_item("o/r", 1, "good item"), _item("o/r", 2, "bad item")],
    )

    def flaky_complete(prompt: str, **kwargs) -> dict:
        if "bad item" in prompt:
            raise llm.LLMError("simulated transient failure")
        return {"category": "rocm-build"}

    monkeypatch.setattr(llm, "complete", flaky_complete)

    classified = analyst.analyze_store(store)

    assert len(classified) == 1
    assert classified[0]["number"] == 1
    stored = {item["number"]: item for item in store.query()}
    assert stored[1]["category"] == "rocm-build"
    assert "category" not in stored[2]  # left pending for a future retry


def test_analyze_store_empty_taxonomy_marks_other_without_calling_llm(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [])
    store.upsert_items([_item("o/r", 1, "x")])

    def boom(*args, **kwargs):
        raise AssertionError("llm.complete should not be called for an empty taxonomy")

    import unittest.mock

    with unittest.mock.patch.object(llm, "complete", boom):
        classified = analyst.analyze_store(store)

    assert classified[0]["category"] == analyst.OTHER


def test_analyze_store_no_pending_items_returns_empty(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, ["rocm-build"])
    store.upsert_items([_item("o/r", 1, "x", category="rocm-build", taxonomy_version=1)])
    assert analyst.analyze_store(store) == []


def test_analyze_store_no_pending_items_and_no_taxonomy_returns_empty(tmp_path) -> None:
    """The pending check runs before the taxonomy lookup, so an empty store never raises."""
    store = JsonlStore(tmp_path)
    assert analyst.analyze_store(store) == []


def test_analyze_store_pending_items_but_no_taxonomy_raises(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "x")])
    with pytest.raises(TaxonomyError, match="no taxonomy exists"):
        analyst.analyze_store(store)
