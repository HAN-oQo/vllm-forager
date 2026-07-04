"""Tests for the Analyst agent (T1.4, path-classifying since T1.5.2) — offline & deterministic.

Per the DEVPLAN todo: mock ``llm.complete`` → item gets a valid path (each level from the
allowed set), citation preserved; an off-taxonomy answer is rejected/normalized. Also covers
the empty-taxonomy short-circuit, per-item failure isolation, delta-only reclassification, and
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


def test_classify_item_sets_path_and_category_and_preserves_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_complete(prompt: str, *, json_schema: dict) -> dict:
        captured["prompt"] = prompt
        captured["json_schema"] = json_schema
        return {"path": ["ROCm/AMD", "DeepSeek-V4", "performance"]}

    monkeypatch.setattr(llm, "complete", fake_complete)
    taxonomy = Taxonomy(
        version=1,
        categories=(("ROCm/AMD", "DeepSeek-V4", "performance"), ("Quantization",)),
    )
    item = _item("o/r", 1, "hipBLAS build fails on gfx90a")

    result = analyst.classify_item(item, taxonomy)

    assert result["path"] == ["ROCm/AMD", "DeepSeek-V4", "performance"]
    assert result["category"] == "ROCm/AMD > DeepSeek-V4 > performance"
    assert result["taxonomy_version"] == 1
    assert result["url"] == item["url"]  # evidence citation preserved
    assert result["title"] == item["title"]  # original fields untouched
    # the schema/prompt actually reach llm.complete, not just decorative constants
    assert captured["json_schema"] == analyst._PATH_SCHEMA
    assert "hipBLAS build fails on gfx90a" in captured["prompt"]
    assert "ROCm/AMD > DeepSeek-V4 > performance" in captured["prompt"]


def test_classify_item_stops_at_the_first_level_that_drifts_off_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1.5.2's named scenario: an off-taxonomy answer is normalized, not rejected outright —
    the valid prefix is kept."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": ["ROCm/AMD", "not-a-real-child"]})
    taxonomy = Taxonomy(version=1, categories=(("ROCm/AMD", "DeepSeek-V4"),))

    result = analyst.classify_item(_item("o/r", 1, "x"), taxonomy)

    assert result["path"] == ["ROCm/AMD"]
    assert result["category"] == "ROCm/AMD"


def test_classify_item_path_match_is_case_and_whitespace_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors taxonomy.add_category's own casefold+strip normalization."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": [" ROCM-BUILD "]})
    taxonomy = Taxonomy(version=1, categories=(("rocm-build",),))

    result = analyst.classify_item(_item("o/r", 1, "x"), taxonomy)

    # the taxonomy's own canonical spelling is used, not the model's raw casing
    assert result["path"] == ["rocm-build"]


def test_classify_item_unknown_root_falls_back_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": ["not-a-real-category"]})
    taxonomy = Taxonomy(version=1, categories=(("rocm-build",),))

    result = analyst.classify_item(_item("o/r", 1, "x"), taxonomy)

    assert result["path"] == [analyst.OTHER]
    assert result["category"] == analyst.OTHER


def test_classify_item_non_dict_reply_falls_back_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "not json")
    taxonomy = Taxonomy(version=1, categories=(("rocm-build",),))
    result = analyst.classify_item(_item("o/r", 1, "x"), taxonomy)
    assert result["path"] == [analyst.OTHER]


def test_classify_item_empty_path_reply_falls_back_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": []})
    taxonomy = Taxonomy(version=1, categories=(("rocm-build",),))
    result = analyst.classify_item(_item("o/r", 1, "x"), taxonomy)
    assert result["path"] == [analyst.OTHER]


def test_canonical_path_non_string_element_falls_back_to_other() -> None:
    taxonomy = Taxonomy(version=1, categories=(("rocm-build",),))
    assert analyst._canonical_path(["rocm-build", 5], taxonomy) == (analyst.OTHER,)


def test_prompt_truncates_long_body() -> None:
    long_body = "x" * (analyst._BODY_CHARS + 500)
    taxonomy = Taxonomy(version=1, categories=(("a",),))
    prompt = analyst._path_prompt(_item("o/r", 1, "t", body=long_body), taxonomy)
    assert "x" * analyst._BODY_CHARS in prompt
    assert "x" * (analyst._BODY_CHARS + 1) not in prompt


def test_analyze_store_classifies_only_pending_items(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, ["rocm-build", "performance"])
    store.upsert_items(
        [
            _item("o/r", 1, "hipBLAS build fails on gfx90a"),
            _item(
                "o/r",
                2,
                "already classified",
                path=["performance"],
                category="performance",
                taxonomy_version=1,
            ),
        ]
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": ["rocm-build"]})

    classified = analyst.analyze_store(store)

    assert len(classified) == 1
    assert classified[0]["number"] == 1
    assert classified[0]["path"] == ["rocm-build"]
    # the store itself now reflects the new classification, and the pre-classified item
    # (#2) was left untouched by this run
    stored = {item["number"]: item for item in store.query()}
    assert stored[1]["path"] == ["rocm-build"]
    assert stored[2]["path"] == ["performance"]


def test_analyze_store_backfills_path_for_a_pre_t1_5_2_flat_category_item(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The delta check switched from "no category" to "no path" — an item classified by the
    pre-T1.5.2 Analyst (has `category`, no `path`) gets re-classified once to backfill it."""
    store = JsonlStore(tmp_path)
    create_taxonomy(store, ["rocm-build"])
    store.upsert_items([_item("o/r", 1, "legacy item", category="rocm-build", taxonomy_version=1)])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"path": ["rocm-build"]})

    classified = analyst.analyze_store(store)

    assert len(classified) == 1
    assert classified[0]["path"] == ["rocm-build"]


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
        return {"path": ["rocm-build"]}

    monkeypatch.setattr(llm, "complete", flaky_complete)

    classified = analyst.analyze_store(store)

    assert len(classified) == 1
    assert classified[0]["number"] == 1
    stored = {item["number"]: item for item in store.query()}
    assert stored[1]["path"] == ["rocm-build"]
    assert "path" not in stored[2]  # left pending for a future retry


def test_analyze_store_empty_taxonomy_marks_other_without_calling_llm(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, [])
    store.upsert_items([_item("o/r", 1, "x")])

    def boom(*args, **kwargs):
        raise AssertionError("llm.complete should not be called for an empty taxonomy")

    import unittest.mock

    with unittest.mock.patch.object(llm, "complete", boom):
        classified = analyst.analyze_store(store)

    assert classified[0]["path"] == [analyst.OTHER]
    assert classified[0]["category"] == analyst.OTHER


def test_analyze_store_no_pending_items_returns_empty(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    create_taxonomy(store, ["rocm-build"])
    store.upsert_items(
        [_item("o/r", 1, "x", path=["rocm-build"], category="rocm-build", taxonomy_version=1)]
    )
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
