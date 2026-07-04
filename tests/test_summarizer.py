"""Tests for the Node summarizer agent (T1.5.3) — offline & deterministic.

Per the DEVPLAN todo: mock ``llm.complete`` → a node summary is produced and every claim
cites ≥1 item URL; an empty node yields no summary (no hallucinated content).
"""

import pytest

from src import llm
from src.agents import summarizer
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m1


def _item(repo: str, number: int, title: str, path: list[str] | None, **overrides) -> dict:
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
    if path is not None:
        rec["path"] = path
    rec.update(overrides)
    return rec


# --------------------------------------------------------------------- summarize_node


def test_summarize_node_produces_a_cited_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DEVPLAN's named scenario: a node summary is produced and cites ≥1 item URL."""
    captured = {}

    def fake_complete(prompt: str, *, json_schema: dict) -> dict:
        captured["prompt"] = prompt
        captured["json_schema"] = json_schema
        return {"summary": "MLA and MoE kernels are the frontier on ROCm."}

    monkeypatch.setattr(llm, "complete", fake_complete)
    items = [
        _item("o/r", 1, "MLA decode regression on MI300", None),
        _item("o/r", 2, "MoE fused gate HIP build", None),
    ]

    result = summarizer.summarize_node(("ROCm/AMD", "DeepSeek-V4"), items)

    assert result is not None
    assert result.text == "MLA and MoE kernels are the frontier on ROCm."
    assert result.path == ("ROCm/AMD", "DeepSeek-V4")
    assert set(result.evidence) == {"http://x/o/r/1", "http://x/o/r/2"}
    assert captured["json_schema"] == summarizer._SUMMARY_SCHEMA
    assert "MLA decode regression on MI300" in captured["prompt"]
    assert "ROCm/AMD > DeepSeek-V4" in captured["prompt"]


def test_summarize_node_empty_items_returns_none_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DEVPLAN's named scenario: an empty node yields no summary (no hallucinated content)."""

    def boom(*a, **k):
        raise AssertionError("llm.complete should not be called for an empty node")

    monkeypatch.setattr(llm, "complete", boom)

    assert summarizer.summarize_node(("ROCm/AMD",), []) is None


def test_summarize_node_blank_reply_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"summary": "   "})
    result = summarizer.summarize_node(("ROCm/AMD",), [_item("o/r", 1, "x", None)])
    assert result is None


def test_summarize_node_non_dict_reply_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "not json")
    result = summarizer.summarize_node(("ROCm/AMD",), [_item("o/r", 1, "x", None)])
    assert result is None


def test_summarize_node_items_with_no_url_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A summary with nothing to cite is exactly the uncited-claim class T1.6/T1.8's own
    guardrails keep out of a report."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"summary": "a synthesis"})
    result = summarizer.summarize_node(("ROCm/AMD",), [_item("o/r", 1, "x", None, url=None)])
    assert result is None


# --------------------------------------------------------------------- summarize_store


def test_summarize_store_groups_by_every_path_prefix(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A depth-2 classified item contributes to both its root node and its full-path node."""
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, "MLA regression", ["ROCm/AMD", "DeepSeek-V4"]),
            _item("o/r", 2, "unrelated build fix", ["ROCm/AMD"]),
        ]
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"summary": "synthesized"})

    summaries = summarizer.summarize_store(store)

    assert set(summaries) == {("ROCm/AMD",), ("ROCm/AMD", "DeepSeek-V4")}
    # the root node ("ROCm/AMD",) is informed by BOTH items
    assert set(summaries[("ROCm/AMD",)].evidence) == {"http://x/o/r/1", "http://x/o/r/2"}
    # the deeper node is informed by only the item actually classified that deep
    assert summaries[("ROCm/AMD", "DeepSeek-V4")].evidence == ("http://x/o/r/1",)


def test_summarize_store_writes_to_the_kb_and_get_node_summary_reads_it_back(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "x", ["ROCm/AMD"])])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"summary": "synthesized"})

    summarizer.summarize_store(store)

    fetched = summarizer.get_node_summary(store, ("ROCm/AMD",))
    assert fetched is not None
    assert fetched.text == "synthesized"
    assert fetched.evidence == ("http://x/o/r/1",)


def test_summarize_store_excludes_unclassified_and_other_items(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, "not yet classified", None),
            _item("o/r", 2, "fell to other", ["Other"]),
        ]
    )

    def boom(*a, **k):
        raise AssertionError("llm.complete should not be called — no real node exists")

    monkeypatch.setattr(llm, "complete", boom)

    assert summarizer.summarize_store(store) == {}


def test_summarize_store_recomputes_and_overwrites_a_stale_summary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node's summary reflects its CURRENT item set — no delta/skip-if-cached shortcut that
    would leave it permanently stale as more items are classified under it."""
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "first item", ["ROCm/AMD"])])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"summary": "just item 1"})
    summarizer.summarize_store(store)

    store.upsert_items([_item("o/r", 2, "second item", ["ROCm/AMD"])])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"summary": "items 1 and 2"})
    summarizer.summarize_store(store)

    fetched = summarizer.get_node_summary(store, ("ROCm/AMD",))
    assert fetched.text == "items 1 and 2"
    assert set(fetched.evidence) == {"http://x/o/r/1", "http://x/o/r/2"}


def test_summarize_store_skips_failing_node_and_persists_the_rest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, "good node item", ["Quantization"]),
            _item("o/r", 2, "bad node item", ["ROCm/AMD"]),
        ]
    )

    def flaky_complete(prompt: str, **kwargs) -> dict:
        if "ROCm/AMD" in prompt:
            raise llm.LLMError("simulated transient failure")
        return {"summary": "quantization synthesis"}

    monkeypatch.setattr(llm, "complete", flaky_complete)

    summaries = summarizer.summarize_store(store)

    assert set(summaries) == {("Quantization",)}
    assert summarizer.get_node_summary(store, ("ROCm/AMD",)) is None


def test_summarize_store_empty_store_returns_empty(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    assert summarizer.summarize_store(store) == {}


def test_get_node_summary_unknown_node_returns_none(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    assert summarizer.get_node_summary(store, ("nonexistent",)) is None


# --------------------------------------------------------------------- NodeSummary


def test_node_summary_json_roundtrip() -> None:
    original = summarizer.NodeSummary(
        path=("a", "b"), text="synthesis", evidence=("http://x/1", "http://x/2")
    )
    restored = summarizer.NodeSummary.from_json(original.path, original.to_json())
    assert restored == original
