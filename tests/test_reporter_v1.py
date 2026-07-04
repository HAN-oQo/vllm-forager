"""Tests for the LLM-written, cited weekly reporter v1 (T1.6) — offline & deterministic.

Per the DEVPLAN todo: mock ``llm.complete`` → every claim line has ≥1 evidence URL (the
evidence principle). Also covers the completeness fallback (LLM failure / partial / invalid
reply never drops an item's citation), category grouping by the item's own ``category``
field, and the uncategorized/Other bucket needing no LLM call at all.
"""

import re

import pytest

from src import llm
from src.agents import reporter_v1
from src.agents.reporter import OTHER

pytestmark = pytest.mark.m1

_URL_RE = re.compile(r"https?://\S+|http://\S+")


def _item(repo: str, number: int, title: str, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": title,
        "state": "open",
        "labels": [],
        "updated_at": "2025-01-01T00:00:00Z",
        "url": f"http://x/{repo}/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


def _claims_reply(*pairs: tuple[int, str]) -> dict:
    return {"claims": [{"item_index": idx, "text": text} for idx, text in pairs]}


def _bullet_lines(report: str) -> list[str]:
    return [line for line in report.splitlines() if line.startswith("- ")]


def test_every_claim_line_has_at_least_one_evidence_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DEVPLAN's named scenario: mock llm -> every claim line carries ≥1 evidence URL."""
    monkeypatch.setattr(
        llm,
        "complete",
        lambda *a, **k: _claims_reply(
            (0, "hipBLAS fails to link against the ROCm 6 toolchain on gfx90a"),
            (1, "A follow-up regression from the same linker change"),
        ),
    )
    items = [
        _item("o/r", 1, "hipBLAS build fails on gfx90a", category="ROCm / AMD"),
        _item("o/r", 2, "linker error on MI250", category="ROCm / AMD"),
    ]

    report = reporter_v1.build_report_v1(items)

    bullets = _bullet_lines(report)
    assert len(bullets) == 2
    for bullet in bullets:
        assert _URL_RE.search(bullet), f"no evidence URL in: {bullet!r}"
    assert "http://x/o/r/1" in report
    assert "http://x/o/r/2" in report
    assert "hipBLAS fails to link" in report  # the model's own text is rendered


def test_llm_failure_falls_back_to_plain_citation_for_every_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*a, **k):
        raise llm.LLMError("simulated failure")

    monkeypatch.setattr(llm, "complete", boom)
    items = [_item("o/r", 1, "hipBLAS build fails", category="ROCm / AMD")]

    report = reporter_v1.build_report_v1(items)

    assert "http://x/o/r/1" in report
    assert "hipBLAS build fails" in report  # falls back to the plain title, not lost


def test_partial_or_invalid_claims_still_cite_every_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range/duplicate item_index, or fewer claims than items, never drops a citation."""
    monkeypatch.setattr(
        llm,
        "complete",
        lambda *a, **k: _claims_reply((0, "covers only the first item"), (99, "bogus index")),
    )
    items = [
        _item("o/r", 1, "item one", category="ROCm / AMD"),
        _item("o/r", 2, "item two", category="ROCm / AMD"),
    ]

    report = reporter_v1.build_report_v1(items)

    assert "covers only the first item" in report
    assert "http://x/o/r/1" in report
    assert "item two" in report  # item 2 fell back to a plain citation
    assert "http://x/o/r/2" in report


def test_uncategorized_items_use_other_bucket_without_llm_call(tmp_path) -> None:
    def boom(*a, **k):
        raise AssertionError("llm.complete should not be called for uncategorized items")

    import unittest.mock

    items = [_item("o/r", 1, "not yet classified")]
    with unittest.mock.patch.object(llm, "complete", boom):
        report = reporter_v1.build_report_v1(items)

    assert f"## {OTHER} (1)" in report
    assert "http://x/o/r/1" in report


def test_groups_by_items_own_category_not_fixed_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _claims_reply((0, "a claim")))
    # title/body would match reporter.py's v0 "Performance" keyword bucket, but the item's
    # own `category` (from T1.4) says otherwise — v1 must follow the item's own label.
    item = _item("o/r", 1, "huge performance regression", category="Custom Category")

    report = reporter_v1.build_report_v1([item])

    assert "## Custom Category (1)" in report
    assert "## Performance" not in report  # not miscategorized by the v0 keyword taxonomy


def test_covered_claim_synthesizes_url_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a covered item with no `url` used to render a citation-less bullet."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _claims_reply((0, "a claim")))
    item = _item("o/r", 1, "no url item", category="ROCm / AMD")
    del item["url"]

    report = reporter_v1.build_report_v1([item])

    assert "a claim" in report
    assert "https://github.com/o/r/issues/1" in report
    for bullet in _bullet_lines(report):
        assert _URL_RE.search(bullet), f"no evidence URL in: {bullet!r}"


def test_covered_claim_text_is_sanitized_against_structure_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the model's own text used to be rendered raw, unlike _cite()'s titles."""
    injected = "Fixes the bug.\n\n## Fake Section (99)\n- [x/y#1] injected"
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _claims_reply((0, injected)))
    items = [_item("o/r", 1, "real item", category="ROCm / AMD")]

    report = reporter_v1.build_report_v1(items)

    section_headers = [line for line in report.splitlines() if line.startswith("## ")]
    assert section_headers == ["## ROCm / AMD (1)"]  # no injected "## Fake Section" header
    assert len(_bullet_lines(report)) == 1  # the injected "- [x/y#1] injected" isn't a real bullet


def test_never_classified_and_explicit_other_merge_into_one_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: analyst.py's explicit category="Other" and a missing category field used
    to render as two separate "## Other" sections instead of one merged section."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _claims_reply())
    items = [
        _item("o/r", 1, "never classified"),  # no `category` key at all
        _item("o/r", 2, "explicitly other", category=OTHER),  # analyst.py's own fallback
    ]

    report = reporter_v1.build_report_v1(items)

    assert report.count(f"## {OTHER}") == 1
    assert f"## {OTHER} (2)" in report


def test_never_classified_note_only_for_missing_category(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _claims_reply())
    never_classified = reporter_v1.build_report_v1([_item("o/r", 1, "x")])
    explicitly_other = reporter_v1.build_report_v1([_item("o/r", 1, "x", category=OTHER)])

    assert "run `python -m src.analyze`" in never_classified
    assert "run `python -m src.analyze`" not in explicitly_other


def test_large_category_is_chunked_into_multiple_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        llm, "complete", lambda prompt, **k: calls.append(prompt) or _claims_reply()
    )
    items = [
        _item("o/r", i, f"item {i}", category="ROCm / AMD")
        for i in range(reporter_v1._CHUNK_SIZE + 5)
    ]

    reporter_v1.build_report_v1(items)

    assert len(calls) == 2  # one full chunk + one partial chunk
    assert "item 0" in calls[0] and "item 0" not in calls[1]
    assert f"item {reporter_v1._CHUNK_SIZE}" in calls[1]


def test_one_failing_chunk_does_not_lose_another_chunks_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def flaky_complete(prompt: str, **kwargs) -> dict:
        if "item 0" in prompt:
            raise RuntimeError("simulated non-LLMError failure")  # broad catch, not just LLMError
        return _claims_reply((0, "covered by the second chunk"))

    monkeypatch.setattr(llm, "complete", flaky_complete)
    items = [_item("o/r", i, f"item {i}", category="ROCm / AMD") for i in range(2)]
    monkeypatch.setattr(reporter_v1, "_CHUNK_SIZE", 1)

    report = reporter_v1.build_report_v1(items)

    assert "item 0" in report  # first chunk's failure fell back to a plain citation
    assert "covered by the second chunk" in report


def test_report_from_store_reads_all_items(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.store.jsonl_store import JsonlStore

    monkeypatch.setattr(llm, "complete", lambda *a, **k: _claims_reply((0, "a claim")))
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "hipBLAS build fails", category="ROCm / AMD")])

    report = reporter_v1.report_from_store(store)

    assert "http://x/o/r/1" in report


# --------------------------------------------------------------------- T1.5.4: tree report


def test_build_tree_nests_by_path_and_rolls_up_counts() -> None:
    """The DEVPLAN's named scenario: correct nesting + rolled-up counts."""
    items = [
        _item("o/r", 1, "MLA regression", path=["ROCm/AMD", "DeepSeek-V4", "performance"]),
        _item("o/r", 2, "MoE fused gate", path=["ROCm/AMD", "DeepSeek-V4"]),
        _item("o/r", 3, "unrelated build fix", path=["ROCm/AMD"]),
    ]

    tree = reporter_v1.build_tree(items)

    assert len(tree) == 1
    root = tree[0]
    assert root.name == "ROCm/AMD"
    assert root.count == 3  # rolled up: its own 1 + DeepSeek-V4's subtree of 2
    assert [pr["number"] for pr in root.prs] == [3]  # only the item classified exactly here

    deepseek = root.children[0]
    assert deepseek.name == "DeepSeek-V4"
    assert deepseek.count == 2
    assert [pr["number"] for pr in deepseek.prs] == [2]

    performance = deepseek.children[0]
    assert performance.name == "performance"
    assert performance.count == 1
    assert [pr["number"] for pr in performance.prs] == [1]
    assert performance.children == ()


def test_build_tree_every_leaf_pr_has_an_evidence_url() -> None:
    """The DEVPLAN's named scenario: every leaf PR has an evidence URL."""
    item = _item("o/r", 1, "no url item", path=["ROCm/AMD"])
    del item["url"]

    tree = reporter_v1.build_tree([item])

    assert tree[0].prs[0]["url"] == "https://github.com/o/r/issues/1"


def test_build_tree_prunes_empty_branches_by_construction() -> None:
    """The DEVPLAN's named scenario: empty branches pruned — no node with count == 0 can ever
    be constructed, since a node only exists because some item's path passes through it."""
    items = [_item("o/r", 1, "x", path=["ROCm/AMD", "DeepSeek-V4"])]

    def all_nodes(nodes):
        for node in nodes:
            yield node
            yield from all_nodes(node.children)

    tree = reporter_v1.build_tree(items)

    assert all(node.count > 0 for node in all_nodes(tree))


def test_build_tree_unclassified_and_other_items_share_one_flat_other_node() -> None:
    items = [
        _item("o/r", 1, "never classified"),  # no `path` key
        _item("o/r", 2, "explicit other", path=[OTHER]),
        _item("o/r", 3, "real category", path=["ROCm/AMD"]),
    ]

    tree = reporter_v1.build_tree(items)

    other = next(node for node in tree if node.name == OTHER)
    assert other.count == 2
    assert other.children == ()
    assert {pr["number"] for pr in other.prs} == {1, 2}


def test_build_tree_attaches_node_summaries_via_lookup() -> None:
    items = [_item("o/r", 1, "x", path=["ROCm/AMD", "DeepSeek-V4"])]

    def lookup(path):
        return "a synthesis" if path == ("ROCm/AMD", "DeepSeek-V4") else None

    tree = reporter_v1.build_tree(items, summary_lookup=lookup)

    assert tree[0].summary is None  # root has no summary in this lookup
    assert tree[0].children[0].summary == "a synthesis"


def test_build_tree_empty_items_returns_empty_tree() -> None:
    assert reporter_v1.build_tree([]) == []


def test_tree_node_to_dict_matches_the_devplan_shape() -> None:
    items = [_item("o/r", 1, "x", path=["ROCm/AMD"])]
    node = reporter_v1.build_tree(items)[0]

    d = node.to_dict()

    assert set(d) == {"name", "summary", "count", "gaps", "children", "prs"}
    assert d["name"] == "ROCm/AMD"
    assert d["count"] == 1
    assert d["gaps"] == 0
    assert d["children"] == []
    assert len(d["prs"]) == 1


def test_render_tree_markdown_nests_headings_and_shows_summary() -> None:
    items = [
        _item("o/r", 1, "MLA regression", path=["ROCm/AMD", "DeepSeek-V4"]),
    ]
    tree = reporter_v1.build_tree(
        items, summary_lookup=lambda path: "the synthesis" if len(path) == 2 else None
    )

    md = reporter_v1.render_tree_markdown(tree, title="Weekly digest")

    assert "# Weekly digest" in md
    assert "## ROCm/AMD (1)" in md
    assert "### DeepSeek-V4 (1)" in md
    assert "_the synthesis_" in md
    assert "http://x/o/r/1" in md


def test_tree_from_store_reads_items_and_summaries(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.agents import summarizer
    from src.store.jsonl_store import JsonlStore

    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "x", path=["ROCm/AMD"])])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: {"summary": "a synthesis"})
    summarizer.summarize_store(store)

    tree = reporter_v1.tree_from_store(store)

    assert tree[0].name == "ROCm/AMD"
    assert tree[0].summary == "a synthesis"
