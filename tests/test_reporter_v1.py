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


def test_report_from_store_reads_all_items(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.store.jsonl_store import JsonlStore

    monkeypatch.setattr(llm, "complete", lambda *a, **k: _claims_reply((0, "a claim")))
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "hipBLAS build fails", category="ROCm / AMD")])

    report = reporter_v1.report_from_store(store)

    assert "http://x/o/r/1" in report
