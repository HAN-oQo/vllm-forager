"""Tests for the baseline weekly reporter (T0.8) — offline & deterministic.

Per the DEVPLAN todo: synthetic items → the report contains every item URL and the correct
per-section counts. Also covers first-match bucketing (an item matching two categories lands in
the earlier one) and the store-backed wrapper.
"""

import re

import pytest

from src.agents import reporter
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m0


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


# Eight items chosen so each lands in a distinct known bucket (ROCm has two → count 2).
def _sample() -> list[dict]:
    return [
        _item("o/r", 1, "hipBLAS build fails on gfx90a", labels=["rocm"]),  # ROCm / AMD
        _item("o/r", 2, "MI250 OOM during warmup", labels=["amd"]),  # ROCm / AMD
        _item("o/r", 3, "Add FP8 quantization support"),  # Quantization
        _item("o/s", 4, "PagedAttention kernel cleanup"),  # Attention / kernels
        _item("o/s", 5, "Tensor parallel hang across nodes"),  # Distributed / serving
        _item("o/s", 6, "Throughput drop after upgrade"),  # Performance
        _item("o/s", 7, "Segfault when loading model"),  # Bugs / crashes
        _item("o/s", 8, "Update installation docs"),  # Other
    ]


def test_categorize_first_match_and_fallback():
    items = _sample()
    got = {i["number"]: reporter.categorize(i) for i in items}
    assert got == {
        1: "ROCm / AMD",
        2: "ROCm / AMD",
        3: "Quantization",
        4: "Attention / kernels",
        5: "Distributed / serving",
        6: "Performance",
        7: "Bugs / crashes",
        8: "Other",
    }


def test_categorize_multi_match_takes_earliest_category():
    # Matches both ROCm (rocm/gfx) and Quantization (fp8/quant); ROCm is earlier → wins.
    item = _item("o/r", 9, "ROCm FP8 quantization on gfx90a")
    assert reporter.categorize(item) == "ROCm / AMD"


def test_categorize_matches_suffixed_hardware_and_dtype_tokens():
    # Prefix-of-token matching: the suffixed forms that dominate real text must still bucket.
    assert reporter.categorize(_item("o/r", 1, "crash on gfx90a")) == "ROCm / AMD"
    assert reporter.categorize(_item("o/r", 2, "OOM on MI300X")) == "ROCm / AMD"
    assert reporter.categorize(_item("o/r", 3, "add fp8e4m3fn support")) == "Quantization"
    # leading boundary still prevents mid-word hits
    assert reporter.categorize(_item("o/r", 4, "fix the microchip driver")) == "Other"


def test_cite_collapses_title_and_synthesizes_missing_url():
    bullet = reporter._cite(_item("o/r", 5, "line1\nline2   spaced", url=""))
    # newline/extra whitespace collapsed so it can't inject Markdown structure
    assert bullet == "- [o/r#5] line1 line2 spaced — https://github.com/o/r/issues/5"
    assert "\n" not in bullet


def test_repo_number_label_present_values():
    assert reporter.repo_number_label({"repo": "o/r", "number": 5}) == "o/r#5"


def test_repo_number_label_treats_none_same_as_missing():
    # a caller-built dict (e.g. reporter_v1's `_pr_entry`) may set these to None rather than
    # omitting them — both must render the same "?" placeholder.
    assert reporter.repo_number_label({"repo": None, "number": None}) == "?#?"
    assert reporter.repo_number_label({}) == "?#?"


def test_build_report_orders_newest_first_then_repo_number_ascending():
    md = reporter.build_report(
        [
            _item("o/z", 1, "rocm A", updated_at="2025-01-02T00:00:00Z"),
            _item("o/a", 2, "rocm B", updated_at="2025-01-01T00:00:00Z"),
            _item("o/a", 1, "rocm C", updated_at="2025-01-01T00:00:00Z"),
        ]
    )
    # newest (o/z#1) first; then the 01-01 tie by repo then number ascending: o/a#1, o/a#2
    order = [md.index(u) for u in ("http://x/o/z/1", "http://x/o/a/1", "http://x/o/a/2")]
    assert order == sorted(order)


def test_build_report_has_every_url_and_correct_counts():
    items = _sample()
    md = reporter.build_report(items)

    # every item is cited by its source link
    for item in items:
        assert item["url"] in md

    # per-section counts and the header total
    assert "8 items across 2 repos." in md
    assert "## ROCm / AMD (2)" in md
    assert "## Quantization (1)" in md
    assert "## Attention / kernels (1)" in md
    assert "## Distributed / serving (1)" in md
    assert "## Performance (1)" in md
    assert "## Bugs / crashes (1)" in md
    assert "## Other (1)" in md
    # the section counts partition the items (2+1*6 = 8)
    assert sum(int(c) for c in re.findall(r"## .+ \((\d+)\)", md)) == 8


def test_build_report_bullet_format_cites_repo_number_and_url():
    md = reporter.build_report([_item("o/r", 1, "hipBLAS build fails on gfx90a", labels=["rocm"])])
    assert "- [o/r#1] hipBLAS build fails on gfx90a — http://x/o/r/1" in md


def test_build_report_empty_is_safe():
    md = reporter.build_report([])
    assert "0 items across 0 repos." in md
    assert "##" not in md  # no sections when there is nothing to report


def test_build_report_omits_empty_sections():
    # Only ROCm items → only the ROCm section appears.
    md = reporter.build_report([_item("o/r", 1, "gfx90a hip crash", labels=["rocm"])])
    assert "## ROCm / AMD (1)" in md
    assert "## Bugs / crashes" not in md  # "crash" present but ROCm matched first


def test_report_from_store_reads_all_items(tmp_path):
    store = JsonlStore(tmp_path)
    items = _sample()
    store.upsert_items(items)
    md = reporter.report_from_store(store)
    for item in items:
        assert item["url"] in md
    assert "8 items across 2 repos." in md
