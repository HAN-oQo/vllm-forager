"""Tests for the report CLI (T0.9) — offline & deterministic.

Per the DEVPLAN todo: ``main()`` on a tmp store creates a non-empty report file. Also covers
the ISO-week filename stamp and the injectable :func:`generate` core (fixed clock → fixed
path, content carries the cited links).
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import report
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
        "updated_at": "2025-01-01T00:00:00Z",
        "url": f"http://x/{repo}/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


def test_week_stamp_is_iso_year_week():
    # 2026-07-04 is in ISO week 27 of ISO year 2026.
    assert report.week_stamp(datetime(2026, 7, 4, tzinfo=timezone.utc)) == "2026-W27"


def test_generate_writes_dated_nonempty_report(tmp_path):
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "hipBLAS build fails on gfx90a", labels=["rocm"])])

    when = datetime(2026, 7, 4, tzinfo=timezone.utc)
    path = report.generate(store, tmp_path / "reports", when=when)

    assert path == tmp_path / "reports" / "2026-W27.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.strip()  # non-empty
    assert "http://x/o/r/1" in text  # the cited source link is present
    assert "2026-W27" in text  # week-stamped title


def test_generate_custom_title_overrides_default(tmp_path):
    # The `title` override replaces the heading; the filename still comes from the week stamp.
    store = JsonlStore(tmp_path)
    when = datetime(2026, 7, 4, tzinfo=timezone.utc)
    path = report.generate(store, tmp_path / "reports", when=when, title="Custom heading")
    assert path.name == "2026-W27.md"
    assert path.read_text(encoding="utf-8").startswith("# Custom heading")


def test_main_creates_nonempty_file_on_tmp_store(tmp_path, capsys):
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "MI250 OOM during warmup", labels=["amd"])])

    rc = report.main(["--data-dir", str(tmp_path)])
    assert rc == 0

    # Trust main's printed path as the source of truth: it derives the filename from its own
    # wall-clock read, so recomputing week_stamp() here could race across an ISO-week rollover.
    printed = Path(capsys.readouterr().out.strip())
    assert printed.parent == tmp_path / "reports"
    assert re.fullmatch(r"\d{4}-W\d{2}\.md", printed.name)  # YYYY-Www.md
    assert printed.exists()
    assert printed.read_text(encoding="utf-8").strip()  # non-empty


def test_main_on_empty_store_still_writes_a_report(tmp_path, capsys):
    # No items collected yet → the report is still produced (safe empty digest).
    JsonlStore(tmp_path)  # nothing upserted
    rc = report.main(["--data-dir", str(tmp_path)])
    assert rc == 0
    written = Path(capsys.readouterr().out.strip())  # trust main's printed path (no clock race)
    assert written.exists()
    assert "0 items across 0 repos." in written.read_text(encoding="utf-8")


def test_main_v0_flag_uses_fixed_keyword_taxonomy_offline(tmp_path, capsys):
    """--v0 is the offline/no-LLM fallback: an uncategorized item still gets a real
    keyword-based bucket (v0 behavior), not v1's blanket "Other"."""
    store = JsonlStore(tmp_path)
    # No `category` field set (as if T1.4's analyst never ran) — v1 would bucket this as
    # "Other"; v0's keyword match should still find "ROCm / AMD" from the title.
    store.upsert_items([_item("o/r", 1, "hipBLAS build fails on gfx90a")])

    rc = report.main(["--data-dir", str(tmp_path), "--v0"])

    assert rc == 0
    written = Path(capsys.readouterr().out.strip())
    text = written.read_text(encoding="utf-8")
    assert "## ROCm / AMD" in text
    assert "## Other" not in text


def test_main_tree_flag_writes_markdown_and_json(tmp_path, capsys):
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "hipBLAS build fails", path=["ROCm/AMD"])])

    rc = report.main(["--data-dir", str(tmp_path), "--tree"])

    assert rc == 0
    printed = capsys.readouterr().out.strip().splitlines()
    md_path, json_path = Path(printed[0]), Path(printed[1])
    assert md_path.name.endswith(".tree.md")
    assert json_path.name == md_path.name.removesuffix(".md") + ".json"
    assert "## ROCm/AMD (1)" in md_path.read_text(encoding="utf-8")
    tree = json.loads(json_path.read_text(encoding="utf-8"))
    assert tree[0]["name"] == "ROCm/AMD"
    assert tree[0]["count"] == 1


def test_main_tree_and_default_reports_never_collide_on_filename(tmp_path, capsys):
    """Regression: --tree's Markdown used to write to the SAME <week>.md as the default flat
    report, silently overwriting whichever ran first."""
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "hipBLAS build fails", path=["ROCm/AMD"])])

    report.main(["--data-dir", str(tmp_path), "--tree"])
    tree_printed = Path(capsys.readouterr().out.strip().splitlines()[0])

    report.main(["--data-dir", str(tmp_path)])
    flat_printed = Path(capsys.readouterr().out.strip())

    assert tree_printed != flat_printed
    assert tree_printed.exists() and flat_printed.exists()
    assert "## ROCm/AMD (1)" in tree_printed.read_text(encoding="utf-8")  # untouched by the 2nd run


def test_main_v0_and_tree_together_errors(tmp_path):
    with pytest.raises(SystemExit):
        report.main(["--data-dir", str(tmp_path), "--v0", "--tree"])


# -------------------------------------------------------------- list_reports / read_tree_report


def test_list_reports_returns_newest_first(tmp_path):
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "x", path=["build"])])
    report.generate_tree(
        store, tmp_path / "reports", when=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )
    report.generate_tree(
        store, tmp_path / "reports", when=datetime(2026, 6, 1, tzinfo=timezone.utc)
    )

    reports = report.list_reports(tmp_path)

    assert [r["stamp"] for r in reports] == ["2026-W23", "2026-W02"]


def test_list_reports_empty_before_any_report_generated(tmp_path):
    assert report.list_reports(tmp_path) == []


def test_list_reports_ignores_a_directory_named_like_a_report(tmp_path):
    """Regression: an earlier version's glob had no is_file() guard, so a stray directory
    ending in .tree.json (e.g. left by an interrupted write) would be listed, then raise
    IsADirectoryError when a caller tried to open it via read_tree_report."""
    store = JsonlStore(tmp_path)
    report.generate_tree(
        store, tmp_path / "reports", when=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )
    (tmp_path / "reports" / "2026-W99.tree.json").mkdir()

    reports = report.list_reports(tmp_path)

    assert [r["stamp"] for r in reports] == ["2026-W02"]


def test_read_tree_report_returns_the_tree_content(tmp_path):
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "x", path=["build"])])
    report.generate_tree(
        store, tmp_path / "reports", when=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )

    tree = report.read_tree_report(tmp_path, "2026-W02")

    assert tree is not None
    assert tree[0]["name"] == "build"
    assert tree[0]["count"] == 1


def test_read_tree_report_unknown_stamp_returns_none(tmp_path):
    store = JsonlStore(tmp_path)
    report.generate_tree(
        store, tmp_path / "reports", when=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )
    assert report.read_tree_report(tmp_path, "2026-W99") is None


def test_read_tree_report_no_reports_dir_returns_none(tmp_path):
    assert report.read_tree_report(tmp_path, "2026-W02") is None


def test_read_tree_report_rejects_relative_path_traversal(tmp_path):
    """Regression, empirically reproduced during code review: an earlier version passed
    `stamp` through unvalidated, so "../secret/leaked" (relative traversal) could read a
    .tree.json file outside reports_dir."""
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "leaked.tree.json").write_text('[{"leaked": true}]')

    assert report.read_tree_report(tmp_path, "../secret/leaked") is None


def test_read_tree_report_rejects_absolute_path_stamp(tmp_path):
    """Regression, empirically reproduced during code review: an earlier version passed
    `stamp` through unvalidated -- an absolute-path stamp made `reports_dir / stamp` discard
    reports_dir entirely (pathlib semantics), reading an arbitrary file on disk."""
    outside = tmp_path / "outside.tree.json"
    outside.write_text('[{"leaked": true}]')

    assert report.read_tree_report(tmp_path, str(outside).removesuffix(".tree.json")) is None


def test_read_tree_report_corrupt_json_returns_none(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "2026-W02.tree.json").write_text("not valid json")

    assert report.read_tree_report(tmp_path, "2026-W02") is None


def test_read_tree_report_directory_at_the_expected_path_returns_none(tmp_path):
    """A directory (not a file) at the exact expected path degrades to None rather than
    raising IsADirectoryError -- part of the broadened, code-review-driven exception
    handling (an earlier version only caught json.JSONDecodeError)."""
    reports_dir = tmp_path / "reports"
    (reports_dir / "2026-W02.tree.json").mkdir(parents=True)

    assert report.read_tree_report(tmp_path, "2026-W02") is None
