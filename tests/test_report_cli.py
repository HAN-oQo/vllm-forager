"""Tests for the report CLI (T0.9) — offline & deterministic.

Per the DEVPLAN todo: ``main()`` on a tmp store creates a non-empty report file. Also covers
the ISO-week filename stamp and the injectable :func:`generate` core (fixed clock → fixed
path, content carries the cited links).
"""

from datetime import datetime, timezone

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


def test_main_creates_nonempty_file_on_tmp_store(tmp_path, capsys):
    store = JsonlStore(tmp_path)
    store.upsert_items([_item("o/r", 1, "MI250 OOM during warmup", labels=["amd"])])

    rc = report.main(["--data-dir", str(tmp_path)])
    assert rc == 0

    # main prints the written path; it must exist under reports/ and be non-empty.
    printed = capsys.readouterr().out.strip()
    written = tmp_path / "reports" / f"{report.week_stamp()}.md"
    assert printed == str(written)
    assert written.exists()
    assert written.read_text(encoding="utf-8").strip()


def test_main_on_empty_store_still_writes_a_report(tmp_path):
    # No items collected yet → the report is still produced (safe empty digest).
    JsonlStore(tmp_path)  # nothing upserted
    rc = report.main(["--data-dir", str(tmp_path)])
    assert rc == 0
    written = tmp_path / "reports" / f"{report.week_stamp()}.md"
    assert written.exists()
    assert "0 items across 0 repos." in written.read_text(encoding="utf-8")
