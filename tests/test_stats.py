"""Tests for the data-summary command (offline)."""

import json

import pytest

from src import config, stats

pytestmark = pytest.mark.m0


def test_summarize_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "o__r.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"number": 1, "type": "issue", "labels": ["rocm"]}),
                json.dumps({"number": 2, "type": "pr", "labels": []}),
                json.dumps({"number": 3, "type": "issue", "labels": ["bug"]}),
            ]
        )
        + "\n"
    )
    summary = stats.summarize()
    assert summary["total"] == 3
    assert summary["repos"]["o__r.jsonl"] == {"total": 3, "pr": 1, "issue": 2, "rocm": 1}


def test_summarize_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert stats.summarize() == {"repos": {}, "total": 0}
