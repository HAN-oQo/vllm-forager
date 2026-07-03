"""Tests for the M0 collector — offline & deterministic (no network calls).

Covers checklist items T0.1–T0.5 in docs/DEVPLAN.md.
"""

import json

import pytest

from src import collector, config

pytestmark = pytest.mark.m0


class FakeResp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("raise_for_status called on an error response")


# ---------------------------------------------------------------- _normalize


def test_normalize_issue_vs_pr():
    issue = {
        "number": 1,
        "title": "t",
        "state": "open",
        "labels": [{"name": "bug"}],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-02T00:00:00Z",
        "html_url": "http://x/1",
        "body": "hello",
    }
    n = collector._normalize(issue, "o/r")
    assert n["type"] == "issue"
    assert n["repo"] == "o/r"
    assert n["number"] == 1
    assert n["labels"] == ["bug"]
    assert n["url"] == "http://x/1"

    pr = dict(issue, number=2, pull_request={"url": "..."})
    assert collector._normalize(pr, "o/r")["type"] == "pr"


def test_normalize_body_truncated_and_defaults():
    assert len(collector._normalize({"number": 3, "body": "x" * 5000}, "o/r")["body"]) == 4000
    assert collector._normalize({"number": 4, "body": None}, "o/r")["body"] == ""
    assert collector._normalize({"number": 5}, "o/r")["labels"] == []


# -------------------------------------------------------------- _merge_jsonl


def test_merge_jsonl_upsert_and_order(tmp_path):
    p = tmp_path / "x.jsonl"
    assert (
        collector._merge_jsonl(
            p,
            [
                {"number": 1, "updated_at": "2025-01-02T00:00:00Z", "title": "a"},
                {"number": 2, "updated_at": "2025-01-01T00:00:00Z", "title": "b"},
            ],
        )
        == 2
    )

    # re-merge: number 1 is overwritten, number 3 is added
    assert (
        collector._merge_jsonl(
            p,
            [
                {"number": 1, "updated_at": "2025-01-05T00:00:00Z", "title": "a2"},
                {"number": 3, "updated_at": "2025-01-03T00:00:00Z", "title": "c"},
            ],
        )
        == 3
    )

    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    assert {r["number"]: r["title"] for r in rows}[1] == "a2"  # upserted
    assert [r["number"] for r in rows] == [2, 3, 1]  # sorted asc by updated_at


def test_merge_jsonl_skips_corrupt_line(tmp_path):
    # a valid record + a truncated/corrupt line (as an interrupted write would leave)
    p = tmp_path / "x.jsonl"
    p.write_text(
        json.dumps({"number": 1, "updated_at": "2025-01-01T00:00:00Z"}) + '\n{"number": 2, "titl\n'
    )
    total = collector._merge_jsonl(p, [{"number": 3, "updated_at": "2025-01-02T00:00:00Z"}])
    assert total == 2  # corrupt #2 skipped; #1 kept, #3 added
    nums = sorted(json.loads(x)["number"] for x in p.read_text().splitlines() if x.strip())
    assert nums == [1, 3]


# ------------------------------------------------------------- state cursor


def test_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")
    assert collector._load_state() == {}  # missing file -> {}
    collector._save_state({"o/r": "2025-01-01T00:00:00Z"})
    assert collector._load_state() == {"o/r": "2025-01-01T00:00:00Z"}


# ----------------------------------------------------------------- _headers


def test_headers_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    h = collector._headers()
    assert "Authorization" not in h
    assert h["Accept"] == "application/vnd.github+json"

    monkeypatch.setenv("GITHUB_TOKEN", "abc")
    assert collector._headers()["Authorization"] == "Bearer abc"


# ------------------------------------------------------ _sleep_for_rate_limit


def test_rate_limit_no_wait_on_ok(monkeypatch):
    slept = []
    monkeypatch.setattr(collector.time, "sleep", lambda s: slept.append(s))
    assert collector._sleep_for_rate_limit(FakeResp(200)) is False
    assert slept == []


def test_rate_limit_waits_on_403(monkeypatch):
    slept = []
    monkeypatch.setattr(collector.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(collector.time, "time", lambda: 1000)
    resp = FakeResp(403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1005"})
    assert collector._sleep_for_rate_limit(resp) is True
    assert slept == [6]  # (1005 - 1000) + 1


# --------------------------------------------------------------- fetch_repo


def test_fetch_repo_pagination(monkeypatch):
    monkeypatch.setattr(config, "PER_PAGE", 2)
    pages = {
        1: [
            {"number": 1, "title": "a", "updated_at": "2025-01-01T00:00:00Z"},
            {"number": 2, "title": "b", "pull_request": {}, "updated_at": "2025-01-02T00:00:00Z"},
        ],
        2: [{"number": 3, "title": "c", "updated_at": "2025-01-03T00:00:00Z"}],
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        return FakeResp(200, pages.get(params["page"], []))

    monkeypatch.setattr(collector.requests, "get", fake_get)
    out = collector.fetch_repo("o/r", "2025-01-01T00:00:00Z")
    assert [r["number"] for r in out] == [1, 2, 3]
    assert out[1]["type"] == "pr"


def test_fetch_repo_404(monkeypatch):
    monkeypatch.setattr(
        collector.requests,
        "get",
        lambda url, headers=None, params=None, timeout=None: FakeResp(404),
    )
    assert collector.fetch_repo("o/missing", "2025-01-01T00:00:00Z") == []
