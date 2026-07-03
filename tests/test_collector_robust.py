"""Robustness tests for the collector (T0.10): deep-pagination cursor-windowing and
per-repo failure isolation. Offline & deterministic (no network)."""

import json
import sys
from datetime import datetime, timezone

import pytest

from src import collector, config

pytestmark = pytest.mark.m0


class FakeResp:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("unexpected error status")


def _rec(n, t):
    return {"number": n, "title": str(n), "updated_at": t}


def test_fetch_repo_windows_past_page_cap(monkeypatch):
    """>PAGE_CAP pages: the `since` cursor advances instead of hitting the 422 wall."""
    monkeypatch.setattr(config, "PER_PAGE", 2)
    monkeypatch.setattr(collector, "PAGE_CAP", 2)
    pages = {
        ("s0", 1): [_rec(1, "t1"), _rec(2, "t2")],
        ("s0", 2): [_rec(3, "t3"), _rec(4, "t4")],
        ("t4", 1): [_rec(4, "t4"), _rec(5, "t5")],  # inclusive boundary re-fetch of #4
        ("t4", 2): [_rec(6, "t6")],
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        return FakeResp(200, pages.get((params["since"], params["page"]), []))

    monkeypatch.setattr(collector.requests, "get", fake_get)
    out = collector.fetch_repo("o/r", "s0")
    assert sorted(r["number"] for r in out) == [1, 2, 3, 4, 5, 6]  # past 2-page cap, #4 deduped


def test_fetch_repo_stall_guard_terminates(monkeypatch):
    """Every page full + one shared timestamp → cursor can't advance; must stop, not hang."""
    monkeypatch.setattr(config, "PER_PAGE", 2)
    monkeypatch.setattr(collector, "PAGE_CAP", 2)

    def fake_get(url, headers=None, params=None, timeout=None):
        p = params["page"]
        return FakeResp(200, [_rec(10 * p + 1, "t"), _rec(10 * p + 2, "t")])

    monkeypatch.setattr(collector.requests, "get", fake_get)
    out = collector.fetch_repo("o/r", "s0")  # returns instead of looping forever
    assert len(out) >= 1


def test_main_isolates_repo_failure_and_saves_state(tmp_path, monkeypatch):
    """One repo raising must not abort the run or lose a healthy repo's cursor."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(
        config, "REPOS", [{"slug": "o/good", "role": "x"}, {"slug": "o/bad", "role": "y"}]
    )
    monkeypatch.setattr(sys, "argv", ["collector"])

    def fake_fetch(slug, since):
        if slug == "o/bad":
            raise RuntimeError("boom")
        return [_rec(1, "2025-01-01T00:00:00Z") | {"repo": slug}]

    monkeypatch.setattr(collector, "fetch_repo", fake_fetch)
    collector.main()

    assert (tmp_path / "o__good.jsonl").exists()
    state = json.loads((tmp_path / "state.json").read_text())
    assert "o/good" in state and "o/bad" not in state  # failed repo's cursor NOT advanced


def test_main_uses_lookback_window(tmp_path, monkeypatch):
    """With no state, main() defaults to a rolling INITIAL_LOOKBACK_DAYS window."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(config, "REPOS", [{"slug": "o/r", "role": "x"}])
    monkeypatch.setattr(config, "INITIAL_LOOKBACK_DAYS", 180)
    monkeypatch.setattr(sys, "argv", ["collector"])

    seen = {}

    def fake_fetch(slug, since):
        seen["since"] = since
        return []

    monkeypatch.setattr(collector, "fetch_repo", fake_fetch)
    collector.main()

    since = datetime.strptime(seen["since"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - since).days
    assert 179 <= age_days <= 181
