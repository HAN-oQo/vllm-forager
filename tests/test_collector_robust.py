"""Robustness tests for the collector (T0.10). Offline & deterministic (no network / sleeps).

Deep-pagination cursor-windowing and per-repo failure isolation shipped first (#3); this file
also covers the rest of T0.10: retry/backoff on transient failures (5xx / connection /
timeout), primary + secondary (`Retry-After`) rate-limit handling, record schema validation
(log + skip malformed), and the configurable `body` cap.
"""

import json
import sys
from datetime import datetime, timezone

import pytest
import requests

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
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _rec(n, t):
    # A minimal-but-valid raw GitHub item (html_url present so it survives schema validation).
    return {"number": n, "title": str(n), "updated_at": t, "html_url": f"http://x/{n}"}


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


def test_merge_jsonl_preserves_unicode_line_separator(tmp_path):
    """A body containing a literal U+2028 must survive a write→re-read merge cycle.

    Records are written with ensure_ascii=False, so U+2028/U+2029/U+0085 land literally in
    the file. Re-reading with str.splitlines() (the old bug) would shatter such a record into
    unparseable fragments and silently drop it; splitting on "\\n" keeps it intact.
    """
    path = tmp_path / "o__r.jsonl"
    body = "line one\u2028line two"  # literal U+2028 LINE SEPARATOR inside the body
    rec = {"number": 1, "title": "t", "updated_at": "2025-01-01T00:00:00Z", "body": body}
    assert collector._merge_jsonl(path, [rec]) == 1
    # Second merge re-reads the file it just wrote; the U+2028 record must not be lost.
    assert collector._merge_jsonl(path, []) == 1
    kept = json.loads(path.read_text().rstrip("\n").split("\n")[0])
    assert kept["body"] == body


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


# ===================================================================================
# T0.10 remaining: retry/backoff, secondary rate limits, schema validation, body cap.
# ===================================================================================


@pytest.fixture
def no_sleep(monkeypatch):
    """Patch out real sleeping; return the list of recorded wait durations."""
    waits: list[float] = []
    monkeypatch.setattr(collector.time, "sleep", lambda s: waits.append(s))
    return waits


def _seq_get(monkeypatch, items):
    """Patch requests.get to return/raise successive `items` (Exception instances are raised)."""
    it = iter(items)

    def fake_get(*args, **kwargs):
        nxt = next(it)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(collector.requests, "get", fake_get)


# --------------------------------------------------------------- retry / backoff


def test_request_retries_5xx_then_succeeds(monkeypatch, no_sleep):
    _seq_get(monkeypatch, [FakeResp(500), FakeResp(502), FakeResp(200, [{"ok": 1}])])
    resp = collector._request("http://x", {})
    assert resp.status_code == 200
    assert len(no_sleep) == 2  # backed off twice before the 200


def test_request_retries_connection_error_then_succeeds(monkeypatch, no_sleep):
    _seq_get(monkeypatch, [requests.ConnectionError("reset"), FakeResp(200, [])])
    assert collector._request("http://x", {}).status_code == 200
    assert len(no_sleep) == 1


def test_request_gives_up_after_max_retries(monkeypatch, no_sleep):
    monkeypatch.setattr(config, "MAX_RETRIES", 2)
    monkeypatch.setattr(collector.requests, "get", lambda *a, **k: FakeResp(503))
    with pytest.raises(requests.HTTPError):
        collector._request("http://x", {})
    assert len(no_sleep) == 2  # retried MAX_RETRIES times, then raised


def test_request_backoff_is_exponential(monkeypatch, no_sleep):
    monkeypatch.setattr(config, "BACKOFF_BASE_S", 1.0)
    _seq_get(monkeypatch, [FakeResp(500), FakeResp(500), FakeResp(200, [])])
    collector._request("http://x", {})
    assert no_sleep == [1.0, 2.0]  # 1*2**0, 1*2**1


# ------------------------------------------------------------------ rate limits


def test_request_honors_secondary_rate_limit_retry_after(monkeypatch, no_sleep):
    _seq_get(monkeypatch, [FakeResp(403, headers={"Retry-After": "3"}), FakeResp(200, [])])
    resp = collector._request("http://x", {})
    assert resp.status_code == 200
    assert no_sleep == [4.0]  # 3 + 1s cushion; a rate-limit wait is not a retry attempt


def test_request_waits_for_primary_rate_limit_then_succeeds(monkeypatch, no_sleep):
    monkeypatch.setattr(collector.time, "time", lambda: 1000)
    limited = FakeResp(403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1002"})
    _seq_get(monkeypatch, [limited, FakeResp(200, [])])
    assert collector._request("http://x", {}).status_code == 200
    assert no_sleep == [3]  # (1002 - 1000) + 1


def test_request_403_without_retry_after_is_returned_not_retried(monkeypatch, no_sleep):
    # A real 403 (no rate-limit signal) is returned so the caller can surface it — not looped on.
    monkeypatch.setattr(collector.requests, "get", lambda *a, **k: FakeResp(403))
    resp = collector._request("http://x", {})
    assert resp.status_code == 403
    assert no_sleep == []


def test_request_gives_up_on_persistent_rate_limit(monkeypatch, no_sleep):
    # A limiter that never clears (Retry-After on every response) must not spin forever: after
    # MAX_RATE_LIMIT_RETRIES waits, _request raises so main()'s per-repo isolation can skip it.
    monkeypatch.setattr(config, "MAX_RATE_LIMIT_RETRIES", 3)
    monkeypatch.setattr(
        collector.requests, "get", lambda *a, **k: FakeResp(429, headers={"Retry-After": "1"})
    )
    with pytest.raises(requests.HTTPError):
        collector._request("http://x", {})
    assert len(no_sleep) == 3  # waited the cap, then raised instead of looping forever


# ------------------------------------------------- schema validation / body cap


def test_fetch_repo_skips_malformed_record(monkeypatch, capsys):
    monkeypatch.setattr(config, "PER_PAGE", 10)
    batch = [
        {
            "number": 1,
            "title": "ok",
            "updated_at": "2025-01-01T00:00:00Z",
            "html_url": "http://x/1",
        },
        {"number": 2, "title": "no url", "updated_at": "2025-01-02T00:00:00Z"},  # missing html_url
    ]
    monkeypatch.setattr(collector.requests, "get", lambda *a, **k: FakeResp(200, batch))
    out = collector.fetch_repo("o/r", "2025-01-01T00:00:00Z")
    assert [r["number"] for r in out] == [1]  # #2 skipped — url is None
    err = capsys.readouterr().err
    assert "skipping malformed item #2" in err and "url" in err


def test_body_cap_is_configurable(monkeypatch):
    monkeypatch.setattr(config, "BODY_MAX_CHARS", 10)
    rec = collector._normalize({"number": 1, "body": "y" * 100}, "o/r")
    assert len(rec["body"]) == 10


# ------------------------------------------------------------- monotonic cursor


def test_state_cursor_is_monotonic(tmp_path, monkeypatch):
    """A successful run only advances a repo's cursor forward (never backwards)."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(config, "REPOS", [{"slug": "o/r", "role": "x"}])

    old_cursor = "2000-01-01T00:00:00Z"
    collector._save_state({"o/r": old_cursor})
    monkeypatch.setattr(collector, "fetch_repo", lambda slug, since: [])
    collector.main([])  # exercise the argv seam (no sys.argv monkeypatch needed)

    new_cursor = json.loads((tmp_path / "state.json").read_text())["o/r"]
    assert new_cursor > old_cursor  # advanced forward — never rewound
