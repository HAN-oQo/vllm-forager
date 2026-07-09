"""Tests for the T5.12 tabbed console shell — offline & deterministic.

Per the DEVPLAN todo: each tab route renders its panel from seeded data. Split into two
layers, mirroring `tests/test_dashboard.py`'s own established split: pure `render_tab_page`
tests (no socket) for panel content, and `_make_handler`-level tests (a fake handler, no real
socket — same `_FakeResponse` shape `test_dashboard.py` already uses) for routing + the two
POST write actions.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dashboard import render, server
from src import gate, pr_followup, report
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m5

_ITEM = {
    "repo": "o/r",
    "number": 1,
    "type": "issue",
    "title": "vLLM crashes on gfx90a with fp8",
    "body": "Running fp8 quant on MI250 raises an assertion.",
    "state": "open",
    "url": "https://github.com/o/r/issues/1",
    "created_at": "2026-01-01T00:00:00Z",
    "category": "build",
    "path": ["build"],
}


def _seeded_store(tmp_path: Path) -> JsonlStore:
    store = JsonlStore(tmp_path)
    store.upsert_items([_ITEM])
    return store


@pytest.fixture(autouse=True)
def _mock_risk_score(monkeypatch: pytest.MonkeyPatch):
    """Isolates every test from the real LLM call `gate._risk_badge` makes (mirrors
    `test_dashboard_review.py`/`test_dashboard_attempts.py`'s own autouse fixture)."""
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )


# --------------------------------------------------------------------- render_tab_page: issues


def test_issues_tab_renders_tree_and_nav(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    page = render.render_tab_page(store, "issues")

    assert '<span class="name">build</span>' in page
    assert '<a href="/tab/issues" class="active">Issues</a>' in page
    assert '<a href="/tab/reports" class="">Reports/Trends</a>' in page


def test_every_tab_page_carries_the_build_stamp_footer(tmp_path: Path) -> None:
    """T5.15's own DEVPLAN test line: a smoke check that the served page carries the current
    build/commit stamp -- so staleness (a merge that landed without a dashboard restart) is
    visible on the page itself, not just discoverable by shelling into the host."""
    from src.version import GIT_SHA

    store = _seeded_store(tmp_path)

    page = render.render_tab_page(store, "issues")

    assert f'<footer class="build-stamp">build {GIT_SHA}</footer>' in page


def test_unknown_tab_name_falls_back_to_issues(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    page = render.render_tab_page(store, "not-a-real-tab")

    assert '<a href="/tab/issues" class="active">Issues</a>' in page


# --------------------------------------------------------------------- render_tab_page: reports


def test_reports_tab_renders_trends_forecasts_and_archive_links(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    report.generate_tree(
        store, store.data_dir / "reports", when=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )

    page = render.render_tab_page(store, "reports")

    assert "build" in page  # trends section
    assert "Reports archive" in page
    assert "2026-W02" in page


def test_reports_tab_opens_a_named_archived_report(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    report.generate_tree(
        store, store.data_dir / "reports", when=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )

    page = render.render_tab_page(store, "reports", report_stamp="2026-W02")

    assert '<span class="name">build</span>' in page  # the opened report's own tree


def test_reports_tab_unknown_report_stamp_shows_placeholder(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    report.generate_tree(
        store, store.data_dir / "reports", when=datetime(2026, 1, 5, tzinfo=timezone.utc)
    )

    page = render.render_tab_page(store, "reports", report_stamp="2026-W99")

    assert "Unknown report." in page


# --------------------------------------------------------------------- render_tab_page: candidates


def _fake_candidate(**overrides) -> dict:
    base = {
        "title": "Fix fp8 dispatch on gfx90a",
        "source": "vllm-project/vllm#1",
        "risk": "low",
        "effort": "medium",
        "impact": "high",
        "evidence": "https://github.com/o/r/issues/1",
        "boost": 0,
        "priority": 1.0,
    }
    base.update(overrides)
    return base


def test_candidates_tab_renders_queue_with_work_this_and_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    monkeypatch.setattr(render.api, "candidates", lambda store, **kw: [_fake_candidate()])

    page = render.render_tab_page(store, "candidates")

    assert "Fix fp8 dispatch on gfx90a" in page
    assert 'class="chip risk-low">low risk</span>' in page
    assert 'name="decision" value="selected"' in page
    assert 'name="decision" value="skip"' in page
    assert '<input type="hidden" name="repo" value="o/r">' in page
    assert '<input type="hidden" name="number" value="1">' in page


def test_candidates_tab_omits_actions_when_evidence_unparseable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    monkeypatch.setattr(
        render.api, "candidates", lambda store, **kw: [_fake_candidate(evidence="not-a-url")]
    )

    page = render.render_tab_page(store, "candidates")

    assert "Fix fp8 dispatch on gfx90a" in page
    assert 'name="decision"' not in page


def test_candidates_tab_shows_current_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    monkeypatch.setattr(render.api, "candidates", lambda store, **kw: [_fake_candidate()])
    from dashboard import select

    select.select_candidate(store, "o/r", 1)

    page = render.render_tab_page(store, "candidates")

    assert 'class="chip decision-selected">selected</span>' in page


def test_candidates_tab_empty_when_none_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    monkeypatch.setattr(render.api, "candidates", lambda store, **kw: [])

    page = render.render_tab_page(store, "candidates")

    assert "No candidates discovered." in page


def test_candidates_tab_after_decision_skips_the_expensive_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: landing on the Candidates tab right after a decision (server.py's own
    ?after_decision=1 redirect) must not re-pay api.candidates(compute=True)'s LLM-scoring
    pass -- a code-review finding that the original version reintroduced an O(N) recompute
    after every single "Work this"/"Skip" click."""
    store = _seeded_store(tmp_path)
    monkeypatch.setattr(
        render.api, "candidates", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )

    page = render.render_tab_page(store, "candidates", after_decision=True)

    assert "Decision recorded." in page
    assert "Refresh the queue" in page


def test_candidates_tab_ordinary_navigation_still_computes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    monkeypatch.setattr(render.api, "candidates", lambda store, **kw: [_fake_candidate()])

    page = render.render_tab_page(store, "candidates", after_decision=False)

    assert "Fix fp8 dispatch on gfx90a" in page


# --------------------------------------------------------------------- render_tab_page: attempts


def _store_with_verify(tmp_path: Path, *, verified: bool) -> JsonlStore:
    store = _seeded_store(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "verify",
            "branch": "forager/o-r-1",
            "patch": "--- a/x.py\n+++ b/x.py\n",
            "command": "pytest",
            "log": "1 passed\n" if verified else "still failing\n",
            "verified": verified,
            "recorded_at": "2026-01-02T00:00:00Z",
        }
    )
    return store


def _make_gate_ready(store: JsonlStore) -> None:
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "self_review",
            "critiques": [{"looks_correct": True, "reason": "ok"}],
            "approve_count": 4,
            "total_votes": 5,
            "advance": True,
            "verify_recorded_at": "2026-01-02T00:00:00Z",
            "recorded_at": "2026-01-03T00:00:00Z",
        }
    )


def test_attempts_tab_lists_worked_candidates(tmp_path: Path) -> None:
    store = _store_with_verify(tmp_path, verified=True)

    page = render.render_tab_page(store, "attempts")

    assert "🟢 verified" in page
    assert "o/r#1" in page


def test_attempts_tab_opens_detail_with_approve_form_when_gate_ready(tmp_path: Path) -> None:
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)

    page = render.render_tab_page(store, "attempts", open_candidate=("o/r", 1))

    assert "Issue overview" in page
    assert 'name="approve" value="true"' in page
    assert 'name="approve" value="false"' in page


def test_attempts_tab_detail_omits_approve_form_when_not_gate_ready(tmp_path: Path) -> None:
    store = _store_with_verify(tmp_path, verified=True)

    page = render.render_tab_page(store, "attempts", open_candidate=("o/r", 1))

    assert "Issue overview" in page
    assert 'name="approve"' not in page


def test_attempts_tab_unknown_open_candidate_shows_placeholder(tmp_path: Path) -> None:
    store = _store_with_verify(tmp_path, verified=True)

    page = render.render_tab_page(store, "attempts", open_candidate=("o/r", 999))

    assert "Unknown attempt." in page


def test_attempts_tab_after_decision_skips_the_review_bundle_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: landing on an attempt's detail view right after a T5.6 decision (server.py's
    own ?after_decision=1 redirect) must not re-pay review_bundle's real LLM call -- a
    code-review finding that the original version re-fetched the bundle purely to redisplay
    content the human had just acted on."""
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)
    monkeypatch.setattr(gate, "_score", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    page = render.render_tab_page(store, "attempts", open_candidate=("o/r", 1), after_decision=True)

    assert "Issue overview" in page
    assert "Decision recorded." in page
    assert 'name="approve"' not in page


# --------------------------------------------------------------------- render_tab_page: prs


def test_upstream_prs_tab_empty_when_nothing_submitted(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    page = render.render_tab_page(store, "prs")

    assert "No open upstream PRs right now." in page


def test_upstream_prs_tab_renders_an_open_pr_with_outstanding_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "gate",
            "approved": True,
            "submitted": True,
            "pr_url": "https://github.com/o/r/pull/1",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    pr_json = {
        "url": "https://github.com/o/r/pull/1",
        "state": "OPEN",
        "mergeable": "MERGEABLE",
        "reviewDecision": "REVIEW_REQUIRED",
        "updatedAt": "2026-01-05T00:00:00Z",
        "statusCheckRollup": [{"conclusion": "FAILURE", "status": "COMPLETED"}],
        "reviewRequests": [{"login": "octocat"}],
    }
    comments = [{"id": 101, "user": {"login": "maintainer"}, "body": "please fix the docs"}]

    def _run(cmd, **kwargs):
        if cmd[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(pr_json))
        if cmd[:2] == ["gh", "api"] and cmd[2] == "user":
            return subprocess.CompletedProcess(cmd, 0, stdout="forager-bot\n")
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/comments"):
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(comments))
        raise AssertionError(f"unexpected gh invocation: {cmd}")

    monkeypatch.setattr(pr_followup.subprocess, "run", _run)

    page = render.render_tab_page(store, "prs")

    assert "o/r#1" in page
    assert "CI: failure" in page
    assert "1 outstanding" in page
    assert "octocat" in page
    assert "please fix the docs" in page


# --------------------------------------------------------------------- render_tab_page: ops


def test_ops_tab_renders_health_and_guardrail_sections(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)

    page = render.render_tab_page(store, "ops")

    assert "Live health" in page
    assert "Guardrails — data quality" in page
    assert "Guardrails — RAG eval" in page
    assert "never run" in page or "ok" in page or "chip" in page  # some status rendered


def test_ops_tab_shows_output_tail_and_error_for_a_failed_stage(tmp_path: Path) -> None:
    """Regression: a stalled/failed stage must show its intermediate output and error, not
    just a bare status word -- CLAUDE.md's own T5.8 guardrail ("the dashboard always shows
    what is running now... and its partial output. A crash must never leave a stage silently
    'running'."), a code-review finding that the original _health_row_html dropped both."""
    store = _seeded_store(tmp_path)
    store.record_run(
        {
            "stage": "collect",
            "status": "failed",
            "step": "fetch_issues",
            "output_tail": "Traceback (most recent call last):\nRateLimitError",
            "error": "RateLimitError: secondary rate limit hit",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )

    page = render.render_tab_page(store, "ops")

    assert "RateLimitError: secondary rate limit hit" in page
    assert "Traceback (most recent call last):" in page


def test_ops_tab_renders_cost_panel_next_to_health(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    store.record_run(
        {
            "stage": "cost",
            "agent": "analyst",
            "run_id": "r1",
            "loop": "intel",
            "provider": "claude_api",
            "model": "claude-sonnet-5",
            "tokens_in": 100,
            "tokens_out": 50,
            "tokens_cache": 0,
            "cost_usd": 3.5,
            "recorded_at": "2026-01-08T00:00:00Z",
        }
    )

    page = render.render_tab_page(store, "ops")

    assert "Live health" in page
    assert page.index("Live health") < page.index("Cost")  # cost sits next to health
    assert "analyst" in page
    assert "$3.50" in page


def test_ops_tab_fetches_list_runs_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: render_ops_tab must share one store.list_runs() fetch between the T5.8
    health panel and the T5.13 cost panel, not let each call fetch independently -- the exact
    "N separate reads" anti-pattern this milestone's own review has already caught and fixed
    for a single panel (dashboard.health's own docstring); this locks it in across panels."""
    store = _seeded_store(tmp_path)
    call_count = 0
    real_list_runs = store.list_runs

    def _counting_list_runs(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_list_runs(*args, **kwargs)

    monkeypatch.setattr(store, "list_runs", _counting_list_runs)

    render.render_tab_page(store, "ops")

    assert call_count == 1


def test_budget_exceeded_chip_has_a_real_css_rule(tmp_path: Path) -> None:
    """Regression: the "gap" chip class used for a stalled/failed health row and an exceeded
    budget must actually be styled -- an earlier version used class="chip gap" with no
    matching CSS rule anywhere, so the "turns red" behavior this todo's whole point silently
    never rendered."""
    store = _seeded_store(tmp_path)

    page = render.render_tab_page(store, "ops")

    assert ".chip.gap" in page or ".chip.risk-high,.chip.decision-skip,.chip.gap" in page


# --------------------------------------------------------------------- server: GET routing


class _FakeHandler:
    """A duck-typed stand-in for `BaseHTTPRequestHandler`'s do_GET/do_POST surface — mirrors
    `tests/test_dashboard.py`'s own `_FakeResponse`, extended with a readable `rfile` and a
    `headers` mapping so `_read_form` (a POST body reader) can be exercised without a socket.
    """

    def __init__(self, path: str = "/", *, body: bytes = b"") -> None:
        self.path = path
        self.status: int | None = None
        self.headers: dict[str, str] = {"Content-Length": str(len(body))}
        self.response_headers: dict[str, str] = {}
        self.body = b""
        self._rfile_data = body

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, key: str, value: str) -> None:
        self.response_headers[key] = value

    def end_headers(self) -> None:
        pass

    class _RFile:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self, n: int) -> bytes:
            return self._data[:n]

    class _WFile:
        def __init__(self, outer: _FakeHandler) -> None:
            self._outer = outer

        def write(self, data: bytes) -> None:
            self._outer.body += data

    @property
    def rfile(self) -> _FakeHandler._RFile:
        return self._RFile(self._rfile_data)

    @property
    def wfile(self) -> _FakeHandler._WFile:
        return self._WFile(self)


def test_do_get_root_serves_issues_tab(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    handler_cls = server._make_handler(store, None)
    fake = _FakeHandler("/")

    handler_cls.do_GET(fake)  # type: ignore[arg-type]

    assert fake.status == 200
    assert b'<span class="name">build</span>' in fake.body


def test_do_get_tab_path_serves_named_tab(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    handler_cls = server._make_handler(store, None)
    fake = _FakeHandler("/tab/ops")

    handler_cls.do_GET(fake)  # type: ignore[arg-type]

    assert fake.status == 200
    assert b"Live health" in fake.body


def test_do_get_unknown_tab_path_falls_back_to_issues(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    handler_cls = server._make_handler(store, None)
    fake = _FakeHandler("/tab/not-a-real-tab")

    handler_cls.do_GET(fake)  # type: ignore[arg-type]

    assert fake.status == 200
    assert b'<span class="name">build</span>' in fake.body


def test_do_get_attempts_tab_reads_repo_and_number_query_params(tmp_path: Path) -> None:
    store = _store_with_verify(tmp_path, verified=True)
    handler_cls = server._make_handler(store, None)
    fake = _FakeHandler("/tab/attempts?repo=o%2Fr&number=1")

    handler_cls.do_GET(fake)  # type: ignore[arg-type]

    assert fake.status == 200
    assert b"Issue overview" in fake.body


# --------------------------------------------------------------------- server: POST write actions


def test_do_post_candidate_decision_selected_records_and_redirects(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    handler_cls = server._make_handler(store, None)
    body = b"repo=o%2Fr&number=1&decision=selected"
    fake = _FakeHandler("/tab/candidates/decide", body=body)

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 303
    assert fake.response_headers["Location"] == "/tab/candidates?after_decision=1"
    from src.selection import latest_decision

    assert latest_decision(store, "o/r", 1) == "selected"


def test_do_post_candidate_decision_skip_records_and_redirects(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    handler_cls = server._make_handler(store, None)
    body = b"repo=o%2Fr&number=1&decision=skip"
    fake = _FakeHandler("/tab/candidates/decide", body=body)

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 303
    from src.selection import latest_decision

    assert latest_decision(store, "o/r", 1) == "skip"


def test_do_post_candidate_decision_missing_fields_is_bad_request(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    handler_cls = server._make_handler(store, None)
    fake = _FakeHandler("/tab/candidates/decide", body=b"repo=o%2Fr")

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 400


def test_do_post_candidate_decision_unknown_item_is_not_found(tmp_path: Path) -> None:
    """Regression: select_candidate/skip_candidate return None for an item that isn't a real
    KB record (src.selection.record_decision's own documented contract) -- the handler must
    surface that as an error, not a silent 303 as if the write had succeeded."""
    store = JsonlStore(tmp_path)  # no items upserted at all
    handler_cls = server._make_handler(store, None)
    body = b"repo=o%2Fr&number=1&decision=selected"
    fake = _FakeHandler("/tab/candidates/decide", body=body)

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 404


def test_do_post_candidate_decision_store_failure_is_internal_server_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _seeded_store(tmp_path)
    handler_cls = server._make_handler(store, None)
    monkeypatch.setattr(
        server.select,
        "select_candidate",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    body = b"repo=o%2Fr&number=1&decision=selected"
    fake = _FakeHandler("/tab/candidates/decide", body=body)

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 500


def test_do_post_attempt_decision_approve_records_and_redirects(tmp_path: Path) -> None:
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)
    handler_cls = server._make_handler(store, tmp_path)
    body = b"repo=o%2Fr&number=1&approve=true"
    fake = _FakeHandler("/tab/attempts/decide", body=body)

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 303
    assert fake.response_headers["Location"] == "/tab/attempts?repo=o/r&number=1&after_decision=1"
    runs = store.list_runs(repo="o/r", number=1, stage="gate")
    assert runs[-1]["approved"] is True


def test_do_post_attempt_decision_approve_writes_draft_under_the_dashboards_own_data_dir(
    tmp_path: Path,
) -> None:
    """Regression: the approve action must resolve pr_drafts_dir against the dashboard's own
    data_dir (threaded from dashboard.__main__ through serve()/_make_handler), not gate.py's
    unrelated module-level default -- a code-review finding."""
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)
    handler_cls = server._make_handler(store, tmp_path)
    body = b"repo=o%2Fr&number=1&approve=true"
    fake = _FakeHandler("/tab/attempts/decide", body=body)

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 303
    assert (tmp_path / "pr_drafts" / "o-r-1.md").exists()


def test_do_post_attempt_decision_hold_records_without_writing_a_draft(tmp_path: Path) -> None:
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)
    handler_cls = server._make_handler(store, tmp_path)
    body = b"repo=o%2Fr&number=1&approve=false"
    fake = _FakeHandler("/tab/attempts/decide", body=body)

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 303
    runs = store.list_runs(repo="o/r", number=1, stage="gate")
    assert runs[-1]["approved"] is False
    assert not (tmp_path / "pr_drafts" / "o-r-1.md").exists()


def test_do_post_attempt_decision_not_gate_ready_is_conflict(tmp_path: Path) -> None:
    """Regression: review_decision returns None when the candidate isn't gate-ready (e.g. a
    race with a fresh verify run) -- the handler must surface that, not a silent 303."""
    store = _store_with_verify(tmp_path, verified=True)  # verified, but no self_review yet
    handler_cls = server._make_handler(store, tmp_path)
    body = b"repo=o%2Fr&number=1&approve=true"
    fake = _FakeHandler("/tab/attempts/decide", body=body)

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 409


def test_do_post_attempt_decision_store_failure_is_internal_server_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)
    handler_cls = server._make_handler(store, tmp_path)
    monkeypatch.setattr(
        server.review,
        "review_decision",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    body = b"repo=o%2Fr&number=1&approve=true"
    fake = _FakeHandler("/tab/attempts/decide", body=body)

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 500


def test_do_post_unknown_path_is_404(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    handler_cls = server._make_handler(store, None)
    fake = _FakeHandler("/tab/candidates", body=b"")

    handler_cls.do_POST(fake)  # type: ignore[arg-type]

    assert fake.status == 404
