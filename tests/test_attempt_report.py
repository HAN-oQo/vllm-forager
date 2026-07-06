"""Tests for the attempt report (T3.12) — offline & deterministic.

Per the DEVPLAN todo: a mock verified run and a mock failed run each produce a report with all
3 sections + outcome + a runnable reproduce block; failure reports are still written.
"""

import pytest

from src import attempt_report, gate
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m3

_ITEM = {
    "repo": "o/r",
    "number": 1,
    "type": "issue",
    "title": "vLLM crashes on gfx90a with fp8",
    "body": "Running fp8 quant on MI250 raises an assertion.",
    "state": "open",
    "url": "https://github.com/o/r/issues/1",
}


def _store_with_verify(tmp_path, *, verified: bool, repro=True) -> JsonlStore:
    store = JsonlStore(tmp_path)
    store.upsert_items([_ITEM])
    if repro:
        store.record_run(
            {
                "repo": "o/r",
                "number": 1,
                "stage": "repro",
                "command": "pytest test_fp8.py",
                "log": "AssertionError: fp8 dispatch failed\n",
                "reproduced": True,
                "recorded_at": "2025-12-30T00:00:00Z",
            }
        )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "verify",
            "branch": "forager/o-r-1",
            "patch": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good\n",
            "command": "pytest test_fp8.py",
            "log": "1 passed\n" if verified else "AssertionError: fp8 dispatch failed\n",
            "verified": verified,
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    return store


@pytest.fixture(autouse=True)
def _isolate_attempts_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Every write test produces a real file -- redirect the module-level default the same way
    `gate.py`'s own `_isolate_pr_drafts_dir` fixture does, so tests never touch the real shared
    `config.DATA_DIR`."""
    monkeypatch.setattr(attempt_report, "_ATTEMPTS_DIR", tmp_path / "attempts")


# --------------------------------------------------------------------- render_attempt_report


def test_render_returns_none_when_no_item(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    assert attempt_report.render_attempt_report(store, "o/r", 1) is None


def test_render_returns_none_when_no_verify_run(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_ITEM])
    assert attempt_report.render_attempt_report(store, "o/r", 1) is None


def test_verified_report_has_all_sections_and_runnable_reproduce_block(tmp_path) -> None:
    store = _store_with_verify(tmp_path, verified=True)

    report = attempt_report.render_attempt_report(store, "o/r", 1)

    assert report is not None
    assert report.verified is True
    assert "vLLM crashes on gfx90a with fp8" in report.issue_overview
    assert "https://github.com/o/r/issues/1" in report.issue_overview
    assert "forager/o-r-1" in report.approach
    assert "```bash" in report.reproduce
    assert "git apply" in report.reproduce
    assert "pytest test_fp8.py" in report.reproduce
    assert "VERIFIED" in report.outcome


def test_failed_report_still_produced_with_all_sections(tmp_path) -> None:
    """Regression for the DEVPLAN's own explicit ask: failure reports are still written, with
    the same 3 sections + outcome + reproduce block as a verified one."""
    store = _store_with_verify(tmp_path, verified=False)

    report = attempt_report.render_attempt_report(store, "o/r", 1)

    assert report is not None
    assert report.verified is False
    assert report.issue_overview
    assert report.approach
    assert "```bash" in report.reproduce
    assert "FAILED" in report.outcome
    assert "AssertionError" in report.outcome


def test_approach_mentions_repro_when_a_repro_run_exists(tmp_path) -> None:
    store = _store_with_verify(tmp_path, verified=True, repro=True)
    report = attempt_report.render_attempt_report(store, "o/r", 1)
    assert report is not None
    assert "Reproduced the failure" in report.approach


def test_approach_omits_repro_mention_when_no_repro_run(tmp_path) -> None:
    store = _store_with_verify(tmp_path, verified=True, repro=False)
    report = attempt_report.render_attempt_report(store, "o/r", 1)
    assert report is not None
    assert "Reproduced the failure" not in report.approach


def test_reproduce_block_has_no_git_apply_when_patch_missing(tmp_path) -> None:
    store = _store_with_verify(tmp_path, verified=True)
    # overwrite the verify run with one that has no patch captured
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "verify",
            "branch": "forager/o-r-1",
            "patch": "",
            "command": "pytest test_fp8.py",
            "log": "1 passed\n",
            "verified": True,
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    report = attempt_report.render_attempt_report(store, "o/r", 1)
    assert report is not None
    assert "git apply" not in report.reproduce
    assert "pytest test_fp8.py" in report.reproduce


# --------------------------------------------------------------------- rendered HTML (gate-ready)


def test_verified_but_not_gate_ready_has_no_rendered_html(tmp_path) -> None:
    """Verified alone isn't enough for the rendered view -- it also needs an *advancing*
    self-review for that same verify run (gate.py's own readiness bar)."""
    store = _store_with_verify(tmp_path, verified=True)
    report = attempt_report.render_attempt_report(store, "o/r", 1)
    assert report is not None
    assert report.rendered_html is None
    assert "Rendered draft" not in attempt_report._format_markdown(report)


def _make_gate_ready(store) -> None:
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "self_review",
            "critiques": [{"looks_correct": True, "reason": "ok"}],
            "approve_count": 4,
            "total_votes": 5,
            "advance": True,
            "verify_recorded_at": "2025-12-31T00:00:00Z",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )


def test_gate_ready_candidate_gets_rendered_html(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)

    report = attempt_report.render_attempt_report(store, "o/r", 1)

    assert report is not None
    assert report.rendered_html is not None
    assert "vLLM crashes on gfx90a with fp8" in report.rendered_html
    assert "Self-review: 4/5 approve" in report.rendered_html
    assert "Rendered draft" in attempt_report._format_markdown(report)


def test_gate_ready_html_uses_composed_pr_author_narrative_when_present(tmp_path) -> None:
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "[Bugfix] composed title",
            "body": "composed body text",
            "verify_recorded_at": "2025-12-31T00:00:00Z",
            "recorded_at": "2026-01-02T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_quality",
            "votes": [{"acceptable": True, "reason": "ok"}],
            "approve_count": 1,
            "total_votes": 1,
            "passes": True,
            "pr_author_recorded_at": "2026-01-02T00:00:00Z",
            "recorded_at": "2026-01-03T00:00:00Z",
        }
    )

    report = attempt_report.render_attempt_report(store, "o/r", 1)

    assert report is not None
    assert report.rendered_html is not None
    assert "composed title" in report.rendered_html
    assert "composed body text" in report.rendered_html
    assert "Quality gate: PASSED" in report.rendered_html


def test_gate_ready_html_ignores_self_review_from_a_different_verify_run(tmp_path) -> None:
    """Regression: a self-review run that doesn't match the *current* verify run's
    `verify_recorded_at` must never be shown, even if it's the most recently recorded one --
    mirrors `gate._readiness`'s own matching, which this module's HTML render must not bypass."""
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)  # approve_count=4/5, matches verify_recorded_at 2025-12-31
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "self_review",
            "critiques": [{"looks_correct": False, "reason": "stale"}],
            "approve_count": 1,
            "total_votes": 1,
            "advance": False,
            "verify_recorded_at": "2020-01-01T00:00:00Z",  # a different, stale verify run
            "recorded_at": "2026-06-01T00:00:00Z",  # recorded *after* the matching one
        }
    )

    report = attempt_report.render_attempt_report(store, "o/r", 1)

    assert report is not None
    assert report.rendered_html is not None
    assert "Self-review: 4/5 approve" in report.rendered_html
    assert "1/1" not in report.rendered_html


def test_gate_ready_html_ignores_pr_author_and_pr_quality_from_a_different_verify_run(
    tmp_path,
) -> None:
    """Regression: a `pr_author`/`pr_quality` pair composed/judged for an older verify run must
    never be shown against the current diff after a re-verification -- the exact staleness bug
    `gate._current_narrative` guards against."""
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "[Bugfix] stale title",
            "body": "stale composed body",
            "verify_recorded_at": "2020-01-01T00:00:00Z",  # a different, stale verify run
            "recorded_at": "2026-06-01T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_quality",
            "votes": [{"acceptable": True, "reason": "stale"}],
            "approve_count": 1,
            "total_votes": 1,
            "passes": True,
            "pr_author_recorded_at": "2026-06-01T00:00:00Z",
            "recorded_at": "2026-06-02T00:00:00Z",
        }
    )

    report = attempt_report.render_attempt_report(store, "o/r", 1)

    assert report is not None
    assert report.rendered_html is not None
    assert "stale title" not in report.rendered_html
    assert "stale composed body" not in report.rendered_html
    assert "No composed PR narrative" in report.rendered_html
    assert "Quality gate: (no record)" in report.rendered_html


def test_gate_ready_html_quality_gate_requires_passes_is_true_not_truthy(tmp_path) -> None:
    """Regression: `passes` must be read via `is True`, not plain truthiness, so a malformed
    non-bool value can never be coerced into an accidental PASSED -- mirrors
    `gate._current_narrative`'s own guard for the identical field."""
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "title",
            "body": "body",
            "verify_recorded_at": "2025-12-31T00:00:00Z",
            "recorded_at": "2026-01-02T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_quality",
            "votes": [{"acceptable": True, "reason": "ok"}],
            "approve_count": 1,
            "total_votes": 1,
            "passes": "true",  # malformed: truthy string, not an actual bool
            "pr_author_recorded_at": "2026-01-02T00:00:00Z",
            "recorded_at": "2026-01-03T00:00:00Z",
        }
    )

    report = attempt_report.render_attempt_report(store, "o/r", 1)

    assert report is not None
    assert report.rendered_html is not None
    assert "Quality gate: NOT PASSED" in report.rendered_html


def test_gate_ready_html_uses_safe_href_for_evidence_link(tmp_path) -> None:
    """Regression: `evidence_url(item)` lands straight into an `<a href>` -- a malformed
    non-http(s) URL in the KB record must not become a clickable link (mirrors
    `dashboard/render.py`'s own `_safe_href` guard for the identical situation)."""
    store = JsonlStore(tmp_path)
    store.upsert_items([{**_ITEM, "url": "javascript:alert(1)"}])
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "verify",
            "branch": "forager/o-r-1",
            "patch": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good\n",
            "command": "pytest test_fp8.py",
            "log": "1 passed\n",
            "verified": True,
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    _make_gate_ready(store)

    report = attempt_report.render_attempt_report(store, "o/r", 1)

    assert report is not None
    assert report.rendered_html is not None
    assert 'href="javascript:alert(1)"' not in report.rendered_html
    assert '<a href="">' in report.rendered_html


def test_gate_ready_html_escapes_untrusted_title_and_body(tmp_path) -> None:
    """The composed title/body are LLM/human-authored text landing straight into an HTML
    document -- must not be interpretable as markup."""
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)
    store.record_run(
        {
            "repo": "o/r",
            "number": 1,
            "stage": "pr_author",
            "title": "<script>alert(1)</script>",
            "body": "<img src=x onerror=alert(1)>",
            "verify_recorded_at": "2025-12-31T00:00:00Z",
            "recorded_at": "2026-01-02T00:00:00Z",
        }
    )

    report = attempt_report.render_attempt_report(store, "o/r", 1)

    assert report is not None
    assert report.rendered_html is not None
    assert "<script>" not in report.rendered_html
    assert "<img " not in report.rendered_html
    assert "&lt;script&gt;" in report.rendered_html
    assert "&lt;img" in report.rendered_html


# --------------------------------------------------------------------- write_attempt_report


def test_write_attempt_report_writes_markdown_file(tmp_path) -> None:
    store = _store_with_verify(tmp_path, verified=True)
    attempts_dir = tmp_path / "attempts"

    path = attempt_report.write_attempt_report(store, "o/r", 1, attempts_dir=attempts_dir)

    assert path is not None
    assert path == attempts_dir / "o-r-1.md"
    assert path.exists()
    text = path.read_text()
    assert "## Issue overview" in text
    assert "## Approach" in text
    assert "## Reproduce" in text
    assert "## Outcome" in text


def test_write_attempt_report_returns_none_and_writes_nothing_when_not_ready(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items([_ITEM])
    attempts_dir = tmp_path / "attempts"

    path = attempt_report.write_attempt_report(store, "o/r", 1, attempts_dir=attempts_dir)

    assert path is None
    assert not attempts_dir.exists()


def test_write_attempt_report_also_writes_html_when_gate_ready(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )
    store = _store_with_verify(tmp_path, verified=True)
    _make_gate_ready(store)
    attempts_dir = tmp_path / "attempts"

    path = attempt_report.write_attempt_report(store, "o/r", 1, attempts_dir=attempts_dir)

    assert path is not None
    html_path = attempts_dir / "o-r-1.html"
    assert html_path.exists()
    assert "<html>" in html_path.read_text()


def test_write_attempt_report_uses_module_default_dir_when_not_given(tmp_path) -> None:
    """Regression for the None-sentinel convention: a caller that doesn't pass `attempts_dir`
    must land under the (test-isolated) module-level default, not the real shared data dir."""
    store = _store_with_verify(tmp_path, verified=True)

    path = attempt_report.write_attempt_report(store, "o/r", 1)

    assert path is not None
    assert path.parent == attempt_report._ATTEMPTS_DIR


# --------------------------------------------------------------------- CLI


def test_cli_writes_report_and_prints_path(tmp_path, capsys) -> None:
    _store_with_verify(tmp_path, verified=True)
    rc = attempt_report.main(["--candidate", "o/r#1", "--data-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "wrote" in out


def test_cli_returns_nonzero_when_not_ready(tmp_path, capsys) -> None:
    JsonlStore(tmp_path).upsert_items([_ITEM])
    rc = attempt_report.main(["--candidate", "o/r#1", "--data-dir", str(tmp_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "no verify run yet" in out
