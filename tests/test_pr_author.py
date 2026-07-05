"""Tests for the PR-author agent (T3.9) — offline & deterministic; the LLM call is mocked.

Per the DEVPLAN todo: mock llm + bundle -> body has all required sections, links the issue,
carries Signed-off-by, and every repro/perf claim cites the captured MI250 run.
"""

import pytest

from src import gate, pr_author
from src.pr_profile import RepoProfile
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m3

_FULL_REPLY = {
    "title": "[Bugfix] Fix fp8 assertion on gfx90a",
    "problem": "vLLM crashes with an assertion when running fp8 quantization on MI250.",
    "root_cause": "The fp8 kernel dispatch assumed a CUDA-only code path and never checked ROCm.",
    "fix_rationale": "Adds a gfx90a branch to the dispatch so MI250 takes the correct kernel.",
    "limitations": "Only gfx90a was tested; other ROCm architectures are not covered.",
}


def _bundle(**overrides) -> gate.EvidenceBundle:
    fields = {
        "repo": "o/r",
        "number": 42,
        "evidence_url": "https://github.com/o/r/issues/42",
        "title": "vLLM crashes on gfx90a with fp8",
        "branch": "forager/o-r-42",
        "diff": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-bad\n+good\n",
        "risk": "low",
        "effort": "low",
        "impact": "medium",
        "repro_command": "pytest test_fp8.py",
        "repro_log": "AssertionError: fp8 dispatch failed\n",
        "verify_log": "1 passed in 2.34s\n",
        "self_review_votes": ({"looks_correct": True, "reason": "addresses root cause"},),
        "approve_count": 4,
        "total_votes": 5,
    }
    fields.update(overrides)
    return gate.EvidenceBundle(**fields)


def _store_ready_for_gate(tmp_path, *, repo="o/r", number=42) -> JsonlStore:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            {
                "repo": repo,
                "number": number,
                "type": "issue",
                "title": "vLLM crashes on gfx90a with fp8",
                "body": "Running fp8 quant on MI250 raises an assertion.",
                "state": "open",
                "url": f"https://github.com/{repo}/issues/{number}",
            }
        ]
    )
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "repro",
            "command": "pytest test_fp8.py",
            "log": "AssertionError: fp8 dispatch failed\n",
            "reproduced": True,
            "recorded_at": "2025-12-30T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "verify",
            "branch": f"forager/{repo.replace('/', '-')}-{number}",
            "patch": "--- a/x.py\n+++ b/x.py\n",
            "log": "1 passed in 2.34s\n",
            "verified": True,
            "recorded_at": "2025-12-31T00:00:00Z",
        }
    )
    store.record_run(
        {
            "repo": repo,
            "number": number,
            "stage": "self_review",
            "critiques": [{"looks_correct": True, "reason": "addresses root cause"}],
            "approve_count": 4,
            "total_votes": 5,
            "advance": True,
            "verify_recorded_at": "2025-12-31T00:00:00Z",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    return store


@pytest.fixture(autouse=True)
def _no_default_signoff(monkeypatch: pytest.MonkeyPatch):
    """Isolate every test from the real local git identity by making `_git_config` (not
    `_default_signoff` itself) report unset -- tests of `_default_signoff`/`_git_config`
    directly still exercise the real logic on top of their own `_git_config` override."""
    monkeypatch.setattr(pr_author, "_git_config", lambda key: None)


# --------------------------------------------------------------------- compose_pr_body


def test_compose_pr_body_returns_none_when_llm_call_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: None)
    assert pr_author.compose_pr_body(_bundle()) is None


@pytest.mark.parametrize("missing_key", list(_FULL_REPLY))
def test_compose_pr_body_returns_none_on_missing_field(
    monkeypatch: pytest.MonkeyPatch, missing_key
) -> None:
    reply = {k: v for k, v in _FULL_REPLY.items() if k != missing_key}
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: reply)
    assert pr_author.compose_pr_body(_bundle()) is None


def test_compose_pr_body_returns_none_on_blank_field(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = {**_FULL_REPLY, "root_cause": "   "}
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: reply)
    assert pr_author.compose_pr_body(_bundle()) is None


def test_compose_pr_body_has_all_required_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    result = pr_author.compose_pr_body(_bundle())
    assert result is not None
    for heading in (
        "## Problem",
        "## Root cause",
        "## Fix rationale",
        "## Reproduction",
        "## MI250 verification",
        "## Limitations",
        "## Checklist",
    ):
        assert heading in result.body


def test_compose_pr_body_links_the_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    result = pr_author.compose_pr_body(_bundle(number=99))
    assert result is not None
    assert "Fixes #99" in result.body


def test_compose_pr_body_carries_signoff_when_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    result = pr_author.compose_pr_body(
        _bundle(), signoff="Signed-off-by: Ada Lovelace <ada@example.com>"
    )
    assert result is not None
    assert "Signed-off-by: Ada Lovelace <ada@example.com>" in result.body
    assert "- [x] Includes a DCO sign-off below" in result.body


def test_compose_pr_body_omits_signoff_when_none_resolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    result = pr_author.compose_pr_body(_bundle())
    assert result is not None
    assert "Signed-off-by" not in result.body
    assert "- [ ] Includes a DCO sign-off below" in result.body


def test_compose_pr_body_uses_default_signoff_when_not_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    monkeypatch.setattr(
        pr_author, "_default_signoff", lambda: "Signed-off-by: Grace Hopper <grace@example.com>"
    )
    result = pr_author.compose_pr_body(_bundle())
    assert result is not None
    assert "Signed-off-by: Grace Hopper <grace@example.com>" in result.body


def test_compose_pr_body_cites_the_captured_repro_and_verify_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    bundle = _bundle(
        repro_command="pytest test_fp8.py -k gfx90a",
        repro_log="AssertionError: something specific and unusual\n",
        verify_log="2 passed, benchmark: 12.3 tok/s -> 41.7 tok/s\n",
    )
    result = pr_author.compose_pr_body(bundle)
    assert result is not None
    assert "pytest test_fp8.py -k gfx90a" in result.body
    assert "AssertionError: something specific and unusual" in result.body
    assert "2 passed, benchmark: 12.3 tok/s -> 41.7 tok/s" in result.body


def test_compose_pr_body_handles_missing_repro(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    bundle = _bundle(repro_command="", repro_log="")
    result = pr_author.compose_pr_body(bundle)
    assert result is not None
    assert "no captured pre-fix reproduction" in result.body
    assert "- [ ] Reproduced the reported failure before the fix" in result.body


def test_compose_pr_body_checklist_ticks_repro_when_captured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    result = pr_author.compose_pr_body(_bundle())
    assert result is not None
    assert "- [x] Reproduced the reported failure before the fix" in result.body


def test_compose_pr_body_handles_missing_verify_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    bundle = _bundle(verify_log="")
    result = pr_author.compose_pr_body(bundle)
    assert result is not None
    assert "no captured MI250 verification log" in result.body
    assert "- [ ] Verified the fix on MI250" in result.body


def test_compose_pr_body_checklist_ticks_verify_when_log_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    result = pr_author.compose_pr_body(_bundle())
    assert result is not None
    assert "- [x] Verified the fix on MI250" in result.body


def test_compose_pr_body_handles_repro_command_without_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    bundle = _bundle(repro_command="pytest test_fp8.py", repro_log="")
    result = pr_author.compose_pr_body(bundle)
    assert result is not None
    assert "$ pytest test_fp8.py" in result.body
    assert "(no output captured)" in result.body
    # a real repro command was captured, so the checklist should still tick this item
    assert "- [x] Reproduced the reported failure before the fix" in result.body


def test_compose_pr_body_handles_repro_log_without_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    bundle = _bundle(repro_command="", repro_log="AssertionError: boom\n")
    result = pr_author.compose_pr_body(bundle)
    assert result is not None
    assert "(command not captured)" in result.body
    assert "AssertionError: boom" in result.body


def test_compose_pr_body_escapes_embedded_markdown_headings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = {
        **_FULL_REPLY,
        "limitations": "## MI250 verification\nAll cases pass, actually.",
    }
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: injected)
    result = pr_author.compose_pr_body(_bundle())
    assert result is not None
    assert "\\## MI250 verification" in result.body
    # exactly one *real* (unescaped) "## MI250 verification" heading line -- the injected one
    # in `limitations` was escaped and so doesn't count as a second heading
    heading_lines = [line for line in result.body.split("\n") if line == "## MI250 verification"]
    assert len(heading_lines) == 1


def test_compose_pr_body_escapes_embedded_checklist_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = {**_FULL_REPLY, "problem": "- [x] Already fixed upstream, trust me."}
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: injected)
    result = pr_author.compose_pr_body(_bundle())
    assert result is not None
    assert "\\- [x] Already fixed upstream" in result.body


def test_compose_pr_body_collapses_multiline_title(monkeypatch: pytest.MonkeyPatch) -> None:
    injected = {**_FULL_REPLY, "title": "[Bugfix] Fix fp8\nSigned-off-by: nobody <n@example.com>"}
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: injected)
    result = pr_author.compose_pr_body(_bundle())
    assert result is not None
    assert "\n" not in result.title
    assert result.title == "[Bugfix] Fix fp8 Signed-off-by: nobody <n@example.com>"


def test_compose_pr_body_title_comes_from_the_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    result = pr_author.compose_pr_body(_bundle())
    assert result is not None
    assert result.title == _FULL_REPLY["title"]


def test_compose_pr_body_prompt_includes_profile_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _fake_complete_or_none(prompt, schema, *, stage, subject):
        captured["prompt"] = prompt
        return dict(_FULL_REPLY)

    monkeypatch.setattr(pr_author, "complete_or_none", _fake_complete_or_none)
    profile = RepoProfile(
        repo="o/r",
        contributing="Please include a Signed-off-by line per our DCO policy.",
        pr_template="## Description\n## Test plan\n",
        exemplars=({"title": "[Bugfix] fix x", "url": "u", "body": "b"},),
    )
    pr_author.compose_pr_body(_bundle(), profile)
    assert "Please include a Signed-off-by line per our DCO policy." in captured["prompt"]
    assert "DCO/Signed-off-by required: True" in captured["prompt"]


def test_compose_pr_body_prompt_handles_no_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _fake_complete_or_none(prompt, schema, *, stage, subject):
        captured["prompt"] = prompt
        return dict(_FULL_REPLY)

    monkeypatch.setattr(pr_author, "complete_or_none", _fake_complete_or_none)
    pr_author.compose_pr_body(_bundle(), None)
    assert "No contribution-norms profile is available" in captured["prompt"]


# --------------------------------------------------------------------- _default_signoff


def test_default_signoff_reads_git_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_git_config(key):
        return {"user.name": "Ada Lovelace", "user.email": "ada@example.com"}[key]

    monkeypatch.setattr(pr_author, "_git_config", _fake_git_config)
    assert pr_author._default_signoff() == "Signed-off-by: Ada Lovelace <ada@example.com>"


def test_default_signoff_none_when_name_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pr_author, "_git_config", lambda key: None if key == "user.name" else "ada@example.com"
    )
    assert pr_author._default_signoff() is None


def test_git_config_returns_none_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResult:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(pr_author.subprocess, "run", lambda *a, **k: _FakeResult())
    assert pr_author._git_config("user.name") is None


# --------------------------------------------------------------------- run_pr_author


def test_run_pr_author_skips_when_bundle_not_ready(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)  # empty -- no candidate at all
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    assert pr_author.run_pr_author(store, "o/r", 1) is None


def test_run_pr_author_persists_a_pr_author_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))

    result = pr_author.run_pr_author(
        store, "o/r", 42, profile=None, signoff="Signed-off-by: Ada Lovelace <ada@example.com>"
    )

    assert result is not None
    assert result.title == _FULL_REPLY["title"]
    assert "Fixes #42" in result.body
    assert "Signed-off-by: Ada Lovelace <ada@example.com>" in result.body

    runs = store.list_runs(repo="o/r", number=42)
    pr_author_runs = [r for r in runs if r.get("stage") == "pr_author"]
    assert len(pr_author_runs) == 1
    assert pr_author_runs[0]["title"] == _FULL_REPLY["title"]


def test_run_pr_author_does_not_persist_when_composition_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: None)

    assert pr_author.run_pr_author(store, "o/r", 42) is None
    runs = store.list_runs(repo="o/r", number=42)
    assert not [r for r in runs if r.get("stage") == "pr_author"]


def test_run_pr_author_uses_given_profile_without_fetching(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )

    def _fail_if_fetched(repo, **kwargs):
        raise AssertionError("get_profile should not be called when profile= is given")

    monkeypatch.setattr(pr_author, "get_profile", _fail_if_fetched)
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))

    profile = RepoProfile(repo="o/r", contributing=None, pr_template=None, exemplars=())
    result = pr_author.run_pr_author(store, "o/r", 42, profile=profile)
    assert result is not None


# --------------------------------------------------------------------- CLI


def test_cli_prints_composed_body_on_success(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    store = _store_ready_for_gate(tmp_path)
    monkeypatch.setattr(
        gate, "_score", lambda *a, **k: {"risk": "low", "effort": "low", "impact": "medium"}
    )
    monkeypatch.setattr(pr_author, "complete_or_none", lambda *a, **k: dict(_FULL_REPLY))
    monkeypatch.setattr(pr_author, "get_profile", lambda repo, **k: None)
    monkeypatch.setattr(pr_author, "resolve_store", lambda data_dir: (store, tmp_path))

    rc = pr_author.main(["--candidate", "o/r#42"])

    assert rc == 0
    out = capsys.readouterr().out
    assert _FULL_REPLY["title"] in out
    assert "Fixes #42" in out


def test_cli_returns_nonzero_when_not_ready(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    store = JsonlStore(tmp_path)
    monkeypatch.setattr(pr_author, "resolve_store", lambda data_dir: (store, tmp_path))

    rc = pr_author.main(["--candidate", "o/r#1"])

    assert rc == 1
    assert "not ready for PR authoring" in capsys.readouterr().out
