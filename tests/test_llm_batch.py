"""Tests for T4.11's cheap-classification mechanisms — offline & deterministic.

Per the DEVPLAN todo: a batched classify call returns per-item labels aligned to their inputs
(mocked); the Batches path parses a batch-result fixture; the cached preamble is sent as a
cache breakpoint (all offline/mocked, no live Anthropic call anywhere here).
"""

from __future__ import annotations

import json

import pytest
import requests

from src import llm, taxonomy
from src.agents import analyst
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m4


class _FakeResp:
    """Minimal stand-in for a requests.Response (mirrors tests/test_llm.py's own helper)."""

    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


# --------------------------------------------------------------------- cache_system (claude_api)


def test_cache_system_sends_a_cache_control_breakpoint_on_claude_api(monkeypatch):
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _FakeResp(
            {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "model": "claude-sonnet-5",
            }
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setattr(llm.requests, "post", fake_post)

    llm.complete("hi", system="a big cacheable preamble", provider="claude_api", cache_system=True)

    assert captured["body"]["system"] == [
        {"type": "text", "text": "a big cacheable preamble", "cache_control": {"type": "ephemeral"}}
    ]


def test_cache_system_false_sends_a_plain_string_system(monkeypatch):
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _FakeResp(
            {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setattr(llm.requests, "post", fake_post)

    llm.complete("hi", system="plain", provider="claude_api")  # cache_system defaults False

    assert captured["body"]["system"] == "plain"


def test_cache_system_populates_cache_token_fields_on_a_hit(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *a, **k: _FakeResp(
            {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 400,
                },
                "model": "claude-sonnet-5",
            }
        ),
    )

    result = llm.complete_detailed(
        "hi", system="preamble", provider="claude_api", cache_system=True
    )

    assert result.meta.cache_creation_tokens == 100
    assert result.meta.cache_read_tokens == 400
    # Anthropic bills cache tokens separately from input_tokens -- both total_tokens and
    # cost_usd must include them, not just the plain input/output tokens.
    assert result.meta.total_tokens == 5 + 3 + 100 + 400
    expected_cost = (
        5 / 1e6 * 3.0  # input
        + 3 / 1e6 * 15.0  # output
        + 100 / 1e6 * 3.0 * 1.25  # cache write
        + 400 / 1e6 * 3.0 * 0.1  # cache read
    )
    assert result.meta.cost_usd == pytest.approx(expected_cost)


def test_a_call_with_no_cache_tokens_has_the_same_cost_as_before_this_feature(monkeypatch):
    """Regression guard: a normal (non-cached) call's cost/total_tokens must be unaffected by
    the cache-token accounting added alongside it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *a, **k: _FakeResp(
            {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "model": "claude-sonnet-5",
            }
        ),
    )

    result = llm.complete_detailed("hi", provider="claude_api")

    assert result.meta.total_tokens == 150
    assert result.meta.cost_usd == pytest.approx(llm._estimate_cost("claude-sonnet-5", 100, 50))


def test_cache_system_is_a_no_op_for_claude_cli(monkeypatch):
    """claude_cli has no cache-control equivalent -- cache_system=True must not error or
    change the invoked command."""
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out = json.dumps({"result": "hi", "is_error": False, "usage": {}})

        class _Proc:
            returncode = 0
            stdout = out
            stderr = ""

        return _Proc()

    monkeypatch.setattr(llm.subprocess, "run", fake_run)

    llm.complete("hi", system="preamble", provider="claude_cli", cache_system=True)

    assert "--append-system-prompt" in captured["cmd"]
    assert "preamble" in captured["cmd"]


# --------------------------------------------------------------------- Message Batches


def test_submit_message_batch_posts_one_request_per_item(monkeypatch):
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        return _FakeResp({"id": "batch_123"})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setattr(llm.requests, "post", fake_post)

    batch_id = llm.submit_message_batch(
        [
            llm.BatchRequest(custom_id="a", prompt="classify a"),
            llm.BatchRequest(
                custom_id="b", prompt="classify b", system="sys", json_schema={"type": "object"}
            ),
        ]
    )

    assert batch_id == "batch_123"
    assert captured["url"].endswith("/v1/messages/batches")
    reqs = captured["body"]["requests"]
    assert [r["custom_id"] for r in reqs] == ["a", "b"]
    assert reqs[0]["params"]["messages"][0]["content"] == "classify a"
    assert "JSON schema" in reqs[1]["params"]["system"]  # json_schema directive appended


def test_submit_message_batch_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(llm.LLMError):
        llm.submit_message_batch([llm.BatchRequest(custom_id="a", prompt="x")])


def test_submit_message_batch_ignores_llm_model_env_var(monkeypatch):
    """Regression: submit_message_batch always targets claude_api regardless of LLM_PROVIDER --
    reading the shared LLM_MODEL env var (meant for whatever provider is *currently*
    configured, e.g. a local vLLM deployment's served model name) would submit an
    Anthropic-incompatible model id with no explicit `model=` override."""
    captured: dict = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_MODEL", "my-local-vllm-model")

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _FakeResp({"id": "batch_1"})

    monkeypatch.setattr(llm.requests, "post", fake_post)

    llm.submit_message_batch([llm.BatchRequest(custom_id="a", prompt="x")])

    assert captured["body"]["requests"][0]["params"]["model"] == "claude-sonnet-5"


def test_batch_request_cache_system_sends_a_cache_control_breakpoint(monkeypatch):
    captured: dict = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _FakeResp({"id": "batch_1"})

    monkeypatch.setattr(llm.requests, "post", fake_post)

    llm.submit_message_batch(
        [llm.BatchRequest(custom_id="a", prompt="x", system="preamble", cache_system=True)]
    )

    assert captured["body"]["requests"][0]["params"]["system"] == [
        {"type": "text", "text": "preamble", "cache_control": {"type": "ephemeral"}}
    ]


def _batch_result_line(custom_id: str, *, succeeded: bool, text: str = "", error: str = "") -> str:
    if succeeded:
        result = {
            "type": "succeeded",
            "message": {
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "model": "claude-sonnet-5",
            },
        }
    else:
        result = {"type": "errored", "error": {"message": error}}
    return json.dumps({"custom_id": custom_id, "result": result})


def test_parse_message_batch_results_returns_an_llmresult_per_succeeded_entry():
    jsonl = "\n".join(
        [
            _batch_result_line("a", succeeded=True, text="path a"),
            _batch_result_line("b", succeeded=True, text="path b"),
        ]
    )

    results = llm.parse_message_batch_results(jsonl)

    assert results["a"].content == "path a"
    assert results["b"].content == "path b"
    assert results["a"].meta.provider == "claude_api"
    assert results["a"].meta.prompt_tokens == 10
    # Anthropic bills a batch request at half the synchronous rate.
    assert results["a"].meta.cost_usd == pytest.approx(
        llm._estimate_cost("claude-sonnet-5", 10, 5) * 0.5
    )


def test_parse_message_batch_results_maps_an_errored_entry_to_an_llmerror_value():
    jsonl = _batch_result_line("a", succeeded=False, error="overloaded_error")

    results = llm.parse_message_batch_results(jsonl)

    assert isinstance(results["a"], llm.LLMError)
    assert "overloaded_error" in str(results["a"])


def test_parse_message_batch_results_skips_a_malformed_line_without_aborting_the_rest(capsys):
    jsonl = "\n".join(["not json", _batch_result_line("a", succeeded=True, text="ok")])

    results = llm.parse_message_batch_results(jsonl)

    assert list(results) == ["a"]
    assert "skipping a malformed batch-result line" in capsys.readouterr().err


def test_parse_message_batch_results_empty_input_returns_empty_dict():
    assert llm.parse_message_batch_results("") == {}


def test_parse_message_batch_results_skips_a_line_with_a_non_object_result(capsys):
    """Regression: a "result" value that parses as valid JSON but isn't an object (a string,
    null, etc.) must be skipped, not crash the whole parse via an uncaught AttributeError on
    `.get`."""
    jsonl = "\n".join(
        [
            json.dumps({"custom_id": "bad", "result": "not-an-object"}),
            _batch_result_line("a", succeeded=True, text="ok"),
        ]
    )

    results = llm.parse_message_batch_results(jsonl)  # must not raise

    assert list(results) == ["a"]
    assert "skipping a malformed batch-result line" in capsys.readouterr().err


# --------------------------------------------------------------------- classify_batch


@pytest.fixture
def store(tmp_path) -> JsonlStore:
    return JsonlStore(tmp_path)


def test_classify_batch_returns_per_item_labels_aligned_to_their_inputs(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch
):
    """DEVPLAN's own worked example: a batched classify call returns per-item labels aligned
    to their inputs (mocked)."""
    taxonomy.create_taxonomy(store, [["speech", "post-training"], ["speech", "streaming"]])
    active = taxonomy.get_active(store)

    captured: dict = {}

    def fake_complete(prompt, *, system=None, json_schema=None, cache_system=False, **kw):
        captured["prompt"] = prompt
        captured["system"] = system
        captured["cache_system"] = cache_system
        return {
            "paths": [
                {"index": 0, "path": ["post-training"]},
                {"index": 1, "path": ["streaming"]},
                {"index": 2, "path": []},
            ]
        }

    monkeypatch.setattr(analyst.llm, "complete", fake_complete)

    items = [
        {"repo": "vllm-project/vllm", "number": 1, "title": "PPO issue", "body": "b1"},
        {"repo": "vllm-project/vllm", "number": 2, "title": "stream issue", "body": "b2"},
        {"repo": "vllm-project/vllm", "number": 3, "title": "unrelated", "body": "b3"},
    ]

    results = analyst.classify_batch(
        items, active, domain="speech", root_options=active.children(("speech",))
    )

    assert [r["path"] for r in results] == [
        ["speech", "post-training"],
        ["speech", "streaming"],
        ["speech"],
    ]
    assert [r["number"] for r in results] == [1, 2, 3]  # aligned to input order
    # The shared instructions/hint went into a cached system prompt, not the per-item prompt.
    assert captured["cache_system"] is True
    assert "post-training" in captured["system"]
    assert "PPO issue" in captured["prompt"]
    assert "post-training" not in captured["prompt"]  # not duplicated into the user prompt


def test_classify_batch_uses_a_precomputed_system_prompt_when_given(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch
):
    """analyze_store computes the system prompt once per domain and reuses it across a
    domain's chunks -- classify_batch must send it as-is, not recompute its own."""
    taxonomy.create_taxonomy(store, [["speech", "post-training"]])
    active = taxonomy.get_active(store)
    captured: dict = {}

    def fake_complete(prompt, *, system=None, **kw):
        captured["system"] = system
        return {"paths": [{"index": 0, "path": []}]}

    monkeypatch.setattr(analyst.llm, "complete", fake_complete)

    analyst.classify_batch(
        [{"repo": "vllm-project/vllm", "number": 1, "title": "a", "body": ""}],
        active,
        domain="speech",
        root_options=active.children(("speech",)),
        system_prompt="a precomputed system prompt",
    )

    assert captured["system"] == "a precomputed system prompt"


def test_classify_batch_short_reply_falls_back_to_domain_only_for_missing_entries(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch
):
    taxonomy.create_taxonomy(store, [["speech", "post-training"]])
    active = taxonomy.get_active(store)

    monkeypatch.setattr(
        analyst.llm,
        "complete",
        lambda *a, **k: {"paths": [{"index": 0, "path": ["post-training"]}]},
    )

    items = [
        {"repo": "vllm-project/vllm", "number": 1, "title": "a", "body": ""},
        {"repo": "vllm-project/vllm", "number": 2, "title": "b", "body": ""},
    ]

    results = analyst.classify_batch(
        items, active, domain="speech", root_options=active.children(("speech",))
    )

    assert results[0]["path"] == ["speech", "post-training"]
    assert results[1]["path"] == ["speech"]  # no entry at index 1 -- falls back, not an error


def test_classify_batch_ignores_a_reordered_or_duplicated_reply_via_explicit_index(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch
):
    """Regression: a reply that reorders entries, duplicates an index, or includes an
    out-of-range index must not silently misalign a later item to the wrong classification --
    each entry is correlated by its own `index`, not array position."""
    taxonomy.create_taxonomy(store, [["speech", "post-training"], ["speech", "streaming"]])
    active = taxonomy.get_active(store)
    monkeypatch.setattr(
        analyst.llm,
        "complete",
        lambda *a, **k: {
            "paths": [
                {"index": 1, "path": ["streaming"]},  # reordered: index 1 appears first
                {"index": 0, "path": ["post-training"]},
                {"index": 0, "path": ["streaming"]},  # duplicate index 0 -- first one wins
                {"index": 99, "path": ["post-training"]},  # out of range -- ignored
            ]
        },
    )

    items = [
        {"repo": "vllm-project/vllm", "number": 1, "title": "a", "body": ""},
        {"repo": "vllm-project/vllm", "number": 2, "title": "b", "body": ""},
    ]

    results = analyst.classify_batch(
        items, active, domain="speech", root_options=active.children(("speech",))
    )

    assert results[0]["path"] == ["speech", "post-training"]  # index 0 -- unaffected by reorder
    assert results[1]["path"] == ["speech", "streaming"]  # index 1


def test_classify_batch_raises_llmerror_and_classifies_nothing(
    store: JsonlStore, monkeypatch: pytest.MonkeyPatch
):
    taxonomy.create_taxonomy(store, [["speech", "post-training"]])
    active = taxonomy.get_active(store)

    def fake_complete(*a, **k):
        raise llm.LLMError("boom")

    monkeypatch.setattr(analyst.llm, "complete", fake_complete)

    with pytest.raises(llm.LLMError):
        analyst.classify_batch(
            [{"repo": "vllm-project/vllm", "number": 1, "title": "a", "body": ""}],
            active,
            domain="speech",
            root_options=active.children(("speech",)),
        )
