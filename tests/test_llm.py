"""Tests for the pluggable LLM wrapper (T0.7) — offline & deterministic.

Each provider is mocked at its transport boundary (``subprocess.run`` for claude_cli,
``requests.post`` for claude_api / local) so we can assert, per the DEVPLAN todo, that:
the prompt/system reach the backend, the reply is parsed, ``json_schema`` yields a dict,
:class:`~src.llm.CallMeta` is populated, and every error path raises :class:`~src.llm.LLMError`.

A live claude_cli smoke test is ``@pytest.mark.integration`` (skipped in the default run).
"""

import json
import subprocess

import pytest
import requests

from src import llm

pytestmark = pytest.mark.m0


class _FakeResp:
    """Minimal stand-in for a requests.Response (json / raise_for_status / status_code)."""

    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


# --------------------------------------------------------------------- claude_cli


def test_claude_cli_parses_result_and_metadata(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out = json.dumps(
            {
                "result": "hello",
                "is_error": False,
                "model": "claude-x",
                "usage": {"input_tokens": 11, "output_tokens": 7},
                "total_cost_usd": 0.002,
            }
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    res = llm.complete_detailed("hi there", system="be terse", provider="claude_cli")

    assert res.content == "hello"
    assert "hi there" in captured["cmd"]  # prompt passed through as an argv element
    assert "--append-system-prompt" in captured["cmd"]  # system prompt forwarded
    assert res.meta.provider == "claude_cli"
    assert res.meta.model == "claude-x"
    assert (res.meta.prompt_tokens, res.meta.completion_tokens, res.meta.total_tokens) == (
        11,
        7,
        18,
    )
    assert res.meta.cost_usd == 0.002
    assert res.meta.latency_s >= 0.0


def test_claude_cli_passes_end_of_options_separator(monkeypatch):
    # A prompt starting with '-' must reach the CLI as a prompt, not be parsed as a flag:
    # the '--' separator must come immediately before the prompt.
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": "y"}), stderr="")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    llm.complete("--verbose please", provider="claude_cli")
    cmd = captured["cmd"]
    assert cmd[-2:] == ["--", "--verbose please"]


def test_claude_cli_null_usage_fields_do_not_crash(monkeypatch):
    # Present-but-null token/cost fields must coerce to 0, not raise TypeError.
    def fake_run(cmd, **kwargs):
        out = json.dumps(
            {
                "result": "ok",
                "usage": {"input_tokens": None, "output_tokens": None},
                "total_cost_usd": None,
            }
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    res = llm.complete_detailed("x", provider="claude_cli")
    assert res.meta.total_tokens == 0
    assert res.meta.cost_usd == 0.0


def test_claude_cli_nonzero_exit_raises(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    with pytest.raises(llm.LLMError):
        llm.complete("x", provider="claude_cli")


def test_default_provider_is_claude_cli(monkeypatch):
    # With no LLM_PROVIDER set, dispatch must default to claude_cli.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    def fake_run(cmd, **kwargs):
        assert cmd[:2] == ["claude", "-p"]
        out = json.dumps({"result": "ok", "usage": {}})
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    assert llm.complete("hi") == "ok"


# --------------------------------------------------------------------- claude_api


def test_claude_api_builds_request_and_parses(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json)
        return _FakeResp(
            {
                "content": [{"type": "text", "text": "world"}],
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "model": "claude-sonnet-5",
            }
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)
    res = llm.complete_detailed("say hi", system="be terse", provider="claude_api")

    assert res.content == "world"
    assert captured["url"].endswith("/v1/messages")
    assert captured["headers"]["x-api-key"] == "sk-test"
    assert captured["body"]["messages"][0]["content"] == "say hi"
    assert captured["body"]["system"] == "be terse"
    assert res.meta.provider == "claude_api"
    assert res.meta.total_tokens == 8
    assert res.meta.cost_usd > 0.0  # sonnet-5 is in the price table


def test_claude_api_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.LLMError):
        llm.complete("x", provider="claude_api")


def test_claude_api_cost_matches_resolved_dated_model(monkeypatch):
    # The API echoes a dated id (claude-sonnet-5-20260514); cost must still resolve via the
    # prefix fallback rather than silently reporting 0.0.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp(
            {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1000, "output_tokens": 1000},
                "model": "claude-sonnet-5-20260514",
            }
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)
    res = llm.complete_detailed("hi", provider="claude_api")
    assert res.meta.model == "claude-sonnet-5-20260514"
    assert res.meta.cost_usd > 0.0  # prefix-matched "claude-sonnet-5" in the price table


# --------------------------------------------------------------------- local (vLLM)


def test_local_openai_shape(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "my-model")
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, body=json)
        return _FakeResp(
            {
                "choices": [{"message": {"content": "local-out"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
                "model": "my-model",
            }
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)
    res = llm.complete_detailed("hello", provider="local")

    assert res.content == "local-out"
    assert captured["url"] == "http://vllm:8000/v1/chat/completions"
    assert captured["body"]["model"] == "my-model"
    assert res.meta.completion_tokens == 6
    assert res.meta.total_tokens == 10
    assert res.meta.cost_usd == 0.0  # self-hosted


def test_local_requires_base_url(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(llm.LLMError):
        llm.complete("x", provider="local")


def test_local_null_content_raises(monkeypatch):
    # content: null (filtered / tool-call / empty) must raise, not return the string "None".
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp({"choices": [{"message": {"content": None}}], "usage": {}})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(llm.LLMError):
        llm.complete("x", provider="local")


def test_http_error_raises(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp({}, status=500)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(llm.LLMError):
        llm.complete("x", provider="local")


# --------------------------------------------------------------------- JSON mode


def test_json_schema_returns_dict_and_injects_directive(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    schema = {"type": "object", "properties": {"category": {"type": "string"}}}

    def fake_post(url, headers=None, json=None, timeout=None):
        # the schema directive must reach the model as a system message
        system_msgs = [m for m in json["messages"] if m["role"] == "system"]
        assert system_msgs and "JSON schema" in system_msgs[0]["content"]
        # reply wrapped in a ```json fence to exercise fence stripping
        return _FakeResp(
            {
                "choices": [{"message": {"content": '```json\n{"category": "rocm-build"}\n```'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = llm.complete("classify this", json_schema=schema, provider="local")
    assert out == {"category": "rocm-build"}


def test_json_mode_strips_glued_closing_fence(monkeypatch):
    # Closing fence on the SAME line as the JSON must still be stripped and parsed.
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp(
            {
                "choices": [{"message": {"content": '```json\n{"category": "rocm-build"}```'}}],
                "usage": {},
            }
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = llm.complete("x", json_schema={"type": "object"}, provider="local")
    assert out == {"category": "rocm-build"}


def test_json_mode_non_json_reply_raises(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp(
            {
                "choices": [{"message": {"content": "definitely not json"}}],
                "usage": {},
            }
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(llm.LLMError):
        llm.complete("x", json_schema={"type": "object"}, provider="local")


# --------------------------------------------------------------------- dispatch


def test_unknown_provider_raises():
    with pytest.raises(llm.LLMError):
        llm.complete("x", provider="does-not-exist")


def test_malformed_timeout_env_raises_llmerror(monkeypatch):
    # A human-friendly-but-invalid value must surface as LLMError, not a bare ValueError.
    monkeypatch.setenv("LLM_TIMEOUT", "30s")
    with pytest.raises(llm.LLMError):
        llm.complete("x", provider="claude_cli")


def test_malformed_max_tokens_env_raises_llmerror(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4k")
    with pytest.raises(llm.LLMError):
        llm.complete("x", provider="local")


# --------------------------------------------------------------------- live smoke


@pytest.mark.integration
def test_claude_cli_live_smoke():
    # Real `claude -p` round-trip (needs local Claude Code auth); skipped by default.
    out = llm.complete("Reply with exactly one word: pong")
    assert isinstance(out, str) and out.strip()
