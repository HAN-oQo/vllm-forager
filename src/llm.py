"""Pluggable LLM wrapper (T0.7).

Every agent reaches the model through one contract — :func:`complete` — over three
interchangeable backends selected by the env var ``LLM_PROVIDER``:

===============  ==============================================  ===============================
``LLM_PROVIDER``  backend                                         config
===============  ==============================================  ===============================
``claude_cli``    ``claude -p`` (Claude Code headless) — default  inherits local Claude Code auth
``claude_api``    Anthropic Messages API                          ``ANTHROPIC_API_KEY``
``local``         OpenAI-compatible endpoint (vLLM server)        ``LLM_BASE_URL`` + ``LLM_MODEL``
===============  ==============================================  ===============================

Because dispatch is by env, agents never name a provider — swapping claude_cli → API →
local vLLM is a config change, not a code change (CLAUDE.md: "Never hardcode a provider").

Two entry points share one implementation:

- :func:`complete` — the agent-facing contract, ``str | dict``. Pass ``json_schema`` to get
  JSON mode: the wrapper appends a "reply with only JSON matching this schema" directive to
  the system prompt and parses the reply into a ``dict``.
- :func:`complete_detailed` — the same call, but returns an :class:`LLMResult` carrying a
  :class:`CallMeta` (provider, model, token counts, latency, cost). The T2.6 cost-aware
  bandit selects a provider from this metadata, so it is captured on every call.

JSON mode uses instruct-and-parse (not a provider-specific structured-output feature) so it
behaves identically across all three backends. Any failure — unknown provider, missing
credential, transport error, non-zero CLI exit, or non-JSON output in JSON mode — raises
:class:`LLMError`.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # works even if python-dotenv isn't installed
    pass


class LLMError(RuntimeError):
    """Any LLM call failure: bad provider/credential, transport error, or unparseable output."""


@dataclass
class CallMeta:
    """Per-call telemetry, captured on every provider so the T2.6 bandit can score choices.

    ``cost_usd`` is provider-reported where available (claude_cli) and otherwise estimated
    from :data:`_PRICES_PER_MTOK`; it is ``0.0`` for a self-hosted ``local`` endpoint and for
    any model absent from the price table.
    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    cost_usd: float


@dataclass
class LLMResult:
    """A completion plus its :class:`CallMeta`.

    ``content`` is a ``dict`` in JSON mode, otherwise ``str``.
    """

    content: str | dict
    meta: CallMeta


DEFAULT_PROVIDER = "claude_cli"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_TOKENS = 4096

# Approximate list price in USD per 1M tokens, as (input, output). Used only to *estimate*
# claude_api cost for the bandit; edit as pricing changes. Unknown model → cost 0.0 (never
# an error — a wrong estimate must not block a call). claude_cli reports its own exact cost.
_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from the price table; unknown model → 0.0.

    The API echoes a *resolved* model id (e.g. ``claude-sonnet-5-20260514``) that won't match
    the bare-alias table keys, so on an exact miss fall back to the longest table key that is
    a prefix of ``model``. Without this the estimate is silently 0 for every real API call.
    """
    rate = _PRICES_PER_MTOK.get(model)
    if rate is None:
        prefixes = [key for key in _PRICES_PER_MTOK if model.startswith(key)]
        if prefixes:
            rate = _PRICES_PER_MTOK[max(prefixes, key=len)]
    if rate is None:
        return 0.0
    price_in, price_out = rate
    return prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out


def _env_int(name: str, default: int) -> int:
    """Read an int env var; raise :class:`LLMError` (not a bare ValueError) if malformed."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LLMError(f"{name}={raw!r} is not a valid integer") from exc


def _env_float(name: str, default: float) -> float:
    """Read a float env var; raise :class:`LLMError` (not a bare ValueError) if malformed."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMError(f"{name}={raw!r} is not a valid number") from exc


def _model_for(provider: str) -> str | None:
    """Resolve the model id: ``LLM_MODEL`` if set, else a per-provider default.

    Returns None for ``claude_cli`` (let the CLI pick its configured model).
    """
    model = os.getenv("LLM_MODEL")
    if model:
        return model
    if provider == "claude_api":
        return "claude-sonnet-5"
    if provider == "local":
        return "default"  # vLLM usually serves one model; set LLM_MODEL to its served name
    return None  # claude_cli: defer to the CLI's own default


def _json_directive(schema: dict) -> str:
    """Instruction appended to the system prompt to force a schema-conforming JSON reply."""
    return (
        "Respond with ONLY a single JSON object that conforms to the following JSON schema. "
        "Do not include any prose, explanation, or markdown code fences.\n"
        f"JSON schema:\n{json.dumps(schema)}"
    )


def _parse_json_object(text: str) -> dict:
    """Parse a model reply into a dict, tolerating a ```json fenced block. Raise on failure."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line (``` or ```json), then a trailing fence wherever it
        # sits — on its own line OR glued to the JSON (…}```), which a line-based strip misses.
        stripped = stripped.split("\n", 1)[1].strip() if "\n" in stripped else ""
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMError(f"JSON mode: reply was not valid JSON: {text[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"JSON mode: expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    """POST JSON and return the parsed response, mapping any transport/HTTP error to LLMError."""
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:  # covers HTTPError, Timeout, ConnectionError
        raise LLMError(f"HTTP request to {url} failed: {exc}") from exc
    try:
        data = resp.json()
    except ValueError as exc:
        raise LLMError(f"{url} returned a non-JSON body") from exc
    if not isinstance(data, dict):
        raise LLMError(f"{url} returned a non-object JSON body")
    return data


def _run_claude_cli(
    prompt: str, system: str | None, model: str | None, timeout: float
) -> tuple[str, CallMeta]:
    """Run ``claude -p --output-format json`` and extract text + usage/cost from its JSON."""
    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    if system:
        cmd += ["--append-system-prompt", system]
    cmd.append("--")  # end-of-options: a prompt starting with '-' must not be read as a flag
    cmd.append(prompt)

    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise LLMError(
            "`claude` CLI not found on PATH — install Claude Code or pick another provider"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise LLMError(f"claude CLI timed out after {timeout}s") from exc
    latency = time.monotonic() - start

    if proc.returncode != 0:
        raise LLMError(f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:500]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise LLMError(f"claude CLI returned non-JSON stdout: {proc.stdout[:200]!r}") from exc
    if data.get("is_error"):
        raise LLMError(f"claude CLI reported an error: {data.get('result')!r}")

    usage = data.get("usage") or {}
    # `... or 0`, not `.get(k, 0)`: a default only applies to a MISSING key, not a present null.
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    meta = CallMeta(
        provider="claude_cli",
        model=str(data.get("model") or model or ""),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_s=latency,
        cost_usd=float(data.get("total_cost_usd") or 0.0),
    )
    return str(data.get("result", "")), meta


def _run_claude_api(
    prompt: str, system: str | None, model: str | None, timeout: float
) -> tuple[str, CallMeta]:
    """Call the Anthropic Messages API over HTTP (keeps deps to `requests`, no SDK)."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("claude_api requires ANTHROPIC_API_KEY")
    model = model or "claude-sonnet-5"
    base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": _env_int("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    start = time.monotonic()
    data = _post(f"{base}/v1/messages", headers, body, timeout)
    latency = time.monotonic() - start

    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    model_used = str(data.get("model") or model)
    meta = CallMeta(
        provider="claude_api",
        model=model_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_s=latency,
        cost_usd=_estimate_cost(model_used, prompt_tokens, completion_tokens),
    )
    return text, meta


def _run_local(
    prompt: str, system: str | None, model: str | None, timeout: float
) -> tuple[str, CallMeta]:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint (a local vLLM server)."""
    base = os.getenv("LLM_BASE_URL")
    if not base:
        raise LLMError("local provider requires LLM_BASE_URL (OpenAI-compatible endpoint)")
    model = model or "default"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": _env_int("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
    }
    headers = {"content-type": "application/json"}
    key = os.getenv("LLM_API_KEY")  # optional — vLLM can be run with an --api-key
    if key:
        headers["Authorization"] = f"Bearer {key}"

    start = time.monotonic()
    data = _post(f"{base.rstrip('/')}/chat/completions", headers, body, timeout)
    latency = time.monotonic() - start

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"local endpoint returned an unexpected shape: {str(data)[:200]}") from exc
    if text is None:
        # A present-but-null content (e.g. a filtered / tool-call / empty completion) would
        # otherwise become the literal string "None"; surface it as an error instead.
        raise LLMError("local endpoint returned null message content (empty completion)")
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    meta = CallMeta(
        provider="local",
        model=str(data.get("model") or model),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
        latency_s=latency,
        cost_usd=0.0,  # self-hosted: no per-token charge
    )
    return str(text), meta


# Provider dispatch table — the single place that maps LLM_PROVIDER → implementation.
_RUNNERS: dict[str, Callable[[str, str | None, str | None, float], tuple[str, CallMeta]]] = {
    "claude_cli": _run_claude_cli,
    "claude_api": _run_claude_api,
    "local": _run_local,
}


def complete_detailed(
    prompt: str,
    *,
    system: str | None = None,
    json_schema: dict | None = None,
    provider: str | None = None,
    timeout: float | None = None,
) -> LLMResult:
    """Run a completion and return content + :class:`CallMeta`.

    Args:
        prompt: The user prompt.
        system: Optional system prompt.
        json_schema: If given, enable JSON mode — a directive is appended to the system
            prompt and the reply is parsed into a ``dict`` (raises :class:`LLMError` if the
            reply is not a JSON object).
        provider: Override ``LLM_PROVIDER`` for this call (mainly for tests).
        timeout: Per-call timeout in seconds; defaults to ``LLM_TIMEOUT`` or
            :data:`DEFAULT_TIMEOUT_S`.

    Returns:
        An :class:`LLMResult`; ``.content`` is a ``dict`` in JSON mode, otherwise ``str``.

    Raises:
        LLMError: unknown provider, missing credential, transport/CLI failure, or a reply
            that isn't valid JSON in JSON mode.
    """
    # `or ... or DEFAULT_PROVIDER` (ending in a str literal) also narrows the type to str.
    resolved_provider = provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER
    resolved_timeout = (
        float(timeout) if timeout is not None else _env_float("LLM_TIMEOUT", DEFAULT_TIMEOUT_S)
    )
    runner = _RUNNERS.get(resolved_provider)
    if runner is None:
        raise LLMError(
            f"unknown LLM_PROVIDER {resolved_provider!r}; expected one of {sorted(_RUNNERS)}"
        )

    if json_schema is not None:
        directive = _json_directive(json_schema)
        system = f"{system}\n\n{directive}" if system else directive

    text, meta = runner(prompt, system, _model_for(resolved_provider), resolved_timeout)
    content: str | dict = _parse_json_object(text) if json_schema is not None else text
    return LLMResult(content=content, meta=meta)


def complete(
    prompt: str,
    *,
    system: str | None = None,
    json_schema: dict | None = None,
    provider: str | None = None,
    timeout: float | None = None,
) -> str | dict:
    """Provider-agnostic completion. See :func:`complete_detailed` for JSON mode and args.

    Returns the reply only (``str``, or ``dict`` when ``json_schema`` is given). Use
    :func:`complete_detailed` when you also need the per-call metadata (e.g. the T2.6 bandit).
    """
    return complete_detailed(
        prompt, system=system, json_schema=json_schema, provider=provider, timeout=timeout
    ).content
