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

import contextvars
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

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

    ``cache_creation_tokens``/``cache_read_tokens`` (T4.11) are only ever non-zero for
    ``claude_api`` — the only provider this module reports prompt-cache usage for (see
    ``complete``'s ``cache_system`` param); ``claude_cli``/``local`` always report ``0`` here,
    closing T4.8's own disclosed "``CallMeta`` doesn't track prompt-cache tokens" gap for the
    one provider where this wrapper can actually observe it.
    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    cost_usd: float
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class LLMResult:
    """A completion plus its :class:`CallMeta`.

    ``content`` is a ``dict`` in JSON mode, otherwise ``str``.
    """

    content: str | dict
    meta: CallMeta


# T4.8 cost capture: every agent reaches the model through `complete`/`complete_detailed`
# (this module's own docstring), so this is the one place a `CallMeta` can be observed for
# *every* call without threading a store/agent-id through each of the ~10 call sites across
# `src/agents/*.py`. A caller wraps a unit of work in `cost_context(sink)`; `sink` receives
# every `CallMeta` produced by a `complete`/`complete_detailed` call made anywhere in that
# dynamic scope (including nested calls inside a helper this module doesn't know about).
# `ContextVar` (not a plain module global) so nested/concurrent `cost_context` scopes in
# different tasks/threads don't clobber each other's sink.
_cost_sink: contextvars.ContextVar[Callable[[CallMeta], None] | None] = contextvars.ContextVar(
    "_cost_sink", default=None
)


@contextmanager
def cost_context(sink: Callable[[CallMeta], None]):
    """Install `sink` to receive every `CallMeta` from a `complete`/`complete_detailed` call
    made within this ``with`` block, then restore whatever sink (if any) was active before.

    `sink` must not raise — it runs synchronously right after a successful call, and an
    exception here would surface as if the LLM call itself had failed. `src.cost.record_cost`
    (the real T4.8 sink) is itself best-effort for this reason.
    """
    token = _cost_sink.set(sink)
    try:
        yield
    finally:
        _cost_sink.reset(token)


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

# Anthropic prices a prompt-cache write/read as a multiplier of the model's own base *input*
# rate, for the default 5-minute "ephemeral" cache (this module never sets an extended-TTL
# beta header — see `_run_claude_api`'s `cache_system` handling). These ratios (a write costs
# more than a plain input token; a read costs much less) have been stable since prompt caching's
# public launch, but — like `_PRICES_PER_MTOK` itself — are an editable estimate, not a live
# lookup; verify against Anthropic's current pricing page if this drifts. Getting this wrong
# only skews the *estimate* `_estimate_cost` produces for the bandit/cost rows — claude_cli
# still reports its own exact provider-billed cost regardless.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.1


def _estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate USD cost from the price table; unknown model → 0.0.

    The API echoes a *resolved* model id (e.g. ``claude-sonnet-5-20260514``) that won't match
    the bare-alias table keys, so on an exact miss fall back to the longest table key that is
    a prefix of ``model``. Without this the estimate is silently 0 for every real API call.

    ``cache_creation_tokens``/``cache_read_tokens`` (T4.11) are priced separately from
    ``prompt_tokens`` — Anthropic's ``usage.input_tokens`` explicitly excludes them, so omitting
    them here would silently drop the exact tokens ``cache_system=True`` exists to spend on
    (see :data:`_CACHE_WRITE_MULTIPLIER`/:data:`_CACHE_READ_MULTIPLIER`).
    """
    rate = _PRICES_PER_MTOK.get(model)
    if rate is None:
        prefixes = [key for key in _PRICES_PER_MTOK if model.startswith(key)]
        if prefixes:
            rate = _PRICES_PER_MTOK[max(prefixes, key=len)]
    if rate is None:
        return 0.0
    price_in, price_out = rate
    return (
        prompt_tokens / 1e6 * price_in
        + completion_tokens / 1e6 * price_out
        + cache_creation_tokens / 1e6 * price_in * _CACHE_WRITE_MULTIPLIER
        + cache_read_tokens / 1e6 * price_in * _CACHE_READ_MULTIPLIER
    )


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


def _merge_json_directive(system: str | None, json_schema: dict | None) -> str | None:
    """`system` with :func:`_json_directive`'s instruction appended, if `json_schema` is given
    — the exact "fold a schema directive into the system prompt" rule both
    :func:`complete_detailed` (the synchronous path) and :func:`_batch_request_params` (the
    Message Batches path, T4.11) need identically, factored out so the two can't drift."""
    if json_schema is None:
        return system
    directive = _json_directive(json_schema)
    return f"{system}\n\n{directive}" if system else directive


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


def _parse_claude_message(
    message: dict, *, provider: str, latency_s: float = 0.0, cost_multiplier: float = 1.0
) -> tuple[str, CallMeta]:
    """Extract reply text + :class:`CallMeta` from one Anthropic Messages-API-shaped
    ``message`` object — the shape a synchronous ``/v1/messages`` response body *is*, and a
    completed Message Batch entry's own ``message`` field also is (see
    :func:`_llm_result_from_batch_message`). Shared by both so a fix to content/usage
    extraction (a new content-block type, a renamed usage field, ...) only has one place to
    land instead of drifting between the synchronous and batch paths.

    ``cost_multiplier`` scales the estimate (:func:`_estimate_cost`) for a caller that already
    knows its own rate differs from the synchronous price — :data:`_BATCH_DISCOUNT` for a
    completed batch entry, ``1.0`` (no change) for a normal call.
    """
    blocks = message.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    usage = message.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    cache_creation_tokens = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
    model = str(message.get("model") or "")
    meta = CallMeta(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        # Cache tokens are billed separately from `input_tokens` (Anthropic's own semantics
        # explicitly exclude them) -- omitting them here would undercount real spend/usage for
        # every `cache_system=True` call, which is the whole point of that flag existing.
        total_tokens=prompt_tokens + completion_tokens + cache_creation_tokens + cache_read_tokens,
        latency_s=latency_s,
        cost_usd=(
            _estimate_cost(
                model,
                prompt_tokens,
                completion_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
            )
            * cost_multiplier
        ),
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )
    return text, meta


def _run_claude_cli(
    prompt: str, system: str | None, model: str | None, timeout: float, *, cache_system: bool
) -> tuple[str, CallMeta]:
    """Run ``claude -p --output-format json`` and extract text + usage/cost from its JSON.

    ``cache_system`` is accepted (for a uniform :data:`_RUNNERS` signature) but has no effect
    here — the CLI's ``--append-system-prompt`` flag has no cache-control equivalent this
    wrapper can drive; see ``complete``'s own docstring.
    """
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
    prompt: str, system: str | None, model: str | None, timeout: float, *, cache_system: bool
) -> tuple[str, CallMeta]:
    """Call the Anthropic Messages API over HTTP (keeps deps to `requests`, no SDK).

    ``cache_system=True`` sends `system` as a cache-control content block
    (``[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]``) instead of
    a plain string — the only one of this module's three providers where an explicit prompt-
    cache breakpoint is meaningful (``claude_cli`` has no such flag; a self-hosted ``local``
    vLLM endpoint already does automatic prefix caching at the KV-cache level with no API call
    needed). The API's own ``usage.cache_creation_input_tokens``/``cache_read_input_tokens``
    (present whenever a cache breakpoint was actually hit, regardless of this flag) populate
    :class:`CallMeta`'s matching fields.
    """
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
        body["system"] = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if cache_system
            else system
        )

    start = time.monotonic()
    data = _post(f"{base}/v1/messages", headers, body, timeout)
    latency = time.monotonic() - start

    # The API may not echo `model` (observed to always do so today, but not contractually
    # guaranteed) -- `data` alone wouldn't know the originally-requested `model` to fall back
    # to, so patch it in before handing off to the shared parser.
    if not data.get("model"):
        data = {**data, "model": model}
    return _parse_claude_message(data, provider="claude_api", latency_s=latency)


def _run_local(
    prompt: str, system: str | None, model: str | None, timeout: float, *, cache_system: bool
) -> tuple[str, CallMeta]:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint (a local vLLM server).

    ``cache_system`` is accepted (for a uniform :data:`_RUNNERS` signature) but has no effect
    here — vLLM already does automatic prefix caching with no explicit API opt-in; see
    ``complete``'s own docstring.
    """
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


class _Runner(Protocol):
    """The exact call signature every entry in :data:`_RUNNERS` must match — a `Protocol`
    (not `Callable[..., tuple[str, CallMeta]]`) so mypy still catches a future runner with a
    wrong/missing/reordered parameter (including the keyword-only `cache_system`, T4.11) at
    review time, the same precision the dispatch table had before `cache_system` was added.
    """

    def __call__(
        self,
        prompt: str,
        system: str | None,
        model: str | None,
        timeout: float,
        *,
        cache_system: bool,
    ) -> tuple[str, CallMeta]: ...


# Provider dispatch table — the single place that maps LLM_PROVIDER → implementation.
_RUNNERS: dict[str, _Runner] = {
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
    cache_system: bool = False,
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
        cache_system: T4.11 — send `system` as an Anthropic prompt-cache breakpoint. Only
            ``claude_api`` honors this (see :func:`_run_claude_api`); a no-op for
            ``claude_cli``/``local``, which have no equivalent this wrapper can drive. Use for
            a large, stable system prompt reused across many calls (e.g. a taxonomy preamble
            classified once per batch) — a cache hit on a later call is billed at a fraction of
            the input-token rate, visible via :class:`CallMeta`'s ``cache_read_tokens``.

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

    system = _merge_json_directive(system, json_schema)

    text, meta = runner(
        prompt, system, _model_for(resolved_provider), resolved_timeout, cache_system=cache_system
    )

    # `meta` reflects real, already-billed spend the instant the provider call returns — record
    # it here, before JSON-mode parsing, so a malformed reply (a realistic failure every real
    # agent already handles per-item, e.g. src/agents/analyst.py) still gets its cost captured.
    # Parsing after this point can raise LLMError without losing that already-recorded spend.
    sink = _cost_sink.get()
    if sink is not None:
        sink(meta)

    content: str | dict = _parse_json_object(text) if json_schema is not None else text
    return LLMResult(content=content, meta=meta)


def complete(
    prompt: str,
    *,
    system: str | None = None,
    json_schema: dict | None = None,
    provider: str | None = None,
    timeout: float | None = None,
    cache_system: bool = False,
) -> str | dict:
    """Provider-agnostic completion. See :func:`complete_detailed` for JSON mode and args.

    Returns the reply only (``str``, or ``dict`` when ``json_schema`` is given). Use
    :func:`complete_detailed` when you also need the per-call metadata (e.g. the T2.6 bandit).
    """
    return complete_detailed(
        prompt,
        system=system,
        json_schema=json_schema,
        provider=provider,
        timeout=timeout,
        cache_system=cache_system,
    ).content


# --------------------------------------------------------------------- Message Batches (T4.11)
#
# Anthropic's Batches API (https://docs.anthropic.com/en/api/creating-message-batches) has no
# equivalent for claude_cli or a self-hosted `local` vLLM endpoint, so — unlike `complete`/
# `complete_detailed` — the functions below always talk to the Anthropic API directly and
# ignore `LLM_PROVIDER`. Only submission (`submit_message_batch`) and result-parsing
# (`parse_message_batch_results`) are implemented: a submitted batch is processed
# **asynchronously** (Anthropic's own docs describe results landing anywhere from minutes to
# 24h later, at roughly half the synchronous per-token price — the ~50% saving T4.11's DEVPLAN
# entry names), which is why it targets classification specifically (latency-insensitive at its
# weekly orchestrator cadence). Polling for a batch's completion and downloading its results
# file from the URL Anthropic returns once it's done are NOT implemented here — no caller in
# this codebase submits a real batch yet, so there's nothing to poll for; a future integration
# (wiring this into `analyze_store`/the orchestrator's intel tick, replacing `classify_batch`'s
# synchronous calls) needs both before this path is usable for real, not just testable offline.

_BATCH_DISCOUNT = 0.5  # Anthropic bills a completed batch request at half the synchronous rate.


@dataclass
class BatchRequest:
    """One request within an Anthropic Message Batch (:func:`submit_message_batch`).

    `custom_id` must be unique within the batch — it's what
    :func:`parse_message_batch_results` correlates a result back to the request that produced
    it (Anthropic doesn't guarantee results are returned in submission order).

    `cache_system` mirrors ``complete``'s own param — see :func:`_run_claude_api`'s docstring;
    it's meaningful here too (a batch of otherwise-independent requests sharing one large,
    stable `system` preamble is exactly the caching pattern `classify_batch` uses).
    """

    custom_id: str
    prompt: str
    system: str | None = None
    json_schema: dict | None = None
    cache_system: bool = False


def _batch_request_params(req: BatchRequest, model: str) -> dict[str, Any]:
    system = _merge_json_directive(req.system, req.json_schema)
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": _env_int("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        "messages": [{"role": "user", "content": req.prompt}],
    }
    if system:
        params["system"] = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if req.cache_system
            else system
        )
    return params


def submit_message_batch(
    requests: list[BatchRequest], *, model: str | None = None, timeout: float | None = None
) -> str:
    """Submit `requests` as one Anthropic Message Batch (``POST /v1/messages/batches``) and
    return the batch's id. See this section's own module-level comment for what's not yet
    implemented (polling/fetching results). Always talks to the Anthropic API directly,
    regardless of ``LLM_PROVIDER`` — there is no ``claude_cli``/``local`` equivalent to dispatch
    to (see this section's own module-level comment).

    Raises:
        LLMError: missing ``ANTHROPIC_API_KEY`` or a transport/HTTP failure.
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("submit_message_batch requires ANTHROPIC_API_KEY (claude_api only)")
    # NOT `_model_for("claude_api")`: that reads the shared `LLM_MODEL` env var, which names
    # whatever provider `LLM_PROVIDER` is *currently* configured to (e.g. a `local` vLLM
    # deployment's served model name) -- meaningless, or actively wrong, for a call that always
    # targets Anthropic regardless of the active provider.
    resolved_model = model or "claude-sonnet-5"
    base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "requests": [
            {"custom_id": req.custom_id, "params": _batch_request_params(req, resolved_model)}
            for req in requests
        ]
    }
    resolved_timeout = (
        float(timeout) if timeout is not None else _env_float("LLM_TIMEOUT", DEFAULT_TIMEOUT_S)
    )
    data = _post(f"{base}/v1/messages/batches", headers, body, resolved_timeout)
    return str(data["id"])


def _llm_result_from_batch_message(message: dict) -> LLMResult:
    """A completed batch entry's own ``message`` object, shaped like a normal (non-batch)
    Messages API response — actually reuses :func:`_parse_claude_message`, the same
    content/usage extraction :func:`_run_claude_api` uses for a synchronous call, priced at
    :data:`_BATCH_DISCOUNT` of the synchronous estimate."""
    text, meta = _parse_claude_message(
        message, provider="claude_api", cost_multiplier=_BATCH_DISCOUNT
    )
    return LLMResult(content=text, meta=meta)


def parse_message_batch_results(results_jsonl: str) -> dict[str, LLMResult | LLMError]:
    """Parse the ``.jsonl`` body of a completed Anthropic Message Batch's results file — one
    JSON object per line, each shaped
    ``{"custom_id": ..., "result": {"type": "succeeded"|"errored", "message": {...}}}`` — into
    a ``custom_id -> LLMResult`` map. An errored entry maps to an :class:`LLMError` **value**
    (not raised) so one bad request in a batch doesn't lose every other entry's result — the
    same per-item degradation this codebase already applies elsewhere (e.g.
    ``src.agents.analyst``'s per-item classification).

    Pure parsing only — this module doesn't fetch the results file itself; a caller downloads
    it from the batch's own ``results_url`` (once polling/fetching is implemented — see this
    section's own module-level comment) and passes the raw text here. A line that isn't valid
    JSON, or doesn't match the expected shape, is skipped with a message to stderr rather than
    aborting every other line's result.

    A succeeded entry's ``.content`` is always the raw reply text, never auto-parsed into a
    dict — unlike ``complete``'s own JSON mode, this function has no per-entry record of
    whether that request used ``json_schema`` (the results file doesn't echo the request), so a
    caller that submitted JSON-mode requests must ``json.loads`` (or reuse
    :func:`_parse_json_object`'s fence-tolerant parsing) the returned entries itself.
    """
    results: dict[str, LLMResult | LLMError] = {}
    for line in results_jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            custom_id = entry["custom_id"]
            result = entry["result"]
            if not isinstance(result, dict):
                raise TypeError(f"'result' must be an object, got {type(result).__name__}")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"llm: skipping a malformed batch-result line: {exc}", file=sys.stderr)
            continue
        if result.get("type") == "succeeded":
            results[custom_id] = _llm_result_from_batch_message(result.get("message") or {})
        else:
            error = (result.get("error") or {}).get("message") or result.get("type")
            results[custom_id] = LLMError(f"batch request {custom_id!r} failed: {error}")
    return results
