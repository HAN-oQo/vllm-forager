# Research: per-agent LLM cost tracking for operations

> **Question:** how should we track LLM **cost per agent** once the pipeline is running, so operating
> the multi-agent system stays observable and doesn't overspend? Prefer open-source / self-hostable.
> **Status:** research + recommendation (not yet a DEVPLAN todo — graduate when we act on it).
> **Date:** 2026-07-06. Landscape moves fast; re-verify vendor/OSS status before adopting.

## TL;DR — recommendation for *this* repo

We already have most of what we need: `src/llm.py::complete()` returns per-call `{tokens, latency, cost, model,
provider}` (T0.7), the T2.6 bandit already consumes it, and we run Firestore. So:

1. **Now — DIY on the metadata we already emit + `ccusage` for the CLI sessions.** No new infra, fits our
   minimal-dependency ethos, and attributes cost **per agent for free** (the wrapper knows its caller):
   - Persist one cost record per call to Firestore (`costs`, or fold into T4.4 run events), tagged
     `{agent, run_id, loop, milestone, model, provider, tokens, cost, ts}`.
   - **Don't hand-maintain prices** — borrow a maintained price table (LiteLLM's
     `model_prices_and_context_window.json`, or ccusage's data) and set explicit prices for the local vLLM models.
   - Aggregate in a dashboard **Cost tab** (M5): by agent / day / model / provider.
   - The `claude -p` **dev-loop / collect-loop** sessions bypass the wrapper → ingest **`ccusage --json`** (MIT,
     zero infra) or Claude Code's native OTel metric `claude_code.cost.usage` into the *same* Firestore schema.
2. **Graduate to the [LiteLLM](https://github.com/BerriAI/litellm) proxy** when we need **hard budget caps** and a
   single costed endpoint across Anthropic API + local vLLM + the Claude Code CLI — especially before the
   **agent-teams** scale-up in `docs/IDEAS.md`. It gives per-key / per-team / **per-tag (= per-agent)** spend and
   budgets, MIT core, and unifies all our backends (point `claude` at it via `ANTHROPIC_BASE_URL`). Pin a clean
   version (see caveat).
3. **Optional richer UI:** self-hosted **[Langfuse](https://langfuse.com/)** if we later want nested
   agent-trace + eval + cost in one pane (and to host the RAG-trust / prediction-grading views too). Heavier
   footprint; it's the only tool here with an *official Claude Code plugin*.
4. **Avoid for now:** Helicone (maintenance mode post-acquisition), Lunary (OSS repo deleted), Portkey (the
   analytics/budgets you'd want are hosted-tier; PANW acquisition), Arize Phoenix (ELv2 license; cost is a
   relative weakness for our use).

**One-liner:** *start DIY (we're 80% there) + ccusage; adopt LiteLLM as the gateway when budgets/scale demand it;
Langfuse only if we want a full trace/eval UI.*

## Facts to anchor on (2026)

- **The OpenTelemetry GenAI standard tracks tokens, not dollars.** `gen_ai.*` conventions standardize
  input/output tokens, model, operation, provider — but define **no cost attribute**. Every "$" in every tool
  below is derived = `tokens × a pricing table the tool maintains`. The spec is still experimental in 2026.
  → whatever we pick, **cost = tokens × prices we keep current**; the only question is who owns that table.
- **Claude Code CLI already has native cost telemetry** (for our operator sessions), two ways: local JSONL in
  `~/.claude/projects/` (what `ccusage` reads), and built-in **OTel export**
  (`CLAUDE_CODE_ENABLE_TELEMETRY=1`) emitting `claude_code.cost.usage` (USD; attrs `model`, `query_source`,
  `agent.name`, …) + a per-call `claude_code.api_request` event with `cost_usd`.
- **2026 ownership churn — verify before committing:** Langfuse → acquired by ClickHouse (Jan 2026, stays MIT,
  active); **Helicone → Mintlify, now maintenance-mode**; Portkey → Palo Alto Networks (roadmap uncertain);
  **Lunary OSS repo deleted (~Dec 2025, 404)**.

## What we specifically need

- **Attribution to an arbitrary label** = the agent name (`analyst`/`scout`/`engineer`/…), plus `run_id` / loop.
- **Providers:** Anthropic API **and** `claude -p` CLI **and** a local **vLLM OpenAI-compatible** server (so:
  custom/local model pricing must be settable).
- **Self-hostable**, ideally without adding a heavy datastore (we already run Firestore; adding
  ClickHouse+Redis+S3 is a real cost).
- **Budget/alert** to protect unattended operation (nice-to-have now, important once agents run as teams).
- An **aggregate query API** so our own M5 dashboard can render a Cost tab.

## The DIY baseline (what we already have + the one gap)

Our wrapper emits `{input/output/cache tokens, latency, computed_cost, model, provider, ts}`. A DIY tracker is
small: **write one Firestore doc per call tagged `agent`+`run_id`, aggregate by day / model / agent / loop.**
This cleanly covers the **API and local-vLLM legs** (both go through the wrapper). The **only gap is the
`claude -p` CLI** sessions (they bypass the wrapper) — closed cheaply with `ccusage --json` or Claude Code's OTel
`claude_code.api_request` event fed into the same schema.

**What a dedicated tool buys over DIY:** nested trace/span **call trees** (flat rows can't reconstruct an agent
loop); a **maintained pricing table**; prebuilt dashboards + **hard budget enforcement** (LiteLLM stops runaway
spend *before* the invoice); OTel standardization; correct async ingestion (no added request latency).
**What DIY keeps:** no new infra/lock-in, full schema control, data already in our store, no per-event metering
fees (span-heavy multi-agent runs get expensive on metered clouds).

## Comparison (OSS, self-hostable)

| Tool | License | Self-host weight | Per-call $ (custom/vLLM) | Attribution | Integration | Budgets/alerts | Stars | 2026 status |
|---|---|---|---|---|---|---|---|---|
| **LiteLLM** (gateway) | MIT (+ent) | Medium: images/Helm + **Postgres** | Yes (`register_model`/config) | **virtual keys / teams / tags** | **proxy** + callbacks + OTel | **Yes** (soft budgets, Slack/webhook; hard caps ent) | ~52k | Active · Claude Code documented |
| **Langfuse** | MIT core (+EE) | **Heavy**: PG+ClickHouse+Redis+S3 | Yes (UI/API regex, tiered) | user/session/**tags**/metadata | SDK + OTLP + OpenInference | ⚠ no native alert | ~30k | Active (ClickHouse) · **official Claude Code plugin** |
| **OpenLIT** | Apache-2.0 | Medium: OTel+ClickHouse+UI | **Yes** (built-in + custom JSON) | OTel attrs / baggage | `openlit.init()` (incl. **vLLM**) → OTLP | ⚠ none (route to Grafana) | ~2.6k | Active · **GPU/energy** (AMD via sysfs collector; ⚠ MI250 unverified) |
| **Arize Phoenix** | **ELv2** (source-available) | Light (→PG prod) | Yes (regex; ⚠ not retroactive) | OpenInference span attrs + sessions | OpenInference → OTLP | ⚠ no OSS alert | ~10k | Very active |
| **OpenLLMetry/Traceloop** | Apache-2.0 | SDK only (no store) | ⚠ **tokens only** in OSS | association_properties | auto-instr → OTLP | via backend | ~7k | Active |
| **Helicone** | Apache-2.0 (gw GPL-3) | Light: compose | Yes; ⚠ vLLM steps unconfirmed | headers (Property/User/Session) | proxy (+ async SDK, OTel) | **Yes incl. cost** | ~6k | ⚠ **maintenance mode** |
| **Portkey** | MIT (gateway only) | Light gw; **analytics hosted** | Yes; custom pricing **hosted** | virtual keys + metadata | proxy (+ OTel) | **hosted-tier** | ~12k | ⚠ **PANW acquisition** |
| **Lunary** | Apache-2.0 (unverif.) | ⚠ compose **EE-only** | Yes (`/v1/models`) | user/tags/**agent decorator** | SDK (⚠ no OTLP) | ⚠ none | ~1.4k | ⚠ **OSS repo 404 (deleted)** |
| **ccusage** | MIT | None (local CLI) | Yes (LiteLLM prices, cache tokens) | day/session/project/model | `npx`, `--json` | no | ~17k | Active · **Claude Code-native** |
| **OTel GenAI semconv** | (standard) | n/a | ⚠ **tokens only, no $** | span attributes | instrument once → any backend | n/a | n/a | Experimental |

Best-fit in one line each: **LiteLLM** = gateway budgets across mixed backends · **Langfuse** = self-hosted
trace/eval/cost hub (+Claude Code plugin) · **OpenLIT** = OTel-native cost + the only GPU/energy angle (MI250, if
its sysfs collector validates on gfx90a) · **Phoenix** = OpenInference tracing/eval (ELv2) · **OpenLLMetry** =
pure OTel exporter (DIY $) · **Helicone** = light proxy w/ cost alerts (maintenance risk) · **Portkey** = MIT
routing gw (analytics hosted) · **Lunary** = great attribution ergonomics but OSS gone · **ccusage** = Claude
Code CLI cost, zero infra · **OTel semconv** = vendor-neutral wire format (tokens, never dollars).

## How this could graduate into the plan (if/when we act)

- **M4 (orchestration):** `T4.x` — persist a per-call cost record (agent/run/model/provider/tokens/cost) from
  `src/llm.py` to the KB; a scheduled roll-up (by day/agent/model). *Test:* mocked calls → correct per-agent sums.
- **M4:** `T4.x` — ingest `ccusage --json` (or Claude Code OTel) for the dev-loop/collect-loop sessions into the
  same schema. *Test:* a fixture JSONL → session cost rows.
- **M5 (dashboard):** `T5.x` — a **Cost tab** (by agent / day / model / provider; a monthly total + a simple
  budget line). Fits the operator-console tabs.
- **Later (agent-teams, IDEAS):** stand up **LiteLLM** as the gateway for hard per-team/per-agent budget caps.

## Sources

- OpenTelemetry GenAI semantic conventions — <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
- Claude Code monitoring/telemetry — <https://code.claude.com/docs/en/monitoring-usage>
- LiteLLM (proxy, budgets, custom pricing) — <https://github.com/BerriAI/litellm> · <https://docs.litellm.ai/docs/proxy/custom_pricing>
- Langfuse (self-host, model cost) — <https://github.com/langfuse/langfuse> · <https://langfuse.com/docs/model-usage-and-cost>
- OpenLIT (cost + GPU collector) — <https://github.com/openlit/openlit> · <https://docs.openlit.io/latest/features/pricing>
- Arize Phoenix — <https://github.com/Arize-ai/phoenix>
- OpenLLMetry / Traceloop — <https://github.com/traceloop/openllmetry>
- Helicone — <https://github.com/helicone/helicone>
- Portkey gateway — <https://github.com/Portkey-AI/gateway>
- ccusage — <https://github.com/ryoppippi/ccusage>

## Caveats to re-verify before adopting

- **LiteLLM:** PyPI **1.82.7 / 1.82.8 were malware-compromised** — pin a known-clean version, never `latest`;
  proxy needs Postgres (+Redis for multi-replica budgets); hard tag-budget *enforcement* is Enterprise (tag
  *tracking* is OSS).
- **Langfuse:** no native budget alerting found; Metrics API **v2 is Cloud-only** for now; heavy 4-datastore
  self-host.
- **OpenLIT + AMD/MI250:** GPU/energy only via the **standalone sysfs collector** (in-process SDK GPU path is
  NVIDIA-only); **gfx90a not explicitly validated**, no carbon tracking — test on our nodes before relying on it.
- **ccusage:** local logs only (sync across ce-master + mi250 nodes yourself); on a Pro/Max subscription the "$"
  is an API-equivalent estimate, not real billed spend.
- Star counts / vendor tiers are point-in-time (~Jul 2026) and drift.
