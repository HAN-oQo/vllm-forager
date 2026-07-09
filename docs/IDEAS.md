# IDEAS & Next-Version Roadmap

> A low-friction place to capture ideas **before** they're committed work. Dump freely in **Inbox**; shape the
> ones worth doing; promote the committed ones to the **vNext roadmap**; when concrete, they graduate into
> `docs/DEVPLAN.md` as milestones/todos.
>
> `DEVPLAN.md` = what we're building **now** (test-backed todos). **This file = what might come next** — no
> commitment, no tests required. Keep the two separate so the plan stays trustworthy and ideas stay cheap.

## How to use (30 seconds)
1. Have an idea? Add a bullet under **📥 Inbox** — one line is fine.
2. Worth pursuing? Move it to **🔨 Shaping** and fill the template (why / scope / risk).
3. Decided to do it next version? Move to **🗺️ vNext roadmap**.
4. Started building it? Create the `Mx`/`Tx` todos in `DEVPLAN.md` and move the idea to **✅ Graduated** (link it).
5. Not doing it? Move to **🧊 Parked** with a one-line reason (so it isn't re-litigated later).

## Idea template
```
### <short title>
- **What:** one line.
- **Why / value:** who benefits, what it unlocks.
- **Scope / effort:** S / M / L.
- **Depends on / risk:** prerequisites, unknowns.
- **Status:** inbox | shaping | vNext | graduated | parked
```

---

## 📥 Inbox
*(raw captures — replace these seed examples with your own)*
- _(seed)_ **MCP server** exposing the KB (candidate queue / trends / parity) to other tools — deferred in `CONTEXT.md`.
- _(seed)_ **RAG beyond human-defined metrics**: let the system pose its own eval queries / self-label once the
  T1.8 golden-set score is stable (ShinkaEvolve's open question).
- _(seed)_ Extend tracking beyond the 5 repos / to other accelerators once the ROCm loop is proven.

## 🔨 Shaping
*(ideas being fleshed out — add the template fields)*

### Cheap issue classification: rules → local classifier → LLM cascade (stop paying `claude -p` per issue)
- **What:** replace "send every issue to `claude -p` to classify" with a **confidence-gated cascade**: (1) free
  rules/labels first (`config.ROCM_HINTS`/`SPEECH_HINTS`, GitHub labels, regex) resolve the easy majority; (2) a
  **local classifier on the already-computed embeddings** (`src/embed.py`) — SetFit / a linear head / centroid-kNN
  — labels most of the rest **with a confidence score**; (3) only **low-confidence** items escalate, first to the
  **local vLLM backend** (`LLM_PROVIDER=local`), and only the hardest tail to Claude. Close the loop with
  **distillation**: Claude's (expensive, confirmed) labels become training data that keeps shrinking the LLM share.
- **Why / value:** classification is high-volume + latency-insensitive (24h cadence) — the worst possible thing to
  spend frontier tokens on one-at-a-time. A rules+embedding cascade typically resolves **80–95% with zero LLM
  calls**; `T0.7` cost-meta + `src/llm_bandit.py` already exist to *measure* and *route* the savings. "Local
  BERT" is directionally right — the catch below is why it can't be a naive frozen model.
- **The catch that shapes the design (evolving taxonomy):** our taxonomy is **versioned + self-evolving** (grader
  / curator update it), so a statically fine-tuned BERT goes stale on every taxonomy bump. The cheap tier must be
  **refit-in-seconds** (linear head / centroid on frozen embeddings) or **example-based** (kNN / SetFit that
  updates by *adding* examples, no retrain) so it tracks the evolving labels for free — self-evolution applied to
  classification.
- **Quick wins (low-effort, independent of the classifier, could graduate to DEVPLAN soon):** batch N issues per
  prompt instead of one-per-call; use the **Anthropic Message Batches API** (~50% cheaper, async — fine at 24h
  cadence); **prompt-cache** the taxonomy/instruction preamble across calls. Big spend cut, no architecture change.
- **Guardrail (mirror T1.8):** a labeled **classification gold set** + accuracy/macro-F1 + a **confidence
  threshold** so the cheap path never silently degrades quality (escalate below threshold; track drift). **Active
  learning:** the low-confidence items you escalate are also the most informative to *train* on — same tail, two
  wins.
- **Scope / effort:** M — new `src/classify.py` (cascade + confidence + gold-set eval) over existing `embed.py`;
  reuse `llm.py`/`llm_bandit.py` for the escalation tier; a small labeled set. Quick wins are S.
- **Depends on / risk:** embeddings available at classify time (they are — data plane computes them for RAG);
  **cold-start** (few labels early → start LLM-heavy, shift to cheap as labels accumulate via distillation); keep
  the gold-set gate so cost-cutting ≠ quality regression. Relates to `docs/research/cost-tracking.md` and the
  `LLM_PROVIDER=local` vLLM backend.
- **Status:** shaping (quick wins are near-DEVPLAN; full cascade lands once the M1 intelligence plane exists to
  classify against).

### Buy-not-build the operator console: adopt an OSS dashboard instead of hand-rolling
- **What:** stop growing the bespoke `dashboard/` (the M1 thin slice that renders the whole KB per request — the
  DEVPLAN `T5.16` scale problem). Offload the generic + observability parts to mature OSS; keep only the bespoke
  domain views custom.
- **Why / value:** the operator console is the most important surface, and scale / pagination / alerting / theming
  / RBAC are exactly what battle-tested tools already solved — far less code to own than re-deriving it.
- **The shape that fits (no single tool covers all — the console is three different natures):**
  - **Ops / observability half → Grafana.** Live health ("what's running", `T5.8`), cost-per-agent over time
    (`T5.13`), guardrail drift with thresholds + **alerting** (`T5.7`), trend charts (`T5.3`). Grafana is
    purpose-built for exactly this and the most-deployed self-hosted viz tool; adopting it deletes those custom
    panels and gives alerting for free. Needs a time-series / SQL / JSON datasource.
  - **Interactive + write-back half → a low-code builder over our read API, or a thin custom app.** Candidate
    "work this" select (writes a decision), patch approve/hold, tables. **Appsmith** (dev/JS, Apache-2.0; connects
    Postgres/Mongo/**REST**/GraphQL) or **ToolJet** (visual+code, AGPL, built-in PG DB) give **natively paginated**
    tables + forms + buttons over the `T5.1` read API; **Budibase** (no-code, GPL) for the simplest CRUD.
    Python-native alternative: **Dash + AG Grid server-side row model** (100k+ rows) or **NiceGUI** (app-style
    monitoring UIs). ⚠ plain **Streamlit does NOT fix scale** — its rerun + per-session-RAM model is the *same*
    full-KB trap unless you paginate and push filtering to the DB.
- **The real prerequisite (why `T5.16` isn't wasted even if we buy):** every one of these tools scales by
  **querying a paginated/aggregated DB, not loading the whole KB** — which *is* `T5.16`'s core (read API +
  pagination + a queryable backend). So `T5.16` isn't competing with adoption; it's the **seam that enables** it.
  Deciding factor is the KB backend: Grafana/BI want SQL/time-series, builders want SQL/REST — so this likely
  pushes the KB toward **Postgres, or exposing `T5.1` as REST** (Firestore isn't first-class in these tools).
- **What stays custom (~30–40%):** the bespoke 大→소→소소 taxonomy tree with node summaries and the per-issue
  attempt report don't map to generic BI widgets — hand-build or embed those; offload the generic ~60% (charts,
  tables, health, cost, guardrails, pagination).
- **Cost / caveat:** each tool is another self-hosted service to deploy/patch/secure on ce-master (the "hidden
  engineering tax"). Clearly worth it for the **observability half (Grafana)**; for the interactive half, a
  builder-service vs. a thin custom app over the same read API is a real toss-up for a single-user, human-gated tool.
- **Scope / effort:** M — Grafana for the ops half is S–M once run/cost events are emitted in a datasource shape;
  the interactive-half build-vs-buy call rides on the `T5.16` backend/read-API work.
- **Depends on / risk:** the KB backend + `T5.1` read API (the seam); license compliance (Grafana/Superset/
  Metabase/ToolJet AGPL, Budibase GPL, Appsmith Apache-2.0); operating extra services. Relates to
  `docs/design/pipelines/README.md`, DEVPLAN `T5.16` (foundational) + `T5.15`.
- **Status:** shaping — recommend adopting **Grafana** for the ops/observability half early (clear win); for the
  interactive half, defer build-vs-buy until `T5.16`'s read-API/backend lands, then evaluate **Appsmith/ToolJet
  over the read API** vs. a thin custom app.
- **Sources (2026):** [Grafana vs Superset vs Metabase](https://www.modern-datatools.com/compare/metabase-vs-superset-vs-grafana)
  · [Appsmith vs ToolJet vs Budibase](https://blog.tooljet.com/appsmith-vs-budibase-vs-tooljet/)
  · [Streamlit/Dash/Reflex/NiceGUI at scale](https://reflex.dev/blog/streamlit-vs-dash-python-dashboards/).

### Latent-need synthesis: derive not-yet-filed upstream features from cross-repo ecosystem trends
- **What:** the discovery agent shouldn't only find *observed* signals (a capability that exists elsewhere, a
  filed issue, a published roadmap line) — it should **synthesize a latent upstream feature that nobody has built
  or filed yet**, by connecting **public** cross-repo trends. Worked example: watch **llm-d**'s recent PRs/issues →
  infer it's moving toward **agentic workflows** → reason "integrating that with vLLM needs stable **session
  identity**" → conclude "**vLLM needs a session-id tag / session-aware routing**" and surface *that* as a
  contribution candidate — *before anyone opens the issue*.
- **Why / value:** the top of the value chain — proactive, anticipatory contributions that put vLLM **ahead of**
  where the ecosystem is heading, not reactive bug/gap chasing. It's the agent reasoning like a systems architect
  reading the whole field — the project's real thesis ("read the ecosystem → find where to contribute") in its
  strongest form. Evidence is **public** (the motivating llm-d PRs + vLLM's current absence), so no privacy issue.
- **How (concrete):**
  - **Read direction, not just items:** feed the synthesis the ecosystem signals the intelligence plane already
    computes — `trends.py` category momentum, the analyst's classifications, the forecaster's "what's becoming
    important" — across llm-d / SGLang / Dynamo / vllm-omni, plus each repo's recent PR/issue themes.
  - **Synthesize (LLM step):** "given where {llm-d, SGLang, …} are heading, what primitive/feature will {vLLM,
    vllm-omni} need to integrate with or keep pace, that doesn't exist there yet?" → a candidate `{trend evidence,
    implied upstream feature, target repo, integration rationale}`.
  - **Validate it's actually latent (the crux):** confirm the feature isn't already shipped or already filed on the
    target (search the target repo / the parity matrix) — otherwise it collapses into an ordinary parity-gap or an
    existing issue. The whole value is finding the *unfiled* need **without hallucinating one**.
  - **Rank + feasibility-gate + human gate** like every candidate.
- **Relation to existing sources:** today's scout sources are all **observed** (parity-gap = exists elsewhere;
  good-first / rocm = filed; roadmap-derived = published). This is the **synthesized / foresight** source, sitting
  at the **forecaster ↔ scout** seam (the forecaster already predicts "what will matter"; this turns that into
  "what feature to build"). Hardest source → highest hallucination risk → the latent-check + feasibility + human
  review are load-bearing, not optional.
- **Variant — first-party demand (privacy-gated):** the same synthesis run over our *own* Confluence/Jira roadmaps
  (via the Atlassian MCP) instead of public trends — highest alignment (dogfooding), but internal docs are a
  **prioritization signal only**: the public PR must be justified on public technical merits and **never quote or
  leak internal strategy / product names** (scrub the output; keep internal specifics out of committed docs too).
  Extends "never spray public without human confirmation" to "never leak internal context into a public artifact."
  Caveat: an interactively-authed MCP can be absent in the unattended/cron runtime.
- **Scope / effort:** L — a synthesis step over the intelligence plane's trend outputs + a "does it already exist"
  latent-validation + integration as a scout source.
- **Depends on / risk:** the intelligence plane (analyst / trends / forecaster, M1) + scout (M2, built);
  **hallucinated needs are the top risk** (a derived feature that's unwanted or already exists) — contained by the
  latent-check, feasibility gate, and mandatory human review; the private variant adds the privacy gate.
- **Status:** shaping — the most ambitious candidate source. Graduate to a **DEVPLAN scout-source todo** (M2/M3
  area) after the simpler observed sources (`candidates.md` §2 roadmap-derived) land and the synthesis's
  hallucination controls (the latent-validation especially) are designed. Not milestone-ready until the synthesis
  method + latent-validation + evidence shape are pinned.

### Reproduce-the-news: turn external announcements into ROCm contributions on MI250
- **What:** watch **external signals beyond GitHub** — HF blog posts, vendor/release announcements, changelogs,
  papers — pick the ROCm-relevant ones, and **actually run them on MI250 to see if they hold on ROCm**. When a
  CUDA-benchmarked feature breaks (or silently underperforms) on gfx90a, *that failure is the contribution point*.
  Uses MI250 as an active **prober**, not just a verification gate.
- **Worked example (this exact blog — [native-speed vLLM transformers backend](https://huggingface.co/blog/native-speed-vllm-transformers-backend)):**
  vLLM's transformers backend (`--model-impl transformers`) claims native speed — but it's benchmarked **only on
  8×H100 (CUDA)**, uses **CUDA Graphs** + `torch.compile` + fused vLLM kernels, plugs vLLM's attention at runtime,
  and one benchmark is **FP8 MoE**. None of that is validated on ROCm. ROCm risk hypotheses the agent would test on
  MI250: (a) the runtime-plugged **attention backend** differs on ROCm (triton/rocm-flash) → may fail or fall
  back; (b) **CUDA Graphs → HIP Graphs** may not be wired for this codepath; (c) **FP8 is effectively an MI300
  feature** — the FP8-MoE path likely won't run on **gfx90a (MI250)** at all; (d) the fused kernels
  (`MergedColumnParallelLinear`/`QKVParallelLinear`) need ROCm builds. → run `vllm serve Qwen/Qwen3-4B
  --model-impl transformers` on MI250 and compare to `--model-impl vllm`: **errors**, **slower on ROCm**, or
  **FP8 unsupported on gfx90a** are each a real, reproducible, well-evidenced contribution.
- **Why / value:** the **purest expression of the hardware edge** — the whole reason to hold MI250 is that nobody
  else checks whether the CUDA world's latest thing works on ROCm. Empirical, high-signal, and every failure ships
  with a ready-made repro (exact command + traceback). A concrete instance of the "active bug-finding"
  post-pipeline vision, sourced from news and driven by reproduction.
- **How (concrete):**
  - **Ingest:** a curated external feed set (HF blog, vLLM/ROCm release notes, vendor blogs, arXiv) polled via
    RSS/WebFetch/WebSearch, then an LLM "ROCm-relevant + reproducible on an inference box?" filter (ROCM_HINTS-aware).
  - **Repro = the M3 Engineer pointed at a *claim*, not a filed issue:** turn the announcement into a runnable
    MI250 smoke test (`repro.py`/`runner.py`), run it, capture pass/fail + logs + a perf delta vs the baseline.
  - **Outcome → candidate + attempt report (T3.12):** fail → a bug/perf contribution with the repro attached;
    pass-but-slow → a perf-parity candidate; pass → a positive "works on ROCm" data point (still worth publishing).
- **Relation:** the **empirical** sibling of "Latent-need synthesis" (that one *reasons* from trends with no
  execution; this one *runs* an external claim on real hardware). Together with roadmap-derived + the observed
  GitHub sources, these form a **pluggable discovery-source catalog** the scout should grow.
- **Scope / effort:** L — external-feed ingestion + a prose→repro harness (the hard part) + the M3 Engineer/MI250
  runner + attempt-report output.
- **Depends on / risk:** the **M3 Engineer / repro / MI250 harness (not built yet)** is the real dependency — this
  *is* that harness pointed at external claims; a vLLM ROCm build must exist on the node; the prose→repro step can
  misreproduce (false gaps) → run logs + human review contain it; a relevance filter is needed so it doesn't chase
  CUDA-only news that can't matter on ROCm.
- **Aspiration (owner):** the north star is the agent doing this **fully end-to-end autonomously** — find the
  announcement, **stand up the vLLM ROCm env on MI250 itself**, run it, diagnose the failure, and propose the fix,
  with a human only at the final PR gate. Today it's blocked on the M3 build/repro harness *and* the fact that the
  MI250 nodes carry no standing vLLM env (mi250-051 is gfx90a + ROCm but has no vLLM installed), so **env-standup
  is part of the capability**, not a precondition to assume.
- **Status:** shaping — graduate once the M3 reproduction harness exists (then this is mostly a new *source*
  feeding it) + the external-feed ingestion + relevance filter are designed.

### Prompt/task-aware model router for cost (OpenRouter-style auto-routing)
- **What:** route each `llm.complete` call to the **cheapest model that can handle it**, decided per request —
  like **OpenRouter's Auto router** (a classifier picks the model for the prompt). Two signals: (a) a **static
  per-task tier** (the task is known at the call site — classify / score / novelty-judge = trivial → cheap;
  claim / node-summary / forecast = mid; patch-writing / adversarial-review / latent-synthesis = frontier), and
  (b) a **prompt-difficulty estimate** where the task type alone isn't decisive. Plus **escalate-on-failure**: run
  the cheap model first, and if its output fails validation / confidence, retry on a stronger one.
- **Why / value:** most calls in this system are trivial/structured (enum classifications, boolean judgments) and
  are wasteful on a frontier model; only a few (patching, review, synthesis) truly need one. A router captures the
  bulk of the savings the cost work (T4.8–T4.11) only *measures / caps* — this is spend-shaping at the routing
  layer, upstream of the meter.
- **How (concrete):** implement the router **inside `src/llm.py`** (the existing provider seam), keyed by a
  `task`/`tier` hint each agent passes (or inferred from the prompt). Backends: the existing `claude_cli` /
  `claude_api` / **local vLLM** (cheap tier) providers; optionally **OpenRouter itself** as one backend
  (`LLM_PROVIDER=openrouter`) to reuse its Auto routing + model catalog — with the caveat it's a third party that
  sees prompt data. **LiteLLM (T4.10)** also does routing/fallbacks, so the router and the budget gateway can be
  one layer.
- **Relation to what exists:** complements **`llm_bandit.py` (T2.6)** — the bandit learns *which provider* is best
  from **outcomes over time**; this router picks a *tier* per request from the **prompt/task** up front, and the
  two compose (router picks the tier, bandit picks within it). It's also the general form of **T4.11 / the
  classification cascade** (that cascade is this router applied to one task), and it needs **T4.8/T4.9 cost
  capture** to prove the savings and tune the tiers.
- **Scope / effort:** M — a routing layer in `llm.py` + per-task tier hints at the call sites + escalate-on-failure;
  a prompt-difficulty classifier only where a static tier isn't enough.
- **Depends on / risk:** a misroute (cheap model on a hard task) degrades quality → contained by output validation
  + confidence + escalate-on-failure; the routing decision itself must be far cheaper than the savings (prefer
  static task tiers; use a prompt classifier sparingly); an external router (OpenRouter) adds a third-party
  data-sharing + availability dependency. Relates to `docs/research/cost-tracking.md`, `llm_bandit.py`, T4.8–T4.11.
- **Status:** shaping — strong cost lever; sequence after T4.8/T4.9 cost capture exists (so the tier choices are
  data-driven, not guessed), and it naturally subsumes the T4.11 cascade as its first instance.

> **Post-pipeline vision (owner's, sequenced).** These kick in *after* the full M0–M5 pipeline is complete and the
> agent workflow is running. They're gated in order: keep the loop healthy → earn a merge track record →
> generalize → scale into teams.

### Continuous maintenance + active bug-finding + security auditing
- **What:** once the full pipeline runs, the dev loop doesn't just maintain — it **proactively hunts bugs**
  (adversarial/fuzz-style discovery, not only tracking others' reported issues) and runs **security checks** on the
  target code paths.
- **Why / value:** shifts the agent from *reactive* (track → fix reported issues) to *proactive* (surface
  unreported bugs + vulnerabilities) — higher-value, more novel contributions, and long-term system health.
- **Scope / effort:** L — new finder/fuzz + security-audit agents layered on the contribution plane.
- **Depends on / risk:** full pipeline live; security work stays **defensive + human-gated** (no offensive or
  mass-scanning use); needs false-positive/noise control so findings stay trustworthy.
- **Status:** shaping

### Generalize: vLLM-specific → repo-agnostic contribution framework
- **What:** after a **track record of merged vLLM PRs**, extract the vLLM-specific parts into a **general agent
  framework for advancing any target repo** — point it at repo X and it tracks / finds / patches / PRs there.
- **Why / value:** the real asset is the *loop*, not the vLLM specialization; generalizing multiplies impact and is
  a product in its own right.
- **Scope / effort:** L — config-drive sources/taxonomy; abstract the ROCm/MI250 check into a pluggable
  **"verification oracle"** per target repo.
- **Depends on / risk:** proven merges first (credibility gate — DGM-style: earn the track record *before*
  generalizing); over-abstraction risk — generalize only what merged PRs actually proved.
- **Status:** shaping

### Scale the maintenance / dev loop into a team
- **What:** the maintenance dev-loop becomes a **team** (not a single session) — parallel maintainers across
  subsystems with a validation/merge gate.
- **Why / value:** throughput + coverage once the work exceeds one context; extends DEVPLAN's "graduate to an agent
  team at M3/M5" note to standing operation.
- **Scope / effort:** M–L — git worktrees, tight per-agent scope, a lead/validator, human still merges.
- **Depends on / risk:** work must split into cleanly separable dirs; coordination + token overhead (Cognition's
  "don't build multi-agents" caveat) — only when the split is genuinely real.
- **Status:** shaping

### Scale the vLLM contribution agent into a team
- **What:** the PR-developing (contribution-plane) agent becomes a **team** — reproducer / patcher / verifier /
  reviewer as specialized roles working candidates in parallel.
- **Why / value:** more candidates advanced concurrently; role specialization (repro vs patch vs adversarial
  review) raises quality — a natural extension of the M3 ensemble self-review.
- **Scope / effort:** L — orchestration across the 3 MI250 nodes, per-role prompts, one shared human gate.
- **Depends on / risk:** MI250 capacity; the **human review gate stays mandatory**; guard against PR-spam /
  reputation risk.
- **Status:** shaping

### Hybrid dev-loop: cheap-model build + Claude review (2-session split)
- **What:** split the dev-loop into a **build session** on a cheap coding model (e.g. GLM 5.1 via z.ai's
  Anthropic-compatible endpoint) that implements todos + opens PRs, and a separate **review session** on Claude
  that runs `/code-review --comment` on each open PR; ambiguous/"hard" todos use the existing STOP-and-ask seam →
  handled on Claude.
- **Why / value:** run the bulk of coding on a much cheaper model while keeping Claude's adversarial review where
  it matters — best cost/quality. Needed because Claude Code is **one model per session**
  (`ANTHROPIC_BASE_URL` is process-level), so build vs review **cannot differ inside a single `/dev-loop` run**.
- **Scope / effort:** M — edit `.claude/commands/dev-loop.md` (drop the in-session `/code-review` step → just open
  the PR), add a Claude "PR reviewer" flow/skill over open PRs, update `docs/RUNBOOK.md` (two clones/sessions + the
  env split: build clone = z.ai base URL, review clone = Anthropic), keep the human merge gate.
- **Depends on / risk:** only worth it once PR volume is high **or** GLM-run `/code-review` proves too weak —
  today **all-GLM** (human merge + CI are the real gate) or **all-Claude** is simpler. A z.ai base URL **disables
  Remote Control** on the build session (use Termius there). Relates to `docs/research/cost-tracking.md`.
- **Status:** shaping / later (deferred by decision — the current dev-loop stays single-model).

## 🗺️ vNext roadmap (committed for the next version)
*(the shortlist that will become DEVPLAN milestones/todos next)*

## ✅ Graduated (now tracked in DEVPLAN)
*(moved into `docs/DEVPLAN.md` — link the milestone/todo)*

## 🧊 Parked (not now)
- _(seed)_ A2A / heavy multi-agent **product** features — deferred to keep the core loop focused (`CONTEXT.md`).
