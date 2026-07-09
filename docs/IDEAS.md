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

### Internal-demand-driven contributions: mine first-party roadmaps (Confluence/Jira) for upstream needs
- **What:** a new candidate source that reads **our own org's** design docs / roadmaps (Confluence, Jira, internal
  repos — via the **Atlassian MCP** already wired into this session) and infers the **upstream** vLLM / llm-d /
  SGLang feature that would unblock what we're actually building. e.g. an internal design doc for an
  agentic-workflow gateway that needs **session affinity / prefix-cache keeping** → infer "vLLM needs a
  first-class **session-id tag / session-aware routing** feature" as a contribution candidate.
- **Why / value:** the **highest-alignment** source there is — you contribute what your org will actually *use and
  maintain* (dogfooding), so the work is high-priority, well-motivated, and sustained. Shifts the agent from
  "track the public ecosystem" to "advance upstream in the direction of **first-party demand**": contributions
  land where they pay us back.
- **How (concrete):**
  - **Ingest:** Atlassian MCP (Confluence pages + Jira issues under a designated space/label set) + internal
    repos; a periodic pull, same cadence shape as collection.
  - **Infer (LLM step):** "given this internal need, which upstream feature in {vLLM, llm-d, SGLang, …} would
    enable/simplify it, and is it missing today?" → a candidate `{internal-need, implied-upstream-feature, target
    repo, evidence}`.
  - **Rank + feasibility-gate** like every candidate (MI250-verifiable? effort? merge-likelihood?), with a
    **first-party-demand boost**. Same machinery as `candidates.md` §2 "roadmap-derived", a different corpus.
- **Privacy / security (non-negotiable — this source touches confidential material):** internal docs are a
  **discovery/prioritization signal only**. The eventual **public PR must be justified on public technical merits
  and must never quote or leak internal strategy / product names** (scrub the output). LLM processing of internal
  docs stays under our own account; the mandatory human gate is the backstop. This extends the existing "never
  spray public without human confirmation" rule to "never leak internal context into a public artifact." (Also:
  keep internal specifics out of committed repo docs — this entry uses a deliberately generic example.)
- **Scope / effort:** M–L — Atlassian ingestion + the internal→upstream inference step + privacy scrub +
  integration as a scout source.
- **Depends on / risk:** the scout/candidate plumbing (M2, built) + ideally the `candidates.md` §2 roadmap-derived
  source landing first (shared machinery); **Atlassian MCP availability in the unattended runtime** (an
  interactively-authed MCP can be absent in headless/cron — a real gap for a scheduled loop); **privacy is the top
  risk**; the internal→upstream inference can hallucinate a "need" — the feasibility + evidence + human gates
  contain it.
- **Status:** shaping — graduate to a **DEVPLAN scout-source todo** (M2/M3 area) once §2 roadmap-derived lands and
  the Atlassian ingestion + privacy scrub are settled. Not milestone-ready until the ingestion path, the inference
  prompt/validation, and the privacy scrub are pinned.

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
