# Inference Trend Tracking & Autonomous vLLM (ROCm) Contribution Agent — Project Plan

> **One-line goal:** Continuously track issues and PRs across several inference-serving repos to distill the
> direction of the inference market and its key techniques, **grade its own predictions to evolve its tracking
> criteria**, and use those signals to run an **always-on, self-improving agent** that goes from
> **discovering vLLM (ROCm) contribution candidates → generating and testing patches → human review → PR**.
>
> **Nature of the project:** ① A domain-specialized agent with its own benchmark (inference serving /
> retrospective grading) · ② Open-ended, self-evolving (evolving tracking criteria + a self-prediction grading
> loop).
>
> **Core objective:** Submit **actual PRs** to vLLM (ROCm path prioritized). Not a one-off — a **continuous
> deliverable** where the agent keeps discovering and refining PR candidates over time.
>
> **Hardware:** The agent runtime only needs a CPU (no GPU required). The GPUs (**3x MI250, ROCm**) are used only
> for building/testing vLLM and reproducing/verifying ROCm bugs.

---

## Design principles

1. **Continuous operation and self-evolution are the essence.** Not build-once-and-done — it keeps running on a
   schedule and keeps improving its own tracking criteria and judgment. Milestones are an *order*, not a
   *deadline*.
2. **Human-in-the-loop is mandatory.** The agent only goes as far as "proposing"; a human reviews before any
   upstream PR is finally submitted. No automated PR spam (reputation risk).
3. **Every claim is backed by evidence.** Summaries and rankings must cite links to the actual issues/PRs.
4. **Everything is evaluable.** The agent's judgments (trend predictions, contribution-candidate rankings) are
   quantitatively measured via a retrospective benchmark, and the results feed back into updating its own
   heuristics.

---

## Development environment & the AMD/ROCm target (strategic advantage)

Having 3 MI250s as the available hardware isn't a constraint — it's a **differentiator**:

- **A less crowded niche.** vLLM's ROCm support is less mature than its CUDA path, meaning there are **more
  unresolved gaps and bugs** = more contribution opportunities.
- **Reproducible physical hardware.** Many vLLM ROCm issues sit unaddressed simply because "there's no AMD
  hardware to reproduce them on." Being able to actually **reproduce, test, and verify** on MI250 is exactly what
  maintainers want most, and it substantially boosts PR credibility.

### Repo weighting (the ROCm source is a two-tier structure)
- **`vllm-project/vllm` (upstream, main repo) — primary target.** Prioritize `rocm`/`amd`-labeled issues and
  ROCm-specific build/kernel/performance failures. **PRs are ultimately submitted here.** Tracking the "AMD
  Development Roadmap (2026 Q2)" issue (#44092) and the broader ROCm roadmap surfaces AMD's priorities (e.g.,
  achieving decode-performance parity with SGLang/ATOM at high concurrency) = the basis for a parity-gap
  strategy.
- **`ROCm/vllm` (AMD's official downstream fork) — primary secondary source.** Where ROCm optimizations often
  land before upstream → source for finding **"exists in the fork but not upstream"** contribution opportunities.
  Downstream-to-upstream porting is a classic, well-received PR pattern.
- **SGLang** — has ROCm support → a **performance-parity comparison baseline** + trend source.
- **NVIDIA Dynamo** — NVIDIA-centric → **trend radar only** (for reading overall direction).
- **llm-d** — a trend source for serving orchestration.

> MI250 = gfx90a (CDNA2) → supported on the mainline ROCm path, so it works fine as a reproduction/test
> environment.

### Why watch multiple repos (the link back to vLLM PRs)
These projects solve the same underlying problems (paged KV cache, continuous batching, speculative decoding,
disaggregated prefill/decode, quantization/kernels) in different, competing/imitating ways, so cross-repo
observation feeds into (1) **discovering parity gaps** (things present in other repos/forks but missing
upstream) and (2) **picking up design context and terminology** (writing PRs that don't get rejected). That
said, the primary source for landing any individual PR is vLLM's own issues, tests, and `good first issue`
labels.

### Contribution scope principles (kernel authoring excluded; focus on repro & fixes)
**Not doing:** writing new GPU kernels (Triton/Composable Kernel/TileLang-style work), compiler/graph-optimization
engine-level work, TensorRT-style porting (NVIDIA-only, not applicable at all). These carry high risk, long
review cycles, and are outside the intended expertise.

**Targeting (leveraging the MI250 advantage without needing to author kernels):** correctness bugs in ROCm code
paths (reproduce → fix → verify), build/packaging failures, missing ROCm config/feature flags, dtype/model
support gaps, ROCm CI/test coverage, performance regressions caught via profiling (Python/config-level), and
upstream-porting of `ROCm/vllm` fork enablement work.

In short, the direction is not "write new kernels" but **reproduce · debug · enable · fix regressions**.

---

## Architecture (at a glance)

```
[Collector] GitHub API/GraphQL → issues, PRs, releases, commits (vllm-project/vllm, ROCm/vllm, SGLang, Dynamo, llm-d)
   ↓
[Normalization + RAG index] Store embeddings of text/labels/diffs
   ↓
[Classification/ranking agent] Classify by taxonomy + score importance (with cited evidence links)
   ↓
[Evolution loop] Propose new categories / retire dead topics + retrospectively grade past predictions → update heuristics
   ↓
[Contribution-candidate discovery] Rank ROCm-reproducible issues / good-first-issues / parity gaps
   ↓
[Patch agent] Create branch → write code → test on MI250 → request human review
   ↓
[Output] Weekly report + PR candidate queue (continuously refreshed on a schedule)
```

**Recommended stack:** Python, GitHub REST/GraphQL API, embeddings + a lightweight vector store (pgvector or
LanceDB), Claude Code/Devin-style tools for the coding stages. Agent runtime on CPU; MI250 used only for vLLM
testing.

---

## Progress stages (milestones — an order, not a deadline)

Each stage leaves behind a "working deliverable" before moving to the next. Even after completion, the agent
keeps running and keeps improving itself.

**M0 — Foundation: collection + baseline summary**
A scheduler + RAG index that collects and stores data from the 4 sources via the GitHub API. A v0 weekly summary
report that classifies using a fixed taxonomy only (each item cites its PR link). ← *The repo currently has the
collector skeleton implemented up to this point.*

**M1 — Self-evolution loop (the differentiator)**
Taxonomy self-evolution (propose new categories / retire dead topics) + self-prediction logging (store
predictions like "this PR/technique will become important," timestamped).

**M2 — Evaluation + contribution-candidate discovery**
A retrospective benchmark (compare past predictions vs. actual outcomes → precision/recall → update heuristics)
+ vLLM contribution-candidate ranking (ROCm-reproducible issues · `good first issue` · parity gaps, ranked by
risk).

**M3 — First contribution: patch → human review → actual vLLM PR**
Starting from the lowest-risk candidates: generate a patch → test on MI250 → human review gate → upstream PR.
Goal: get an actual PR submitted (docs/typing/tests/small ROCm bugs are sufficient to start).

**M4 — Sustained autonomous operation**
Always-on via schedule: the weekly report and PR-candidate queue keep refreshing, and scoring keeps improving
based on self-prediction grading results.

---

## Human review gate — implementation approach
- **Default (recommended, zero extra development):** `gh` CLI + a draft PR on my fork + the GitHub diff view.
  The agent commits patches to a branch on my fork → `gh pr create --draft` → review via the GitHub diff screen
  → promote to the upstream vLLM repo once approved. Supplementary tools: VS Code/Cursor diff, GitHub Desktop,
  `delta` in the terminal.
- **Stretch (if there's spare time):** a local web review dashboard that surfaces candidate patches in one place
  (risk badges + diff preview + approve/hold buttons).
- ⚠️ Nothing goes upstream **without human approval first.**

---

## Getting started (first steps)

1. **Create a public GitHub repo.** Start the README with the goal and an architecture sketch.
2. **Get one successful vLLM (ROCm) build on MI250.** The prerequisite for every later "reproduce and verify"
   step.
3. **Minimal collector.** Pull `vllm-project/vllm` + `ROCm/vllm` issues via the GitHub API into local storage.
   ← *Implemented (`src/collector.py`)*
4. **One baseline summary.** Hand-write a single weekly summary with cited evidence links once, then have the
   agent learn to reproduce it.
5. **Wire up the scheduler.** Run automatically daily/weekly → this is where "always-on" actually starts. Layer
   the M1 evolution loop on top after that.

---

## Risks & mitigations
- **Risk of automated PR spam** → fixed human review gate, start with the lowest-risk PRs.
- **ROCm build/environment pitfalls** → finish environment validation early in M0; use containers for
  reproducibility.
- **GitHub API rate limits** → auth token + incremental collection (caching).
- **Scope creep** → lock in a "working deliverable" at each stage; keep high-difficulty work like kernel
  authoring explicitly out of scope.
- **Possibility that a PR never merges** → regardless of merge outcome, "agent + benchmark + open PR + demo" is
  the core asset.
