# CONTEXT — Design Decision Log (for resuming work)

This document summarizes the decisions finalized during pre-project brainstorming.
Read this file first when resuming work on a CPU node, or when handing context to an AI assistant.

## One-line project summary
Continuously track issues/PRs across inference-serving repos → distill trends + evolve tracking criteria by
grading its own predictions → use those signals to discover vLLM (ROCm) contribution candidates → generate/test
patches → a self-improving agent that goes all the way to **an actual PR after human review**.

## Confirmed core decisions

1. **Target platform = AMD/ROCm (not CUDA).**
   Reason: available hardware is 3x MI250. The ROCm path is less mature than CUDA → more contribution
   opportunities and less competition. Having reproduction hardware is itself an edge on neglected ROCm issues.

2. **Hardware role separation.**
   - Agent runtime (collection · RAG · classification · evolution · patch-generation code) = **start development
     on a CPU node.**
   - MI250 is used only for validating the vLLM build (once, early on) + patch testing/bug reproduction (M3).

3. **Tracked repos (weighted).**
   - `vllm-project/vllm` — primary target, where PRs get submitted. Prioritize ROCm/AMD-labeled issues. Track
     the roadmap issue #44092.
   - `ROCm/vllm` — AMD's official downstream fork. Source for upstream-porting opportunities.
   - `sgl-project/sglang` — performance-parity comparison + trend source.
   - `ai-dynamo/dynamo` (NVIDIA Dynamo) — trend radar only.
   - `llm-d/llm-d` — serving-orchestration trend source.
   > Check/adjust exact repo slugs in `src/config.py`.

4. **Contribution scope — kernel authoring excluded.**
   Won't do: new GPU kernels (Triton/CK/TileLang), compiler/graph-optimization work, TensorRT porting
   (NVIDIA-only, not applicable).
   Will do: reproduction · debugging · enablement · regression fixes — correctness bugs, build/packaging,
   config/feature-flag gaps, dtype/model support, ROCm CI/tests, Python/config-level performance regressions,
   fork-to-upstream porting.

5. **Human review gate.**
   Default: `gh` CLI + a draft PR on my fork + the GitHub diff view (zero extra development). Only opens an
   upstream PR after approval. A custom review dashboard is a stretch goal (once the core loop is established).

6. **Self-evolution = this project's differentiator.**
   - Taxonomy self-evolution: propose new categories / retire dead topics based on observed PR activity.
   - Self-prediction grading: log predictions like "this will become important" → compare against actual
     outcomes later → update scoring.

7. **Infrastructure & LLM providers.**
   - Agent runs on **`ce-master`** (CPU) under **tmux**. **`mi250-051` / `mi250-052` / `mi250-053`** (MI250,
     gfx90a) are used only for vLLM build / bug reproduction / patch verification (the M3 oracle) and optionally
     hosting a local LLM. Reached over ssh via those aliases.
   - LLM backend is **pluggable** via env `LLM_PROVIDER`: `claude_cli` (`claude -p`, default) · `claude_api`
     (Anthropic API) · `local` (OpenAI-compatible **vLLM** server). One wrapper `src/llm.py`, provider-agnostic
     agents. Using a local vLLM server as the provider dogfoods the very project target.

8. **Deliberately left out.**
   - MCP/A2A components: not forced in (to avoid diluting the core loop). Maybe one lightweight custom MCP
     server if there's spare time.
   - "4-week deadline" framing: doesn't fit the always-on, self-evolving concept, so milestones (M0-M4, ordered)
     are used instead.

## Progress
- **M0 (bootstrapping)** — repo skeleton + minimal GitHub collector (`src/collector.py`) implemented.
- Next: baseline weekly summary from collected data → M1 evolution loop.

## Open questions / decisions for later
- Primary datastore: planned move from JSONL → a document DB (Firebase/Firestore) for collected items + state
  (see PLAN.md "Storage evolution"). The embedding/vector store is a separate concern (pgvector vs LanceDB) —
  decide whether Firestore holds documents while a vector store holds embeddings, or consolidate.
- Per-agent prompt structure / templates (the LLM *backend* is decided — pluggable `claude -p` / Claude API /
  local vLLM; see decision 7).
- Criteria for "became important" in the retrospective benchmark (merged / shipped in a release / adopted by
  other repos).
- Schedule cadence (daily collection / weekly report).
- Dashboard (M5) stack: how to serve the observability UI (e.g., Streamlit/FastAPI+static, or a small React app)
  and how it reads stage output (directly from `data/*.jsonl` + state, or via a small API layer). Chart/diagram
  library for the trend + engine parity visualizations.

## How to resume
1. Read this file (why) + `docs/PLAN.md` (architecture), then **`docs/DEVPLAN.md`** — the resumable
   Milestone → to-do checklist. It is the source of truth for *what to build next*.
2. Run the test suite: `.venv/bin/python -m pytest` (must be green). Follow the Quick Start in `README.md`
   to run the collector once → check `data/*.jsonl`.
3. In `docs/DEVPLAN.md`, find the first unchecked `[ ]` and continue. Rule: a box is checked only when its
   named test passes.
