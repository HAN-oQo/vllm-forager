# CLAUDE.md — vllm-forager

Context for any Claude session or subagent working in this repo. **Read this first.**

## What this is
An autonomous, **human-gated** agent that continuously advances **vLLM on ROCm (MI250)**: it tracks issues/PRs
across the inference-serving ecosystem (vLLM · ROCm/vllm · SGLang · Dynamo · llm-d), evolves its own tracking
criteria by grading its predictions, and turns those signals into vLLM (ROCm) contribution candidates → tested
patches → **human review → PR**.
Status: **M0 (bootstrapping)** — collector implemented + tested; everything else is planned in `docs/DEVPLAN.md`.

## Start here (in order)
1. **`docs/DEVPLAN.md`** — the resumable Milestone→to-do checklist. *This is what to build next.*
2. `docs/CONTEXT.md` — decisions and the reasoning behind them.
3. `docs/PLAN.md` — architecture + roadmap.
4. `docs/SURVEY_RSI.md` — prior art (recursive self-improvement / self-evolving agents).

## Golden workflow (non-negotiable)
- Work `DEVPLAN.md` top-to-bottom; do the **first unchecked `[ ]`**. Read its milestone's Expected output / Demo /
  Acceptance block first.
- Per todo: write code (docstrings + comments + type hints) → write/adjust its **named test** → run its **Demo**
  command to watch it actually work → `pytest` + `pre-commit` green → check the box → commit.
- **A box is checked ONLY when its named test passes and `pre-commit` is clean.** Partial work stays `[ ]`.
- **Evidence principle:** every KB record and every report claim carries a source issue/PR link.
- **Human review gate is mandatory:** nothing goes upstream to vLLM without explicit human approval.

## Commands
```bash
source .venv/bin/activate                 # activate! tools + hooks live here
pip install -r requirements-dev.txt
pre-commit install                        # once
pytest                                    # full offline suite (keep green)
pytest -m m0                              # one milestone (markers: m0, m0_6, m1 … m5)
pre-commit run --all-files                # black + ruff + mypy + hygiene (venv must be active)
python -m src.collector                   # run the collector once → data/*.jsonl
```

## Architecture (one paragraph)
Three planes over one knowledge base (Firestore, planned): **data plane** (collect→normalize→embed, no LLM),
**intelligence plane** (classify / forecast / cited report, scheduled LLM), **contribution plane**
(reproduce→patch→**verify on MI250**→ensemble self-review→human gate→draft PR). A **versioned policy/taxonomy** is
the learning state; the **MI250 is the empirical verification oracle**. Details in `docs/PLAN.md`.

## Guardrails — must stay green (trust the numbers)
- **Collection data-quality** (T0.10–T0.11): robust fetch + reconciliation vs GitHub totals + issue-number gap
  scan → `data_quality` metric.
- **RAG trust score** (T1.8): labeled golden set + Recall@k / MRR / nDCG + faithfulness / hallucination-rate,
  thresholded and drift-tracked.
- **Liveness** (T4.5) + **live health panel** (T5.8): every stage emits `started→heartbeat→finished/failed` with
  its current step + intermediate output, so the dashboard always shows *what is running now, whether it's alive
  or stalled, and its partial output.* A crash must never leave a stage silently "running".

## Conventions
- Python: **black + ruff + mypy**, line length **100** (`pyproject.toml`); code in `src/`, agents in
  `src/agents/`, tests in `tests/test_<module>.py` tagged with a milestone marker (`pytestmark = pytest.mark.m1`),
  web UI in `dashboard/`.
- Tests are **offline/deterministic by default**; anything hitting live GitHub / Firestore emulator / MI250 / a
  live LLM is `@pytest.mark.integration` (skipped in the default run).

## LLM backend — pluggable
Everything calls `src/llm.py` (planned), selected by env `LLM_PROVIDER`: `claude_cli` (`claude -p`, default) ·
`claude_api` · `local` (OpenAI-compatible **vLLM** server). Never hardcode a provider.

## Infrastructure
Agent runs on **`ce-master`** (CPU) under **tmux**. **`mi250-051` / `mi250-052` / `mi250-053`** (gfx90a) = vLLM
build / bug reproduction / patch verification + optional local LLM, over ssh. Collection cadence = **24h**
(`config.COLLECT_INTERVAL_HOURS`).

## Developing with Claude agents
Default to a **single main session** working `DEVPLAN.md`; the checklist + `pytest`/`pre-commit` are the "team"
(manager = the plan, QA = the gates, developer = the session). Use **ephemeral subagents** (not a standing team)
for research fan-out, a reviewer pass on the diff (`/code-review`), and test-failure triage. Reserve full **agent
teams** for M3/M5 when work splits into separate directories — with git worktrees, 3–5 agents, a validation step.
Full rationale + sources: the "Developing with Claude agents" section in `docs/DEVPLAN.md`.
