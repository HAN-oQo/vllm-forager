# vllm-forager

> autonomous vLLM/ROCm contributor — continuously forages for contribution opportunities, fixes them, and gives back.

An **autonomous agent that continuously advances vLLM on ROCm (MI250)**.
It continuously tracks issues and PRs across the inference-serving ecosystem (vLLM · SGLang · NVIDIA Dynamo · llm-d)
to map out the direction of the inference market and its key techniques, **grades its own predictions to evolve its
tracking criteria**, and uses those signals to run an always-on, self-improving loop that goes from
**discovering vLLM (ROCm) contribution candidates → generating and testing patches → human review → PR**.

> Status: **M0 (bootstrapping)** — collector skeleton stage.

## Why ROCm

Available hardware is 3x MI250 (AMD Instinct, ROCm). vLLM's ROCm path is less mature than CUDA, so it has more
unresolved gaps and bugs, and many issues are left unaddressed because there's no AMD hardware to reproduce them on
→ **having real hardware is both the barrier to entry and the edge.**

- Agent runtime: **CPU is enough** (no GPU needed).
- MI250 is used **only for building/testing vLLM and reproducing/verifying ROCm bugs.**

## Quick start

```bash
# 1) Virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Configure GitHub token (avoid rate limits + access private repos)
cp .env.example .env
# open .env and fill in GITHUB_TOKEN (PAT with repo, read access)

# 3) Run the minimal collector once → generates data/*.jsonl
python -m src.collector

# 4) Incremental collection (only items updated since last run, based on state)
python -m src.collector
```

## Repo structure

```
vllm-forager/
├── README.md
├── CLAUDE.md          # Auto-loaded context for every Claude session (rules + pointers)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml     # black / ruff / mypy config
├── .pre-commit-config.yaml
├── pytest.ini         # pytest config + milestone markers (m0…m5)
├── .env.example
├── .claude/
│   └── settings.json  # Shared permissions allowlist (applies to every session)
├── .github/
│   ├── workflows/ci.yml          # CI merge gate: pre-commit + pytest
│   └── PULL_REQUEST_TEMPLATE.md  # PR checklist (test green · pre-commit · /code-review)
├── docs/
│   ├── PLAN.md        # Project plan + architecture (milestones M0-M5)
│   ├── DEVPLAN.md     # Resumable Milestone → to-do checklist (each todo has a test)
│   ├── CONTEXT.md     # Design decision log
│   ├── SURVEY_RSI.md  # Prior-art survey: recursive self-improvement / self-evolving agents
│   ├── IDEAS.md       # Idea backlog → next-version roadmap
│   ├── RUNBOOK.md     # How to operate the dev loop (tmux; PR → human merge → poll)
│   └── CONTRIBUTING.md # Git & code-review rules (PR=one todo; /code-review --comment; human merge)
├── src/
│   ├── config.py      # Tracked repos + paths + 24h collection cadence
│   └── collector.py   # GitHub issue/PR incremental collector (M0)
├── scripts/
│   ├── wait-merge.sh  # Poll a PR until merged, then sync main (dev-loop primitive)
│   ├── collect.sh     # Run the collector + record health (cron entrypoint)
│   └── triage.sh      # On failure: claude -p → fix PR (human-gated self-heal)
├── tests/             # pytest suite (offline/deterministic)
└── data/              # Collection output (gitignored)
```

## Next steps

Work the checklist in **`docs/DEVPLAN.md`** — it's the resumable Milestone → to-do list, and every todo names a
test. Find the first unchecked box and continue; a box is checked only when its test passes.

```bash
pip install -r requirements-dev.txt
python -m pytest        # offline suite — keep it green
```

See `docs/PLAN.md` for the architecture and `docs/CONTEXT.md` for the reasoning behind decisions.

## Running the agents (how to start the workflow)

The agents run on **ce-master** and are **human-gated** — they open PRs; **you merge**. Your **Mac runs no
agent**; it's the control plane (review + merge PRs on GitHub).

**Autonomous developer** — builds the DEVPLAN, one todo per PR:

```bash
ssh ce-master
tmux new -s dev                          # persistent; survives disconnects
cd vllm-forager && source .venv/bin/activate
claude                                    # then: Shift+Tab (Accept-Edits) → /dev-loop
```

`/dev-loop` loops: first unchecked DEVPLAN todo → branch → implement + test → `pytest`/`pre-commit` green → open
PR → `/code-review --comment` → poll until **you** merge → next todo, pausing at each milestone boundary. It never
self-merges. (Omit `/dev-loop` to drive a manual, step-by-step session.)

**Data collector** — runs the product on a schedule. It also does `git checkout`/branches, so give it its **own
clone** (never collides with `/dev-loop`) and point both clones at one shared data dir:

```bash
# on ce-master, one-time
mkdir -p ~/forager-data
gh repo clone HAN-oQo/vllm-forager ~/vllm-forager-collect
cd ~/vllm-forager-collect
python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
cp ~/vllm-forager/.env .env
echo "FORAGER_DATA_DIR=$HOME/forager-data" >> .env                 # share data
echo "FORAGER_DATA_DIR=$HOME/forager-data" >> ~/vllm-forager/.env  # dev clone too
```

Then run the collect agent in that clone:

```bash
cd ~/vllm-forager-collect && tmux new -s collect
claude                                    # then: Shift+Tab (Accept-Edits) → /loop 6h /collect-loop
```

`/collect-loop` each cycle: collect → report counts, or on failure diagnose (transient → wait; real bug → fix PR,
never merges). **Separate clones = no git collision; shared `FORAGER_DATA_DIR` = shared data.** (`docs/RUNBOOK.md`
also covers a simpler cron + `scripts/collect.sh` fallback.)

Watch or steer either: `ssh ce-master && tmux attach -t dev`. Full operator guide: **`docs/RUNBOOK.md`**.
