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
├── requirements.txt
├── .env.example
├── docs/
│   ├── PLAN.md        # Project plan (milestones M0-M4)
│   └── CONTEXT.md     # Design decision log — context for resuming work
├── src/
│   ├── __init__.py
│   ├── config.py      # Tracked repos and path configuration
│   └── collector.py   # GitHub issue/PR incremental collector (M0)
└── data/              # Collection output (gitignored)
```

## Next steps

Follow the milestone order in `docs/PLAN.md`. M0 = collection + baseline summary comes first.
See `docs/CONTEXT.md` for design background and decisions made so far.
