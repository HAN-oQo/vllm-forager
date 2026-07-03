# RUNBOOK — Operating the dev loop

How this project is built day-to-day: a **session-driven, human-merge-gated** loop.
You manage from your Mac; **Developer Claude** builds on `ce-master`; **MI250** is only for vLLM.
`docs/DEVPLAN.md` = *what* to build; this file = *how*.

## Topology

```
┌─────────────┐   review + MERGE PRs (the human gate)   ┌────────────────────────────┐
│  Your Mac   │ ──────────────────────────────────────▶ │  GitHub (origin)            │
│ control     │ ◀── PR opened + CI status ─────────────  │  PRs · Actions CI · main    │
│ plane (you) │──ssh + tmux attach (observe/steer)─┐     └────────────────────────────┘
└─────────────┘                                    ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  ce-master  (CPU)                                          │
                 │  • Developer Claude   (tmux: dev)     → builds DEVPLAN, PRs│
                 │  • Product agent runtime (tmux: forager) → collector…/M4   │
                 └───────────────┬──────────────────────────────────────────┘
                                 │ ssh: build / reproduce / verify
                                 ▼
                 ┌──────────────────────────────────────────┐
                 │  mi250-051 / 052 / 053 (GPU, gfx90a)       │
                 │  vLLM build + bug repro + patch verify only│
                 └──────────────────────────────────────────┘
```

**Two Claudes on `ce-master` — don't conflate:**
- **Developer Claude** (dev-time): builds DEVPLAN todos → PRs. *This runbook.*
- **Product agent runtime** (run-time): vllm-forager itself (collector / intelligence / M4 orchestrator). Runs
  later, alongside — it *is* the thing Developer Claude builds.

## Roles
- **Mac (you)** = control plane: review + **merge** PRs (the human gate), steer. No heavy compute.
- **ce-master (CPU)**: Developer Claude (tmux `dev`) + product runtime (tmux `forager`).
- **mi250-051/052/053 (GPU)**: vLLM build / repro / verify only, over ssh from ce-master.

## Prerequisites (once, on ce-master)
- `git clone` the repo; `python -m venv .venv && source .venv/bin/activate`; `pip install -r requirements-dev.txt`;
  `pre-commit install`.
- Claude Code installed + authenticated (the `claude_cli` provider / Developer Claude).
- `gh auth login` with scopes `repo`, `workflow`.
- ssh access to `mi250-05x`.

## The loop (session-driven, one session per milestone)

On **ce-master**:
```bash
tmux new -s dev      # persistent; survives ssh disconnects
claude               # boots with CLAUDE.md; tell it which milestone to work
```

For each todo, Developer Claude:
1. picks the **first unchecked `[ ]`** in `docs/DEVPLAN.md`;
2. `git checkout -b t<id>-slug`;
3. implements it + its named test; `pytest` + `pre-commit` green;
4. `gh pr create` — **never merges**;
5. runs `scripts/wait-merge.sh` → polls the PR every 60s; on **your merge** it syncs `main` + prunes the branch;
6. continues to the **next** todo.

At the **milestone boundary** it stops and waits for your explicit go. Start a **fresh `dev` session for the next
milestone** (CLAUDE.md re-boots full context) to avoid long-session drift/cost.

**Autonomy:** run the session in **`acceptEdits` + a curated allowlist** — file edits auto-accepted; safe commands
allowed (`pytest`, `git`, `gh`, `ruff`, `black`, `python -m …`); destructive/outward actions still prompt; and it
**never self-merges**.

## Your side (Mac)
- Review the PR + CI (green) → optional `/code-review` on the diff → **merge**. That merge is what releases the
  poll and lets the loop advance.
- Watch or steer anytime: `ssh ce-master` → `tmux attach -t dev`.

## Continue-after-merge = poll
`scripts/wait-merge.sh [<pr>]` polls the PR until `MERGED`, then `git checkout main && git pull --ff-only` and
prunes the merged branch. Override cadence with `POLL_INTERVAL` (default 60s). If the PR is **closed unmerged**,
it stops (exit 1) instead of advancing.

## Branch protection (set once — GitHub → Settings → Branches → `main`)
- Require the **CI** status check to pass before merging.
- Require a pull request before merging (blocks direct agent pushes).
- Block force-pushes.

This makes the human-in-the-loop merge **structural**, not just convention.
