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

## Autonomous mode (`/dev-loop`)

To have the Developer Claude work the checklist hands-off: in the tmux `dev` session, switch to **Accept-Edits**
(Shift+Tab) and run **`/dev-loop`**. Each iteration it: picks the first unchecked todo → branch → implement +
test → `pytest`/`pre-commit` green → `gh pr create` → **`/code-review --comment`** (review posted on the PR) →
`scripts/wait-merge.sh` (polls until **you** merge) → next todo. It **pauses at the milestone boundary** for your
go, **STOPs and asks** on anything ambiguous, and **never self-merges** (`gh pr merge` is denied). Stop it anytime
with Esc; start a fresh session per milestone to keep context clean.

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

## Scheduling the collector (run-time)

Run the collector on a schedule via **cron** (simple, no sudo). `scripts/collect.sh` runs it, writes health to
`data/last_run.json`, logs to `data/logs/`, and on failure kicks off self-heal triage.

```bash
crontab -e
# daily at 04:07 UTC (deliberately off the hour); adjust to taste
7 4 * * *  cd ~/vllm-forager && scripts/collect.sh >> data/logs/cron.log 2>&1
```

Cadence is 24h (`config.COLLECT_INTERVAL_HOURS`); the collector only fetches items updated since the last run
(`data/state.json`).

**Progress & health:** running `./scripts/collect.sh` **streams progress live** to the terminal (via `tee`; the
collector prints `… <repo>: +N (M so far)` per page) while also writing `data/logs/`. `data/last_run.json` holds
the last run's `status`, `exit_code`, timestamps, **record counts** (per repo + `total_records`), and (on failure)
an `error_tail` — so `cat data/last_run.json` tells you if the last run was healthy, recent, and how much data you
have. `python -m src.stats` prints a per-repo issue/PR (+ ROCm-tagged) breakdown any time. "Alert" today =
`status:"error"` + a non-zero exit in `data/logs/cron.log`.

**Self-heal (alert + fix PR):** on failure `collect.sh` calls `scripts/triage.sh`, which — if `claude` and `gh`
are on PATH **and the working tree is clean** — asks `claude -p` to diagnose and, **only for a code bug**, fix it
on a `triage/*` branch, add a test, and **open a PR** (never merges; one open triage PR at a time). You review +
merge like any PR. Network / auth / rate-limit failures change nothing — just an explanation.

> Run the scheduler from a **separate clone/worktree** than an active `/dev-loop` session so triage's git
> operations can't collide with in-progress dev work. `claude -p` needs `claude`+`gh` authenticated for the cron
> user; otherwise triage degrades to alert-only (the scheduled run + health still work).
