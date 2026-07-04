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

**Run it as a tracked background task, not `nohup ... & disown`.** A Claude Code session must launch
`wait-merge.sh` via its tool's own background-execution option (e.g. the Bash tool's `run_in_background: true`)
so the harness keeps a handle on the process and can notify the session when it exits. `nohup ... & disown`
detaches the process from the shell's job table — the script still runs and still syncs `main` correctly, but the
session loses any way to learn it happened, and will sit waiting even after you've merged.

## Branch protection (set once — GitHub → Settings → Branches → `main`)
- Require the **CI** status check to pass before merging.
- Require a pull request before merging (blocks direct agent pushes).
- Block force-pushes.

This makes the human-in-the-loop merge **structural**, not just convention.

## Collect loop (agentic — `/collect-loop`)

The **smarter** way to run collection: a Claude session runs one cycle and, on failure, diagnoses in-context and
opens a fix PR — better self-heal than the cron+shell fallback below. Schedule it with the `/loop` skill (no
crontab):

```bash
# on ce-master
tmux new -s collect
cd vllm-forager && source .venv/bin/activate
claude            # then: Shift+Tab (Accept-Edits) → /loop 6h /collect-loop
```

Each cycle `/collect-loop`: pulls main → runs `python -m src.collector` (streamed) → on success reports counts;
on failure classifies (transient → wait for next cycle; real bug → fix on a `triage/*` branch + test + **PR** +
`/code-review`, never merges). You merge; the next cycle picks it up. It runs as a **live session** (keep the tmux
session up). For a truly fresh session per cycle instead, have cron/systemd launch `claude -p "/collect-loop"`
every N hours.

### Running dev-loop and collect-loop together (no collisions + shared data)

Both loops do `git checkout main && git pull` and create branches, so they **can't share one working tree** —
give the collector its **own clone**. But `data/` is **gitignored**, so a separate clone wouldn't see the data;
point every clone at **one shared data dir** with `FORAGER_DATA_DIR`:

```bash
# on ce-master
mkdir -p ~/forager-data                                     # the single shared data dir
gh repo clone HAN-oQo/vllm-forager ~/vllm-forager-collect   # collector's own clone
cd ~/vllm-forager-collect
python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
cp ~/vllm-forager/.env .env
echo "FORAGER_DATA_DIR=$HOME/forager-data" >> .env          # <-- share data
tmux new -s collect                                         # → claude → /loop 6h /collect-loop
```

Set the same `FORAGER_DATA_DIR=$HOME/forager-data` in the dev clone's `.env` too, so both read/write the one
dataset. Branches never overlap (`t*` vs `triage/*`); both push to the same origin.

> Directory-sharing is the stop-gap while data is JSONL. **M0.6 replaces it properly**: once the KB is Firestore,
> the collector writes to a shared store and every clone/process/node reads from it — no shared directory needed.

## Scheduling the collector (run-time, simpler fallback)

*(No live Claude session — cheaper and dead-simple, but self-heal is only best-effort. Prefer `/collect-loop`
above when you want smart recovery.)*

Run the collector on a schedule via **cron** (simple, no sudo). `scripts/collect.sh` runs it, writes health to
`data/last_run.json`, logs to `data/logs/`, and on failure kicks off self-heal triage.

```bash
crontab -e
# daily at 04:07 UTC (deliberately off the hour); adjust to taste
7 4 * * *  cd ~/vllm-forager && scripts/collect.sh >> data/logs/cron.log 2>&1
```

Cadence is 24h (`config.COLLECT_INTERVAL_HOURS`); the collector only fetches items updated since the last run
(`data/state.json`).

> **Ordering matters for the report:** `python -m src.report` (T1.6) groups items by the category the Analyst
> assigned them (`python -m src.analyze`, T1.4) — an item the Analyst hasn't classified yet renders under a plain
> "Other" bucket instead of a real category. If you're scheduling collection, also schedule `python -m
> src.analyze` to run after each collection cycle and before `python -m src.report`, or every report after a
> fresh collection will show mostly-uncategorized items until analyze catches up.

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
