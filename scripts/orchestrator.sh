#!/usr/bin/env bash
#
# orchestrator.sh — run one orchestrator tick (`python -m src.orchestrator --once`), STREAM
# progress to the terminal AND a log, and notify on failure. Mirrors scripts/collect.sh's
# overall shape (log + notify-on-failure); does not yet mirror its data/last_run.json health
# record or triage.sh self-heal call -- known gaps, see docs/RUNBOOK.md.
#
# Meant to be run under cron (T4.2) or manually inside the `forager` tmux session
# (docs/RUNBOOK.md). Overlap protection (two ticks never run concurrently) lives in
# `src.orchestrator.main()` itself now (`src.locking.run_lock`, T4.3), not here -- T4.2
# originally added a shell-level `flock` in this script, but that only covered invocations
# that went through this wrapper; T4.3 moved it into Python so it covers *every* invocation
# path (this script, a bare `python -m src.orchestrator --once`, a future caller) and is
# exercised by an offline pytest test (`tests/test_locking.py`) instead of only a manual demo.
# A backed-off tick still exits 0 -- see main()'s own `LockHeld` handling.
#
# Never uses `set -e` so a failure is still logged/notified before we exit.
#
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

mkdir -p data/logs
ts="$(date -u +%Y%m%dT%H%M%SZ)"
log="data/logs/orchestrator-$ts.log"

[ -f .venv/bin/activate ] && source .venv/bin/activate

# -u = unbuffered so progress streams live; tee shows it on the terminal AND writes the log.
python -u -m src.orchestrator --once 2>&1 | tee "$log"
code=${PIPESTATUS[0]}

if [ "$code" -eq 0 ]; then
  # A backed-off tick (another tick already holding the lock, T4.3) also exits 0 -- it isn't a
  # failure, but it isn't "ran" either; distinguish the two so this line doesn't claim a tick
  # completed when zero stages actually ran.
  if grep -q -- "-- skipping this tick" "$log"; then
    echo "[orchestrator] tick skipped (lock already held) — log: $log"
  else
    echo "[orchestrator] tick ok — log: $log"
  fi
else
  # Keep the outbound push generic (no raw log content) -- collect.sh's own failure ping is a
  # fixed string too; log details stay local (data/logs/), not forwarded to a third-party
  # ntfy endpoint.
  bash scripts/notify.sh "orchestrator tick FAILED (exit ${code}) — see $log" \
    "vllm-forager:orchestrator" || true
  echo "[orchestrator] tick FAILED exit=$code log=$log" >&2
fi
exit "$code"
