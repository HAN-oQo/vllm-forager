#!/usr/bin/env bash
#
# orchestrator.sh — run one orchestrator tick (`python -m src.orchestrator --once`), STREAM
# progress to the terminal AND a log, and notify on failure. Mirrors scripts/collect.sh's
# overall shape (log + notify-on-failure); does not yet mirror its data/last_run.json health
# record or triage.sh self-heal call -- known gaps, see docs/RUNBOOK.md.
#
# Meant to be run under cron (T4.2) or manually inside the `forager` tmux session
# (docs/RUNBOOK.md). Guards against two ticks overlapping with a non-blocking flock: T4.3
# (KB-level idempotency/locking) isn't built yet, and the contribution stage in particular has
# no cadence cursor at all (it reruns every tick it's triggered on, cheap-check or not) -- so a
# second tick starting before the first finishes is a real, not hypothetical, risk of
# concurrent JsonlStore state-file writes, not just wasted work.
#
# Never uses `set -e` so a failure is still logged/notified before we exit.
#
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

mkdir -p data/logs
ts="$(date -u +%Y%m%dT%H%M%SZ)"
log="data/logs/orchestrator-$ts.log"
lock="data/orchestrator.lock"

exec 9>"$lock"
if ! flock -n 9; then
  echo "[orchestrator] another tick is already running (lock: $lock) -- skipping" | tee "$log"
  exit 0
fi

[ -f .venv/bin/activate ] && source .venv/bin/activate

# -u = unbuffered so progress streams live; tee shows it on the terminal AND writes the log.
python -u -m src.orchestrator --once 2>&1 | tee "$log"
code=${PIPESTATUS[0]}

if [ "$code" -eq 0 ]; then
  echo "[orchestrator] tick ok — log: $log"
else
  # Keep the outbound push generic (no raw log content) -- collect.sh's own failure ping is a
  # fixed string too; log details stay local (data/logs/), not forwarded to a third-party
  # ntfy endpoint.
  bash scripts/notify.sh "orchestrator tick FAILED (exit ${code}) — see $log" \
    "vllm-forager:orchestrator" || true
  echo "[orchestrator] tick FAILED exit=$code log=$log" >&2
fi
exit "$code"
