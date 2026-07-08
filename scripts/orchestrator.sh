#!/usr/bin/env bash
#
# orchestrator.sh — run one orchestrator tick (`python -m src.orchestrator --once`), STREAM
# progress to the terminal AND a log, and notify on failure. Mirrors scripts/collect.sh's
# shape. Meant to be run under cron (T4.2) or manually inside the `forager` tmux session
# (docs/RUNBOOK.md) -- each invocation is one tick; the orchestrator itself decides per-stage
# (collect/intel/contribution) whether there's anything due/triggered this tick, so cron can
# fire this more often than any one stage's own cadence without over-running anything.
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
  echo "[orchestrator] tick ok — log: $log"
else
  tail_msg=$(tail -n 5 "$log" | tr '\n' ' ')
  bash scripts/notify.sh "orchestrator tick FAILED (exit ${code}): ${tail_msg}" \
    "vllm-forager:orchestrator" || true
  echo "[orchestrator] tick FAILED exit=$code log=$log" >&2
fi
exit "$code"
