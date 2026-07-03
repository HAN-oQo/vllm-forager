#!/usr/bin/env bash
#
# collect.sh — run the collector once, record health, and on failure kick off self-heal
# triage. Designed for cron on ce-master (see docs/RUNBOOK.md). Never uses `set -e` so a
# collector failure is still recorded before we exit.
#
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

mkdir -p data/logs
ts="$(date -u +%Y%m%dT%H%M%SZ)"
log="data/logs/collect-$ts.log"

[ -f .venv/bin/activate ] && source .venv/bin/activate

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -m src.collector >"$log" 2>&1
code=$?
finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
status=$([ "$code" -eq 0 ] && echo ok || echo error)

# JSON-encode the last lines of the log when we failed, else an empty string.
if [ "$code" -ne 0 ]; then
  err=$(tail -n 20 "$log" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
else
  err='""'
fi

cat > data/last_run.json <<JSON
{
  "started_at": "$started",
  "finished_at": "$finished",
  "exit_code": $code,
  "status": "$status",
  "log": "$log",
  "error_tail": $err
}
JSON

echo "[collect] status=$status exit=$code log=$log"
if [ "$code" -ne 0 ]; then
  echo "[collect] failure recorded; attempting self-heal triage" >&2
  bash scripts/triage.sh "$log" || echo "[collect] triage unavailable/skipped" >&2
fi
exit "$code"
