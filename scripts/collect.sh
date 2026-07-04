#!/usr/bin/env bash
#
# collect.sh — run the collector once, STREAM progress to the terminal AND a log, record
# health (status + record counts) to data/last_run.json, and on failure kick off self-heal
# triage. Works under cron, and `./scripts/collect.sh` also shows live progress. Never uses
# `set -e` so a failure is still recorded before we exit.
#
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

mkdir -p data/logs
ts="$(date -u +%Y%m%dT%H%M%SZ)"
log="data/logs/collect-$ts.log"

[ -f .venv/bin/activate ] && source .venv/bin/activate

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# ntfy: a start ping (best-effort; no-op if NOTIFY_URL isn't configured — see scripts/notify.sh).
bash scripts/notify.sh "collect started" "vllm-forager:collect" || true
# -u = unbuffered so progress streams live; tee shows it on the terminal AND writes the log.
python -u -m src.collector 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
status=$([ "$code" -eq 0 ] && echo ok || echo error)

# Count records currently on disk (per file + total) for the health record. Delegates to
# src.stats.summarize(), which resolves config.DATA_DIR (respects FORAGER_DATA_DIR — a
# hardcoded "data/*.jsonl" glob here would silently count 0 records whenever the shared
# data dir is configured, since the actual files live outside ./data).
counts=$(python - <<'PY'
import json
from src.stats import summarize

s = summarize()
out = {name: r["total"] for name, r in s["repos"].items()}
out["_total"] = s["total"]
print(json.dumps(out))
PY
)
total=$(printf '%s' "$counts" | python -c 'import json,sys; print(json.load(sys.stdin)["_total"])')

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
  "total_records": $total,
  "counts": $counts,
  "error_tail": $err
}
JSON

echo "[collect] status=$status exit=$code total_records=$total log=$log"
# ntfy: an end ping (best-effort; no-op if NOTIFY_URL isn't configured).
if [ "$code" -eq 0 ]; then
  bash scripts/notify.sh "collect done: ok · ${total} records" "vllm-forager:collect" || true
else
  bash scripts/notify.sh "collect FAILED (exit ${code}) — triage attempted" "vllm-forager:collect" || true
  echo "[collect] failure recorded; attempting self-heal triage" >&2
  bash scripts/triage.sh "$log" || echo "[collect] triage unavailable/skipped" >&2
fi
exit "$code"
