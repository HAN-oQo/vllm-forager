#!/usr/bin/env bash
#
# dashboard-deploy.sh — the actual "restart it as part of deploy / `git pull`" T5.15 asks for:
# pull `main` in the dashboard's own stable serving clone, then restart the supervised session
# (scripts/dashboard-restart.sh) so a merged code change reaches the live page. Run this
# manually after a merge, from cron, or wire it into a future deploy hook / the M4
# orchestrator once it grows one (DEVPLAN's own "a deploy hook, or the M4 orchestrator after
# it pulls" -- either integration point calls this same script; neither exists yet, so this
# is deliberately still a standalone, human/cron-invokable script rather than wired into
# orchestrator.py itself, which has no deploy-time hook to attach to today).
#
# Refuses to deploy over a dirty working tree (mirrors scripts/triage.sh's own identical
# guard) -- a stable serving clone should never carry uncommitted local changes, and pulling
# over one could silently discard them or fail outright.
#
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[dashboard-deploy] working tree dirty -- refusing to pull; resolve by hand first" >&2
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" != "main" ]; then
  echo "[dashboard-deploy] on branch '$branch', not 'main' -- refusing to deploy a non-main checkout" >&2
  exit 1
fi

before="$(git rev-parse HEAD)"
if ! git pull --ff-only; then
  echo "[dashboard-deploy] git pull --ff-only failed -- not restarting the stale process" >&2
  bash scripts/notify.sh "dashboard-deploy: git pull failed, dashboard NOT restarted" \
    "vllm-forager:dashboard" || true
  exit 1
fi
after="$(git rev-parse HEAD)"

if [ "$before" = "$after" ]; then
  echo "[dashboard-deploy] already up to date ($after) -- nothing to restart"
  exit 0
fi

echo "[dashboard-deploy] pulled $before -> $after -- restarting the dashboard"
if bash scripts/dashboard-restart.sh "$@"; then
  bash scripts/notify.sh "dashboard-deploy: restarted at $after" "vllm-forager:dashboard" || true
else
  bash scripts/notify.sh "dashboard-deploy: pulled to $after but restart FAILED" \
    "vllm-forager:dashboard" || true
  exit 1
fi
