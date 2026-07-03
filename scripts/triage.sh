#!/usr/bin/env bash
#
# triage.sh — self-heal, HUMAN-GATED. On a collector failure, ask `claude -p` to diagnose
# and, only if it's a code bug, fix it on a triage/* branch + add a test + open a PR.
# It NEVER merges (`.claude/settings.json` denies `gh pr merge`). Best-effort: skips
# cleanly if the toolchain/auth isn't present or the working tree is busy. See RUNBOOK.
#
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
log="${1:-}"

command -v claude >/dev/null 2>&1 || { echo "[triage] claude not on PATH — alert only" >&2; exit 0; }
command -v gh >/dev/null 2>&1 || { echo "[triage] gh not on PATH — alert only" >&2; exit 0; }

# Don't clobber an active dev session: only proceed on a clean working tree.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[triage] working tree dirty (another session may be active) — alert only" >&2
  exit 0
fi

# One triage PR at a time — don't spam on a recurring failure.
open_triage=$(gh pr list --state open --json headRefName \
  --jq '[.[] | select(.headRefName | startswith("triage/"))] | length' 2>/dev/null || echo 0)
if [ "${open_triage:-0}" != "0" ]; then
  echo "[triage] an open triage/* PR already exists — skipping" >&2
  exit 0
fi

[ -f .venv/bin/activate ] && source .venv/bin/activate
git checkout main >/dev/null 2>&1 && git pull --ff-only >/dev/null 2>&1

logtail="$(tail -n 60 "$log" 2>/dev/null)"
prompt="The scheduled collector (python -m src.collector) just failed on ce-master. Log tail:
---
$logtail
---
Diagnose the root cause. If it is a code bug in THIS repo: create a branch named
triage/collector-fix, fix it, add or adjust a test that covers it, get pytest and
pre-commit green, commit, and open a PR with gh (title prefixed 'fix(collector): ').
DO NOT merge the PR. If it is NOT a code bug (network, auth, rate limit, GitHub outage),
change nothing and print a one-line explanation. Follow docs/CONTRIBUTING.md and CLAUDE.md."

# acceptEdits so the headless run can edit files; the settings.json allowlist still
# governs commands and denies `gh pr merge`.
claude -p "$prompt" --permission-mode acceptEdits 2>&1 | tail -25
