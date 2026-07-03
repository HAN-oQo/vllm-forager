#!/usr/bin/env bash
#
# wait-merge.sh — continue-after-merge primitive for the session-driven dev loop.
#
# Polls a pull request until it is MERGED, then syncs main and prunes the local
# feature branch. Developer Claude runs this right after `gh pr create`, so the loop
# advances to the next todo only once a human merges. See docs/RUNBOOK.md.
#
# Usage:
#   scripts/wait-merge.sh [<pr-number>]     # defaults to the PR for the current branch
# Env:
#   POLL_INTERVAL   seconds between polls (default 60)
#
set -euo pipefail

interval="${POLL_INTERVAL:-60}"
branch="$(git rev-parse --abbrev-ref HEAD)"
pr="${1:-$(gh pr view --json number --jq .number)}"

echo "Polling PR #${pr} every ${interval}s until merged (Ctrl-C to stop)…"
while true; do
  state="$(gh pr view "$pr" --json state --jq .state 2>/dev/null || echo UNKNOWN)"
  case "$state" in
    MERGED) echo "PR #${pr} merged." ; break ;;
    CLOSED) echo "PR #${pr} was closed without merging — stopping." ; exit 1 ;;
    *)      sleep "$interval" ;;
  esac
done

git checkout main
git pull --ff-only
if [ "$branch" != "main" ]; then
  git branch -d "$branch" 2>/dev/null || echo "note: kept branch '${branch}' (not fully merged locally)"
fi
echo "main synced — pick the next unchecked todo in docs/DEVPLAN.md."
