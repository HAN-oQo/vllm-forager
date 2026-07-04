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
#   REMIND_EVERY    seconds between "still waiting" re-pings (default 1800 = 30 min)
#   NOTIFY_URL      ntfy topic URL for phone push — see scripts/notify.sh for details.
#
set -euo pipefail

interval="${POLL_INTERVAL:-60}"
remind_every="${REMIND_EVERY:-1800}"
branch="$(git rev-parse --abbrev-ref HEAD)"
pr="${1:-$(gh pr view --json number --jq .number)}"

# Resolve alongside this script (not cwd) so wait-merge.sh works regardless of where it's
# invoked from; notify.sh itself still reads ./.env relative to the *caller's* cwd, matching
# how NOTIFY_URL / GITHUB_TOKEN are already resolved elsewhere in this repo's tooling.
notify_sh="$(dirname "${BASH_SOURCE[0]}")/notify.sh"

# One metadata fetch: PR url + title (tab-separated); tolerate gh failure.
meta="$(gh pr view "$pr" --json url,title --jq '.url + "\t" + .title' 2>/dev/null || printf '\t')"
url="${meta%%$'\t'*}"
title="${meta#*$'\t'}"

echo "Polling PR #${pr} every ${interval}s until merged (Ctrl-C to stop)…"
"$notify_sh" "PR #${pr} needs your merge: ${title}" "vllm-forager:review" "$url"

waited=0
while true; do
  state="$(gh pr view "$pr" --json state --jq .state 2>/dev/null || echo UNKNOWN)"
  case "$state" in
    MERGED)
      echo "PR #${pr} merged."
      "$notify_sh" "PR #${pr} merged ✓ — dev-loop continuing." "vllm-forager:merged" "$url"
      break ;;
    CLOSED)
      echo "PR #${pr} was closed without merging — stopping."
      "$notify_sh" "PR #${pr} closed unmerged — dev-loop stopped." "vllm-forager:closed" "$url"
      exit 1 ;;
    *)
      sleep "$interval"
      waited=$((waited + interval))
      if [ "$waited" -ge "$remind_every" ]; then
        "$notify_sh" "PR #${pr} still waiting for merge: ${title}" "vllm-forager:reminder" "$url"
        waited=0
      fi ;;
  esac
done

git checkout main
git pull --ff-only
if [ "$branch" != "main" ]; then
  git branch -d "$branch" 2>/dev/null || echo "note: kept branch '${branch}' (not fully merged locally)"
fi
echo "main synced — pick the next unchecked todo in docs/DEVPLAN.md."
