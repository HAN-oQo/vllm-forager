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
#   NOTIFY_URL      ntfy topic URL for phone push, e.g. https://ntfy.sh/<secret-topic>.
#                   Read from the environment, or (if unset) from NOTIFY_URL= in ./.env.
#                   Leave unset to disable notifications. Keep the topic secret —
#                   anyone who knows the URL can read/post to it.
#
set -euo pipefail

interval="${POLL_INTERVAL:-60}"
remind_every="${REMIND_EVERY:-1800}"
branch="$(git rev-parse --abbrev-ref HEAD)"
pr="${1:-$(gh pr view --json number --jq .number)}"

# Resolve NOTIFY_URL from the env, falling back to ./.env (gitignored, per-machine).
notify_url="${NOTIFY_URL:-}"
if [ -z "$notify_url" ] && [ -f .env ]; then
  notify_url="$(grep -E '^NOTIFY_URL=' .env | tail -1 | cut -d= -f2- || true)"
fi

# notify <message> [title] [click-url] — best-effort ntfy push; never breaks the loop.
notify() {
  [ -n "$notify_url" ] || return 0
  local args=(-s --max-time 10 -H "Title: ${2:-vllm-forager}")
  [ -n "${3:-}" ] && args+=(-H "Click: $3")
  curl "${args[@]}" -d "$1" "$notify_url" >/dev/null 2>&1 || true
}

# One metadata fetch: PR url + title (tab-separated); tolerate gh failure.
meta="$(gh pr view "$pr" --json url,title --jq '.url + "\t" + .title' 2>/dev/null || printf '\t')"
url="${meta%%$'\t'*}"
title="${meta#*$'\t'}"

echo "Polling PR #${pr} every ${interval}s until merged (Ctrl-C to stop)…"
notify "PR #${pr} needs your merge: ${title}" "vllm-forager:review" "$url"

waited=0
while true; do
  state="$(gh pr view "$pr" --json state --jq .state 2>/dev/null || echo UNKNOWN)"
  case "$state" in
    MERGED)
      echo "PR #${pr} merged."
      notify "PR #${pr} merged ✓ — dev-loop continuing." "vllm-forager:merged" "$url"
      break ;;
    CLOSED)
      echo "PR #${pr} was closed without merging — stopping."
      notify "PR #${pr} closed unmerged — dev-loop stopped." "vllm-forager:closed" "$url"
      exit 1 ;;
    *)
      sleep "$interval"
      waited=$((waited + interval))
      if [ "$waited" -ge "$remind_every" ]; then
        notify "PR #${pr} still waiting for merge: ${title}" "vllm-forager:reminder" "$url"
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
