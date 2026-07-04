#!/usr/bin/env bash
#
# notify.sh — shared ntfy phone-push helper for the dev loop (best-effort; silent no-op
# if NOTIFY_URL isn't configured). Extracted out of wait-merge.sh so every dev-loop
# event — merge-wait, milestone-complete stop, blocked/ambiguous stop — sends through
# one notify path instead of duplicating the NOTIFY_URL-resolution + curl logic per caller.
#
# Usage:
#   scripts/notify.sh <message> [title] [click-url]
# Env:
#   NOTIFY_URL   ntfy topic URL, e.g. https://ntfy.sh/<secret-topic>. Read from the
#                environment, or (if unset) from NOTIFY_URL= in ./.env. Leave unset to
#                disable notifications entirely — this script becomes a silent no-op.
#                Keep the topic secret — anyone who knows the URL can read/post to it.
#
set -euo pipefail

notify_url="${NOTIFY_URL:-}"
if [ -z "$notify_url" ] && [ -f .env ]; then
  notify_url="$(grep -E '^NOTIFY_URL=' .env | tail -1 | cut -d= -f2- || true)"
fi

message="${1:-}"
[ -n "$notify_url" ] || exit 0
[ -n "$message" ] || exit 0

args=(-s --max-time 10 -H "Title: ${2:-vllm-forager}")
[ -n "${3:-}" ] && args+=(-H "Click: $3")
curl "${args[@]}" -d "$message" "$notify_url" >/dev/null 2>&1 || true
