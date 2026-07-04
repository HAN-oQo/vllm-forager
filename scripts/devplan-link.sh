#!/usr/bin/env bash
#
# devplan-link.sh — print a GitHub permalink to a DEVPLAN todo, pinned to a commit SHA +
# line (survives later edits). Paste it in the PR body so each PR traces to its todo.
#
# Usage: scripts/devplan-link.sh T0.6 [git-ref]      # ref defaults to origin/main
#
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
todo="${1:?usage: devplan-link.sh T<id> [git-ref]}"
ref="${2:-origin/main}"

sha="$(git rev-parse "$ref")"
line="$(grep -n "\*\*${todo} " docs/DEVPLAN.md | head -1 | cut -d: -f1)"
[ -n "$line" ] || { echo "!! '${todo}' not found in docs/DEVPLAN.md" >&2; exit 1; }
repo="$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || echo HAN-oQo/vllm-forager)"

echo "https://github.com/${repo}/blob/${sha}/docs/DEVPLAN.md#L${line}"
