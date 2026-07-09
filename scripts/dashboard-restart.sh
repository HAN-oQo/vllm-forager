#!/usr/bin/env bash
#
# dashboard-restart.sh — (re)launch the dashboard under a supervised tmux session (T5.15), so
# a code change actually reaches the live page instead of silently continuing to serve
# whatever `dashboard/render.py` the running process happened to import at its own last
# startup (Python never hot-reloads it -- see src/version.py's own module docstring for the
# real incident this fixes: a process served the pre-tree flat renderer for two days after
# the tree-view design landed, until someone noticed and manually restarted it).
#
# Meant to run from the dashboard's own STABLE serving clone (not the dev-loop's live working
# tree -- see docs/RUNBOOK.md's identical "give the collector its own clone" convention for
# why: both a dev-loop session and a long-running server checking out `main` at different
# times can't safely share one working tree). Idempotent: killing a session that doesn't
# exist, or one already dead, is not an error.
#
# Usage: scripts/dashboard-restart.sh [--host HOST] [--port PORT] [--data-dir DIR]
# Env:   FORAGER_DASHBOARD_TMUX_SESSION (default: forager-dashboard)
#        FORAGER_DATA_DIR — read by `dashboard/__main__.py` itself (via src.store) when
#          --data-dir isn't passed; set it in the environment/`.env` this session's shell
#          inherits.
#
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# POSIX single-quote escaping (portable across bash/dash/zsh) -- tmux runs the pane's
# shell-command through whatever `default-shell`/$SHELL the pane inherits, which is not
# guaranteed to be bash, so bash's own `printf %q` (which can emit bash-only $'...' ANSI-C
# quoting that a non-bash shell won't parse the same way) isn't safe to embed there.
_shq() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

host="127.0.0.1"
port="8765"
data_dir_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --host) host="${2:?dashboard-restart: --host requires a value}"; shift 2 ;;
    --port) port="${2:?dashboard-restart: --port requires a value}"; shift 2 ;;
    --data-dir)
      data_dir_args=(--data-dir "${2:?dashboard-restart: --data-dir requires a value}")
      shift 2
      ;;
    *) echo "dashboard-restart: unknown argument: $1" >&2; exit 2 ;;
  esac
done

session="${FORAGER_DASHBOARD_TMUX_SESSION:-forager-dashboard}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "dashboard-restart: tmux not found on PATH" >&2
  exit 1
fi

mkdir -p data/logs || { echo "dashboard-restart: failed to create data/logs" >&2; exit 1; }
ts="$(date -u +%Y%m%dT%H%M%SZ)"
# `-$$` (this script's own PID) so two restarts landing in the same UTC second don't
# collide on one log path and have the second `tee` truncate the first restart's log.
log="data/logs/dashboard-$ts-$$.log"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "[dashboard-restart] killing existing session: $session"
  tmux kill-session -t "$session"
fi

venv_cmd=""
[ -f .venv/bin/activate ] && venv_cmd="source .venv/bin/activate && "

dashboard_cmd="python -m dashboard --host $(_shq "$host") --port $(_shq "$port")"
if [ "${#data_dir_args[@]}" -gt 0 ]; then
  dashboard_cmd="$dashboard_cmd --data-dir $(_shq "${data_dir_args[1]}")"
fi

# `2>&1 | tee` inside the tmux pane itself (not this script's own stdout) -- the pane must
# keep running (and logging) long after this script exits.
tmux new-session -d -s "$session" \
  "${venv_cmd}${dashboard_cmd} 2>&1 | tee $(_shq "$log")"

# tmux new-session -d returns as soon as the session/pane is created, even if the command
# inside it then immediately fails and the pane (and with it, the session) closes -- give it a
# moment before checking whether it's actually still alive.
sleep 1
if tmux has-session -t "$session" 2>/dev/null; then
  echo "[dashboard-restart] relaunched — session=$session host=$host port=$port log=$log"
else
  bash scripts/notify.sh "dashboard-restart FAILED to relaunch — see $log" \
    "vllm-forager:dashboard" || true
  echo "[dashboard-restart] FAILED to relaunch — see $log" >&2
  exit 1
fi
