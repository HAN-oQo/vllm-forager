#!/usr/bin/env bash
# CPU node bootstrap: virtualenv + dependencies + .env setup.
# Usage: bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/3] creating venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/3] installing dependencies"
pip install -U pip >/dev/null
pip install -r requirements.txt

echo "[3/3] preparing .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  .env created — fill in GITHUB_TOKEN."
else
  echo "  .env already exists — skipping."
fi

cat <<'EOF'

Done. Next:
  source .venv/bin/activate
  # after entering GITHUB_TOKEN in .env
  python -m src.collector        # collect data/*.jsonl
EOF
