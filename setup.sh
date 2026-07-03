#!/usr/bin/env bash
# CPU 노드 부트스트랩: 가상환경 + 의존성 + .env 준비.
# 사용: bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/3] venv 생성"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/3] 의존성 설치"
pip install -U pip >/dev/null
pip install -r requirements.txt

echo "[3/3] .env 준비"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  .env 생성됨 — GITHUB_TOKEN 채우세요."
else
  echo "  .env 이미 있음 — 건너뜀."
fi

cat <<'EOF'

완료. 다음:
  source .venv/bin/activate
  # .env 에 GITHUB_TOKEN 입력 후
  python -m src.collector        # data/*.jsonl 수집
EOF
