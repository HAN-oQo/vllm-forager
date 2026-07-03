"""추적 대상 레포와 경로 설정.

레포 slug는 실제 GitHub 경로 기준. 확실치 않은 건 GitHub에서 확인 후 조정할 것.
weight 는 나중에 랭킹/필터 단계에서 쓰기 위한 힌트 (수집 자체는 전부 동일하게 함).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_PATH = DATA_DIR / "state.json"

# role: primary(주 타깃/PR 제출처) / fork(다운스트림 포크) / parity(성능 비교) / radar(트렌드만)
REPOS = [
    {"slug": "vllm-project/vllm", "role": "primary"},
    {"slug": "ROCm/vllm",         "role": "fork"},
    {"slug": "sgl-project/sglang", "role": "parity"},
    {"slug": "ai-dynamo/dynamo",  "role": "radar"},
    {"slug": "llm-d/llm-d",       "role": "radar"},
]

# ROCm 관련 신호로 가점할 라벨/키워드 (M2 랭킹에서 사용 예정)
ROCM_HINTS = ["rocm", "amd", "hip", "mi250", "mi300", "gfx", "hipblas", "instinct"]

# 수집 파라미터
PER_PAGE = 100
# 첫 실행 시 이 날짜 이후만 (너무 오래된 이슈까지 안 긁도록). ISO8601.
INITIAL_SINCE = "2025-01-01T00:00:00Z"
