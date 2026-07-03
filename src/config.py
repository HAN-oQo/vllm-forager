"""Tracked repos and path configuration.

Repo slugs use the actual GitHub paths. Verify any uncertain ones on GitHub and adjust.
`weight` is a hint for the later ranking/filtering stage (collection itself treats all
repos equally).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_PATH = DATA_DIR / "state.json"

# role: primary = main target / where PRs are submitted; fork = downstream fork;
#       parity = performance comparison; radar = trends only.
REPOS = [
    {"slug": "vllm-project/vllm", "role": "primary"},
    {"slug": "ROCm/vllm", "role": "fork"},
    {"slug": "sgl-project/sglang", "role": "parity"},
    {"slug": "ai-dynamo/dynamo", "role": "radar"},
    {"slug": "llm-d/llm-d", "role": "radar"},
]

# Labels/keywords that boost the ROCm-relevance signal (to be used in M2 ranking)
ROCM_HINTS = ["rocm", "amd", "hip", "mi250", "mi300", "gfx", "hipblas", "instinct"]

# Collection parameters
PER_PAGE = 100
# On the first run, only collect items updated after this date (avoid pulling very old
# issues). ISO8601.
INITIAL_SINCE = "2025-01-01T00:00:00Z"
# Scheduler cadence: how often the data-plane collection runs, in hours.
# 24h (daily) for now; the M4 orchestrator reads this to decide when to re-collect.
COLLECT_INTERVAL_HOURS = 24
