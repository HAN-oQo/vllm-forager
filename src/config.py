"""Tracked repos and path configuration.

Repo slugs use the actual GitHub paths. Verify any uncertain ones on GitHub and adjust.
`weight` is a hint for the later ranking/filtering stage (collection itself treats all
repos equally).
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # works even if python-dotenv isn't installed
    pass

ROOT = Path(__file__).resolve().parent.parent
# Data lives here. Set FORAGER_DATA_DIR to point every clone/process at ONE shared
# directory, so a separately-cloned collector and the dev checkout share the dataset.
DATA_DIR = Path(os.getenv("FORAGER_DATA_DIR") or (ROOT / "data"))
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
# First-run / --full window: only collect items updated in the last N days (rolling, so it
# stays recent without editing a date). ~6 months keeps the initial backfill sane.
INITIAL_LOOKBACK_DAYS = 180
# Scheduler cadence: how often the data-plane collection runs, in hours.
# 24h (daily) for now; the M4 orchestrator reads this to decide when to re-collect.
COLLECT_INTERVAL_HOURS = 24

# Collector robustness (T0.10)
# Longest issue/PR body we persist, in characters. Capped to keep JSONL small in M0; raise
# via env FORAGER_BODY_MAX_CHARS when the body feeds RAG chunking (M1) and truncation loses
# signal. Malformed env → ValueError at import (a config error should be loud, not silent).
BODY_MAX_CHARS = int(os.getenv("FORAGER_BODY_MAX_CHARS") or 4000)
# Per-request HTTP timeout (seconds) — a hung connection must not stall the whole run.
REQUEST_TIMEOUT_S = 30
# Transient-failure retries (5xx / timeout / connection error) before giving up on a request.
MAX_RETRIES = 4
# Exponential backoff base (seconds): wait before retry N = BACKOFF_BASE_S * 2**N.
BACKOFF_BASE_S = 2.0
# Max rate-limit waits (primary or secondary) before giving up on a request. Bounds a stuck
# limiter — a persistent Retry-After, or a past/stale X-RateLimit-Reset — so it can't spin
# forever; per-repo isolation in main() then skips just that repo instead of hanging the run.
MAX_RATE_LIMIT_RETRIES = 10

# Data-quality guardrail (T0.11): flag a repo's collection when the fraction of missing
# issue/PR numbers in the collected range exceeds this. Deleted/transferred items make small
# gaps normal, so this is a ratio (5%), not a zero-tolerance check.
DATA_QUALITY_GAP_RATIO_THRESHOLD = 0.05
