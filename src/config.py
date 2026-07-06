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

# Each entry: slug + role + domain.
#   role:   primary = contribution target (where PRs go) · parity = perf/feature comparison ·
#           source = ecosystem trend/parity source (watched, not contributed to) · radar = trends
#           only.
#   domain: engine · speech · rl · omni — groups a target with the ecosystem watched around it.
# `src.parity.find_gaps` compares any tracked engine (of any role) against a `primary` target --
# there's no dedicated "fork" role anymore (ROCm/vllm retired upstream; T3.14 generalized parity
# off the old fork-vs-upstream premise before this list dropped it).
# Note: collection treats all repos equally; role/domain are hints for the M2 ranking/parity.
# ⚠ Some source repos are large (verl, diffusers, OpenRLHF, slime) — see DEVPLAN for tuning
#   collection per role (shorter window / issues-only) so the KB isn't swamped.
REPOS = [
    # ── contribution targets ──
    {
        "slug": "vllm-project/vllm",
        "role": "primary",
        "domain": "speech",
    },  # upstream; ROCm SPEECH is the current priority
    {
        "slug": "vllm-project/vllm-omni",
        "role": "primary",
        "domain": "omni",
    },  # omni-modality (has a ROCm roadmap)
    {"slug": "vllm-project/vime", "role": "primary", "domain": "rl"},  # RL post-training
    # ── core inference engines (parity / trend) ──
    {
        "slug": "sgl-project/sglang",
        "role": "parity",
        "domain": "engine",
    },  # ROCm parity baseline + omni
    {"slug": "ai-dynamo/dynamo", "role": "radar", "domain": "engine"},
    {"slug": "llm-d/llm-d", "role": "radar", "domain": "engine"},
    # ── RL / post-training ecosystem (watch around vime) ──
    {"slug": "verl-project/verl", "role": "source", "domain": "rl"},
    {"slug": "OpenRLHF/OpenRLHF", "role": "source", "domain": "rl"},
    {"slug": "NVIDIA-NeMo/RL", "role": "source", "domain": "rl"},
    {"slug": "THUDM/slime", "role": "source", "domain": "rl"},
    {"slug": "NovaSky-AI/SkyRL", "role": "source", "domain": "rl"},
    {"slug": "PrimeIntellect-ai/prime-rl", "role": "source", "domain": "rl"},
    # ── omni / multimodal / diffusion-video ecosystem (watch around vllm-omni) ──
    {"slug": "huggingface/diffusers", "role": "source", "domain": "omni"},
    {"slug": "xdit-project/xDiT", "role": "source", "domain": "omni"},
    {"slug": "hao-ai-lab/FastVideo", "role": "source", "domain": "omni"},
    {"slug": "vipshop/cache-dit", "role": "source", "domain": "omni"},
]

# Labels/keywords that boost the ROCm-relevance signal (to be used in M2 ranking)
ROCM_HINTS = ["rocm", "amd", "hip", "mi250", "mi300", "gfx", "hipblas", "instinct"]

# Speech/audio-domain keywords. vLLM's ROCm SPEECH work is the current top priority, so the M2
# ranking boosts vllm-project/vllm items matching SPEECH_HINTS (especially ∩ ROCM_HINTS).
SPEECH_HINTS = [
    "speech",
    "asr",
    "tts",
    "audio",
    "whisper",
    "wav2vec",
    "voice",
    "vocoder",
    "transcrib",
    "phoneme",
    "diarization",
    "mel",
]

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

# Storage backend (T0.6.2): src.store.get_store() reads this to select jsonl vs. firestore —
# named here like every other tunable env var in this file, rather than a bare os.getenv()
# inside store/__init__.py, so `monkeypatch.setattr(config, "STORE_BACKEND", ...)` works the
# same way tests already override every other config value.
STORE_BACKEND = os.getenv("STORE") or "jsonl"
# FIRESTORE_PROJECT is only used by the firestore backend — None lets the client library fall
# back to ambient credentials' default project (GOOGLE_CLOUD_PROJECT, gcloud config, or the
# GCE/Cloud Run metadata server); set it to override, or to point at a local emulator project
# alongside FIRESTORE_EMULATOR_HOST.
FIRESTORE_PROJECT = os.getenv("FIRESTORE_PROJECT") or None
