"""Baseline weekly report v0 — fixed taxonomy, no LLM (T0.8).

The first human-readable deliverable: turn collected KB items (issues/PRs) into a weekly
Markdown digest. Every item is bucketed into a **fixed** taxonomy by keyword match (no LLM
yet — the LLM-classified report is T1.6), and **every line cites its source issue/PR link**,
setting the "every claim carries a source link" bar (CLAUDE.md evidence principle) before any
model is involved.

Design:
- :data:`TAXONOMY` is an ordered list of ``(category, keyword-regex)``. Each item is assigned
  to the **first** category whose keywords match its title/labels/body (single bucket, so the
  per-section counts partition the items and sum to the total); anything unmatched falls to
  :data:`OTHER`.
- :func:`build_report` is pure and deterministic — it takes a list of items and returns
  Markdown, so it is trivially testable. :func:`report_from_store` is the thin wrapper that
  reads items from a :class:`~src.store.base.Store` (used by the T0.9 CLI).
"""

from __future__ import annotations

import re

from .. import config
from ..store.base import Store

OTHER = "Other"

# Fixed taxonomy: ordered (category, keywords). Order matters — an item is filed under the
# FIRST category it matches, so put the most specific/highest-signal buckets first. The ROCm
# bucket reuses config.ROCM_HINTS (single source of truth for the ROCm-relevance keywords).
_TAXONOMY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("ROCm / AMD", list(config.ROCM_HINTS)),
    ("Quantization", ["quant", "fp8", "int8", "int4", "awq", "gptq", "marlin", "bitsandbytes"]),
    (
        "Attention / kernels",
        [
            "attention",
            "flashattention",
            "flash attn",
            "paged",
            "kernel",
            "triton",
            "cuda graph",
            "cudagraph",
        ],
    ),
    (
        "Distributed / serving",
        [
            "tensor parallel",
            "pipeline parallel",
            "distributed",
            "ray",
            "serving",
            "scheduler",
            "openai",
            "api server",
        ],
    ),
    (
        "Performance",
        [
            "performance",
            "latency",
            "throughput",
            "speedup",
            "perf",
            "benchmark",
            "regression",
            "memory usage",
        ],
    ),
    (
        "Bugs / crashes",
        [
            "crash",
            "segfault",
            "traceback",
            "exception",
            "hang",
            "deadlock",
            "oom",
            "out of memory",
            "assert",
            "error",
            "fail",
        ],
    ),
]


def _compile(keywords: list[str]) -> re.Pattern[str]:
    """Word-boundary, case-insensitive alternation over the keywords (reduces false hits)."""
    alternation = "|".join(re.escape(kw) for kw in keywords)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


# Precompiled once at import — the report may scan thousands of items.
TAXONOMY: list[tuple[str, re.Pattern[str]]] = [
    (name, _compile(keywords)) for name, keywords in _TAXONOMY_KEYWORDS
]


def _haystack(item: dict) -> str:
    """The text a category is matched against: title + labels + body."""
    labels = " ".join(str(lbl) for lbl in item.get("labels") or [])
    return f"{item.get('title') or ''} {labels} {item.get('body') or ''}"


def categorize(item: dict) -> str:
    """Return the fixed-taxonomy category for `item` (first match wins, else :data:`OTHER`)."""
    text = _haystack(item)
    for name, pattern in TAXONOMY:
        if pattern.search(text):
            return name
    return OTHER


def _cite(item: dict) -> str:
    """One Markdown bullet for an item, always carrying its source link.

    e.g. ``- [vllm-project/vllm#123] hipBLAS build fails on gfx90a — https://github.com/…``
    """
    repo = item.get("repo", "?")
    number = item.get("number", "?")
    title = (item.get("title") or "").strip() or "(no title)"
    url = item.get("url") or ""
    return f"- [{repo}#{number}] {title} — {url}"


def _sort_key(item: dict) -> tuple[str, str, int]:
    """Newest first, then by repo/number for a stable, deterministic order."""
    number = item.get("number")
    return (
        item.get("updated_at") or "",
        item.get("repo") or "",
        number if isinstance(number, int) else 0,
    )


def build_report(items: list[dict], *, title: str = "vLLM (ROCm) weekly digest") -> str:
    """Render `items` into a Markdown digest bucketed by the fixed taxonomy.

    Deterministic and LLM-free. Sections appear in taxonomy order (``Other`` last) and only
    when non-empty; each section header carries its item count, and every item is emitted as a
    bullet that cites its source link. Args: ``title`` — the top-level heading (the T0.9 CLI
    passes a week-stamped title).
    """
    # Partition into buckets (insertion order preserved; Other appended last).
    buckets: dict[str, list[dict]] = {name: [] for name, _ in TAXONOMY}
    buckets[OTHER] = []
    for item in items:
        buckets[categorize(item)].append(item)

    repos = {item.get("repo") for item in items if item.get("repo")}
    lines = [f"# {title}", "", f"{len(items)} items across {len(repos)} repos.", ""]

    for name in [n for n, _ in TAXONOMY] + [OTHER]:
        bucket = buckets[name]
        if not bucket:
            continue
        lines.append(f"## {name} ({len(bucket)})")
        for item in sorted(bucket, key=_sort_key, reverse=True):
            lines.append(_cite(item))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def report_from_store(store: Store, *, title: str = "vLLM (ROCm) weekly digest") -> str:
    """Read all items from `store` and render the report (thin wrapper over :func:`build_report`).

    Used by the T0.9 CLI.
    """
    return build_report(store.query(), title=title)
