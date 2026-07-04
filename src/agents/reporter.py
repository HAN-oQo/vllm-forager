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
    r"""Case-insensitive alternation matching a keyword as a **token prefix**.

    Only a boundary *before* the keyword is required (``\b``); the keyword may be followed by
    more word chars. This is deliberate: a trailing ``\b`` would exclude the suffixed forms that
    dominate real text — ``gfx`` → ``gfx90a``/``gfx942``, ``mi300`` → ``MI300X``, ``fp8`` →
    ``fp8e4m3`` — while the leading boundary still blocks mid-word hits (``hip`` ∌ ``chip``).
    """
    alternation = "|".join(re.escape(kw) for kw in keywords)
    return re.compile(rf"\b(?:{alternation})", re.IGNORECASE)


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


def _num(item: dict) -> int:
    """The item's issue/PR number as an int (0 if missing/non-int) — for stable sorting."""
    number = item.get("number")
    return number if isinstance(number, int) else 0


def evidence_url(item: dict) -> str:
    """`item`'s own ``url``, or a synthesized canonical GitHub link (GitHub redirects
    ``/issues/N`` ↔ ``/pull/N``) if it has none — so every caller gets the identical
    URL-or-synthesized-fallback treatment from one place, rather than each reimplementing it
    (shared by :func:`_cite` here and :func:`~src.agents.reporter_v1._pr_entry`).

    Known limitation: if `item` has no ``url`` AND is also missing ``repo``/``number``, this
    returns ``""`` — an item that arrives with none of the three has no evidence to construct
    a link from. Every item this codebase actually produces (the collector always sets all
    three) never hits this; it's only reachable from a malformed/hand-built record.
    """
    url = item.get("url") or ""
    if not url and item.get("repo") and item.get("number") is not None:
        url = f"https://github.com/{item['repo']}/issues/{item['number']}"
    return url


def repo_number_label(item: dict) -> str:
    """``repo#number`` for `item`, ``?`` for a missing or present-but-``None`` field.

    Shared by :func:`_cite` here and by ``dashboard.render``'s PR/issue rows, since a
    caller-built dict (e.g. reporter_v1's ``_pr_entry``) may set ``repo``/``number`` to `None`
    rather than omitting them — ``.get(key, "?")`` alone wouldn't catch that, so both
    "missing" and "present but None" must be checked explicitly here, once, for every caller.
    """
    repo = item.get("repo") or "?"
    number = item.get("number")
    number = number if number is not None else "?"
    return f"{repo}#{number}"


def _cite(item: dict, *, text: str | None = None) -> str:
    """One Markdown bullet for an item, always carrying its source link.

    e.g. ``- [vllm-project/vllm#123] hipBLAS build fails on gfx90a — https://github.com/…``

    `text` overrides the item's own title (e.g. reporter_v1's LLM-written claim) — it gets
    the exact same treatment as the title: internal whitespace collapsed so a stray newline
    can't inject report structure (which would also desync the per-section count from the
    visible bullets), and the same URL-or-synthesized-fallback below. This way every caller —
    whichever text it renders — gets the identical evidence-principle guarantee, from one
    place, rather than each reimplementing it.

    If the item has no ``url``, synthesize the canonical GitHub link (GitHub redirects
    ``/issues/N`` ↔ ``/pull/N``) so every bullet still cites a source (evidence principle).
    """
    raw_text = text if text is not None else (item.get("title") or "")
    rendered = " ".join(raw_text.split()) or "(no title)"
    return f"- [{repo_number_label(item)}] {rendered} — {evidence_url(item)}"


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

    # `buckets` is already in TAXONOMY-then-OTHER order (dicts keep insertion order).
    for name, bucket in buckets.items():
        if not bucket:
            continue
        # Newest first; ties broken by repo then number, both ascending (stable two-pass).
        ordered = sorted(bucket, key=lambda item: (item.get("repo") or "", _num(item)))
        ordered.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        lines.append(f"## {name} ({len(bucket)})")
        lines.extend(_cite(item) for item in ordered)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def report_from_store(store: Store, *, title: str = "vLLM (ROCm) weekly digest") -> str:
    """Read all items from `store` and render the report (thin wrapper over :func:`build_report`).

    Used by the T0.9 CLI.
    """
    return build_report(store.query(), title=title)
