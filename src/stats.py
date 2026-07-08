"""Summarize collected data: per-repo issue/PR counts (+ ROCm-tagged) and a grand total.

Usage:
    python -m src.stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import config


def summarize_items(items: list[dict]) -> dict:
    """Per-item type/rocm-tag counts + total: ``{"total": N, "issue": N, "pr": N, "rocm": N}``.

    The one counting rule :func:`summarize` (fed records read from a `data/*.jsonl` file) and
    :func:`~dashboard.panels.collection_stats` (fed `Store.query()`) both share, so a dashboard
    panel and this CLI can't independently drift on what counts as ROCm-relevant
    (:data:`~src.config.ROCM_HINTS`) or how an item's type is tallied -- a code-review finding:
    an earlier version of `dashboard.panels.collection_stats` re-implemented this exact loop
    instead of sharing it.
    """
    counts = {"total": 0, "issue": 0, "pr": 0, "rocm": 0}
    for item in items:
        counts["total"] += 1
        item_type = item.get("type")
        if item_type in ("issue", "pr"):
            counts[item_type] += 1
        labels = " ".join(str(x).lower() for x in (item.get("labels") or []))
        if any(hint in labels for hint in config.ROCM_HINTS):
            counts["rocm"] += 1
    return counts


def summarize(data_dir: Path | None = None) -> dict:
    """Count records in each data/*.jsonl. Returns {"repos": {name: {...}}, "total": N}."""
    data_dir = data_dir or config.DATA_DIR
    repos: dict[str, dict] = {}
    total = 0
    for path in sorted(data_dir.glob("*.jsonl")):
        records = []
        # Split on "\n" only — the collector writes records with ensure_ascii=False, so a
        # body may contain a literal U+2028/U+2029/U+0085; str.splitlines() would split on
        # those and break the record. Skip a genuinely corrupt line rather than crashing the
        # whole summary (mirrors collector._merge_jsonl's tolerance).
        for lineno, line in enumerate(path.read_text().split("\n"), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  !! {path.name}:{lineno} skipping corrupt line ({exc})", file=sys.stderr)
                continue
        repos[path.name] = summarize_items(records)
        total += repos[path.name]["total"]
    return {"repos": repos, "total": total}


def main() -> None:
    summary = summarize()
    for name, r in summary["repos"].items():
        print(f"{name:<32} {r['total']:>6}  (pr={r['pr']} issue={r['issue']} rocm~={r['rocm']})")
    print(f"{'TOTAL':<32} {summary['total']:>6}")


if __name__ == "__main__":
    main()
