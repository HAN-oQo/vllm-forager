"""Summarize collected data: per-repo issue/PR counts (+ ROCm-tagged) and a grand total.

Usage:
    python -m src.stats
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import config


def summarize(data_dir: Path | None = None) -> dict:
    """Count records in each data/*.jsonl. Returns {"repos": {name: {...}}, "total": N}."""
    data_dir = data_dir or config.DATA_DIR
    repos: dict[str, dict] = {}
    total = 0
    for path in sorted(data_dir.glob("*.jsonl")):
        counts: Counter[str] = Counter()
        n = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            counts[rec.get("type", "?")] += 1
            labels = " ".join(str(x).lower() for x in (rec.get("labels") or []))
            if any(hint in labels for hint in config.ROCM_HINTS):
                counts["rocm"] += 1
        repos[path.name] = {
            "total": n,
            "pr": counts["pr"],
            "issue": counts["issue"],
            "rocm": counts["rocm"],
        }
        total += n
    return {"repos": repos, "total": total}


def main() -> None:
    summary = summarize()
    for name, r in summary["repos"].items():
        print(f"{name:<32} {r['total']:>6}  (pr={r['pr']} issue={r['issue']} rocm~={r['rocm']})")
    print(f"{'TOTAL':<32} {summary['total']:>6}")


if __name__ == "__main__":
    main()
