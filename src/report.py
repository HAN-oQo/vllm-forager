"""Report CLI (T0.9) — ``python -m src.report`` writes the weekly Markdown digest.

One command turns the collected KB into a dated report on disk: it reads every item from
the store, renders the baseline digest (:mod:`src.agents.reporter`, T0.8), and writes it to
``data/reports/YYYY-Www.md`` (ISO-year + ISO-week, e.g. ``2026-W27.md``), then prints the
path. This is the on-demand / scheduled entry point for the report — the rendering itself
lives in the reporter so it stays pure and testable.

Design:
- :func:`generate` is the injectable core (store + output dir + clock in, path out) so tests
  never touch the real ``data/`` tree or wall-clock.
- :func:`main` is the thin CLI wrapper: it wires the default :class:`JsonlStore` over
  :data:`src.config.DATA_DIR` (overridable with ``--data-dir``) and prints the written path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .agents import reporter
from .store.base import Store
from .store.jsonl_store import JsonlStore


def week_stamp(when: datetime | None = None) -> str:
    """The ISO-year/ISO-week stamp used for the filename, e.g. ``2026-W27``.

    ``%G``/``%V`` are the ISO-8601 year and week (not ``%Y``/``%U``): near a year boundary
    the ISO week's year can differ from the calendar year, and this keeps week numbers
    contiguous (…W52, W53?, W01…). Defaults to now (UTC) when `when` is omitted.
    """
    when = when or datetime.now(timezone.utc)
    return when.strftime("%G-W%V")


def generate(
    store: Store,
    reports_dir: Path,
    *,
    when: datetime | None = None,
    title: str | None = None,
) -> Path:
    """Render the weekly report from `store` and write it under `reports_dir`.

    Writes ``reports_dir/<week_stamp>.md`` (creating `reports_dir` if needed) and returns the
    path. `when` fixes the week stamp (defaults to now, UTC); `title` overrides the heading
    (defaults to a week-stamped title so the file names itself in its first line too).
    """
    stamp = week_stamp(when)
    resolved_title = title or f"vLLM (ROCm) weekly digest — {stamp}"
    md = reporter.report_from_store(store, title=resolved_title)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{stamp}.md"
    path.write_text(md, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: write ``<data-dir>/reports/YYYY-Www.md`` and print its path.

    `argv` is parsed (defaults to ``sys.argv`` when None). ``--data-dir`` overrides the store
    location (defaults to :data:`src.config.DATA_DIR`); the report is always written to a
    ``reports/`` subdirectory of that data dir. Returns a process exit code (0 on success).
    """
    ap = argparse.ArgumentParser(
        prog="python -m src.report",
        description="Write the weekly vLLM (ROCm) Markdown digest from the collected KB.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="KB data directory to read the store from (default: config.DATA_DIR).",
    )
    args = ap.parse_args(argv)

    store = JsonlStore(args.data_dir)
    path = generate(store, args.data_dir / "reports")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
