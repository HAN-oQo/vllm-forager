"""Report CLI (T0.9, upgraded to v1 in T1.6, tree-structured via T1.5.4) — ``python -m
src.report`` writes the weekly Markdown digest.

One command turns the collected KB into a dated report on disk: it reads every item from
the store, renders the LLM-written, cited digest (:mod:`src.agents.reporter_v1`, T1.6), and
writes it to ``data/reports/YYYY-Www.md`` (ISO-year + ISO-week, e.g. ``2026-W27.md``), then
prints the path. This is the on-demand / scheduled entry point for the report — the
rendering itself lives in the reporter module so it stays pure and testable.

(The fixed-keyword, LLM-free v0 renderer, :mod:`src.agents.reporter`, T0.8, still exists — v1
builds on its citation helpers and its own tests still pin its standalone behavior — but this
CLI's default output is v1's.)

``--tree`` (T1.5.4, opt-in — the default output is unchanged, so existing consumers like
T1.9's dashboard never see a surprise format change) renders the taxonomy-tree digest instead
(:func:`~src.agents.reporter_v1.build_tree`/``render_tree_markdown``, nested by classified
``path`` with T1.5.3's node summaries) via :func:`generate_tree`, and additionally writes
``<week>.tree.json`` — the nested ``{name, summary, count, gaps, children, prs}`` structure
T1.5.5's dashboard will read directly, rather than re-deriving the tree from raw items itself.

Design:
- :func:`generate` is the injectable core (store + output dir + clock in, path out) so tests
  never touch the real ``data/`` tree or wall-clock.
- :func:`main` is the thin CLI wrapper, using :func:`~src.store.resolve_store` (T1.11) for
  store selection: with no ``--data-dir``, it uses :func:`src.store.get_store` (T0.6.2) — so
  ``STORE=firestore`` reports from Firestore just like the collector does. ``--data-dir`` is a
  JSONL-specific override (pre-dates the store factory): passing it always reads a
  :class:`JsonlStore` at that path, regardless of ``STORE`` — there's no equivalent "read
  Firestore instead" flag, so mixing the two isn't meaningful.
- :func:`week_stamp` is re-exported from :mod:`src.trends` (T1.7), which is where it's
  actually defined — a pure, dependency-free function belongs in the lowest-altitude shared
  module, not this CLI, so a future consumer (e.g. M5's dashboard) can use it without pulling
  in argparse/the store/the LLM-backed reporter stack this module imports.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .agents import reporter, reporter_v1
from .store import Store, resolve_store
from .trends import week_stamp

__all__ = ["week_stamp", "generate", "generate_tree", "main"]


def generate(
    store: Store,
    reports_dir: Path,
    *,
    when: datetime | None = None,
    title: str | None = None,
    render: Callable[..., str] = reporter_v1.report_from_store,
) -> Path:
    """Render the weekly report from `store` and write it under `reports_dir`.

    Writes ``reports_dir/<week_stamp>.md`` (creating `reports_dir` if needed) and returns the
    path. `when` fixes the week stamp (defaults to now, UTC); `title` overrides the heading
    (defaults to a week-stamped title so the file names itself in its first line too). `render`
    is the report-building function (`store, *, title -> markdown`) — defaults to v1's
    LLM-written digest; pass ``reporter.report_from_store`` (v0) as an offline/no-LLM fallback
    (the CLI's ``--v0`` flag does this).
    """
    stamp = week_stamp(when)
    resolved_title = title or f"vLLM (ROCm) weekly digest — {stamp}"
    md = render(store, title=resolved_title)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{stamp}.md"
    path.write_text(md, encoding="utf-8")
    return path


def generate_tree(
    store: Store,
    reports_dir: Path,
    *,
    when: datetime | None = None,
    title: str | None = None,
) -> tuple[Path, Path]:
    """Render the weekly report as a taxonomy tree (T1.5.4) and write both forms.

    Writes ``reports_dir/<week_stamp>.md`` (indented 大→소(summary)→소소→PRs Markdown,
    :func:`~src.agents.reporter_v1.render_tree_markdown`) and
    ``reports_dir/<week_stamp>.tree.json`` (the same tree as a list of
    :meth:`~src.agents.reporter_v1.TreeNode.to_dict`) — T1.5.5's dashboard reads the JSON
    directly rather than re-deriving the tree from raw items. Returns ``(md_path, json_path)``.
    """
    stamp = week_stamp(when)
    resolved_title = title or f"vLLM (ROCm) weekly digest — {stamp}"
    nodes = reporter_v1.tree_from_store(store)
    md = reporter_v1.render_tree_markdown(nodes, title=resolved_title)

    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"{stamp}.md"
    md_path.write_text(md, encoding="utf-8")
    json_path = reports_dir / f"{stamp}.tree.json"
    json_path.write_text(json.dumps([node.to_dict() for node in nodes], indent=2), encoding="utf-8")
    return md_path, json_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: write ``<data-dir>/reports/YYYY-Www.md`` and print its path.

    `argv` is parsed (defaults to ``sys.argv`` when None). The report is always written to a
    ``reports/`` subdirectory of the data dir (``--data-dir`` if given, else
    :data:`src.config.DATA_DIR`). Returns a process exit code (0 on success).

    Store selection is :func:`~src.store.resolve_store`'s shared contract: with ``--data-dir``,
    reads a :class:`~src.store.jsonl_store.JsonlStore` at that explicit path (the pre-T0.6.2
    behavior, kept for anyone pointing this at an arbitrary JSONL export). Without it, uses
    :func:`~src.store.get_store` — so ``STORE=firestore`` reports read from Firestore, matching
    the collector's own backend selection.
    """
    ap = argparse.ArgumentParser(
        prog="python -m src.report",
        description="Write the weekly vLLM (ROCm) Markdown digest from the collected KB.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Read a JSONL store at this path instead of the STORE-selected backend "
            "(default: config.DATA_DIR, backend from env STORE=jsonl|firestore)."
        ),
    )
    ap.add_argument(
        "--v0",
        action="store_true",
        help=(
            "Render the T0.8 fixed-keyword, LLM-free digest instead of v1's LLM-written one "
            "— an offline fallback for an LLM outage/cost/bad-output incident."
        ),
    )
    ap.add_argument(
        "--tree",
        action="store_true",
        help=(
            "Render the T1.5.4 taxonomy-tree digest (nested by classified path, with T1.5.3 "
            "node summaries) instead of the flat per-category one, and also write a "
            "<week>.tree.json alongside the Markdown for T1.5.5's dashboard. Mutually "
            "exclusive with --v0 (v0's fixed keyword taxonomy has no notion of a path)."
        ),
    )
    args = ap.parse_args(argv)
    if args.v0 and args.tree:
        ap.error("--v0 and --tree are mutually exclusive")

    store, data_dir = resolve_store(args.data_dir)
    reports_dir = data_dir / "reports"

    if args.tree:
        md_path, json_path = generate_tree(store, reports_dir)
        print(md_path)
        print(json_path)
        return 0

    render = reporter.report_from_store if args.v0 else reporter_v1.report_from_store
    path = generate(store, reports_dir, render=render)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
