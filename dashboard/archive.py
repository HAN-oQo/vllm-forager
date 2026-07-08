"""Reports/Trends archive tab (T5.9): browse past tree reports (:mod:`src.report`'s own
``--tree`` output), newest first.

Composes over :func:`~src.report.list_reports`/:func:`~src.report.read_tree_report` -- an
earlier version implemented the filesystem-scanning logic directly in this module, a
code-review finding that this inverted the established ``src`` (owns state) / ``dashboard``
(reads it) dependency direction T5.6 (:mod:`dashboard.review` over :mod:`src.gate`) and T5.7
(:mod:`dashboard.guardrails` over :mod:`src.audit`) already established, and left no
non-dashboard caller (a future CLI, a narrative composer) able to reuse the same logic without
reaching into a dashboard module. This module is now just the `Store` -> `data_dir` resolution
+ degrade-on-missing-config glue.

Scoped to what :func:`~src.report.generate_tree` actually writes -- one report per
:func:`~src.trends.week_stamp` (e.g. ``"2026-W27"``), not literally "one per calendar day" as
this todo's own title says: :mod:`src.orchestrator`'s real "intel" stage (which calls
``report.main``) runs on a **weekly** cadence (``_INTEL_INTERVAL_HOURS = 24 * 7``), a
deliberate cost-bounding choice (T4.8-T4.11), not something this dashboard-read todo should
change to manufacture a literal daily cadence that doesn't exist. Mirrors the same "the
DEVPLAN 'e.g.' is illustrative, not a literal contract" pattern already established for
T5.5/T5.8's own per-agent "e.g." nodes that aren't independently tracked at that granularity
either.

Not `Store`-abstracted: :mod:`src.report`'s tree/markdown files are raw filesystem artifacts
under ``<data_dir>/reports/``, the same "``data_dir``, not `Store`" shape :mod:`src.gate`'s PR
drafts and :mod:`src.audit`'s data-quality sink already have -- `data_dir` is accepted the
same optional, override-if-given way T5.6/T5.7 already established, resolved via
:func:`~src.audit.try_resolve_data_dir` (not a fourth independently-derived copy of the same
one-liner) and degraded to an empty/`None` result rather than raised, matching every other
read endpoint's own established convention.
"""

from __future__ import annotations

from pathlib import Path

from src.audit import try_resolve_data_dir
from src.report import list_reports, read_tree_report
from src.store.base import Store


def list_archived_reports(store: Store, *, data_dir: Path | None = None) -> list[dict]:
    """Every archived tree report, newest first -- see :func:`~src.report.list_reports` for
    the exact shape. Degrades to ``[]`` if `data_dir` can't be resolved."""
    resolved = try_resolve_data_dir(store, data_dir)
    return list_reports(resolved) if resolved is not None else []


def open_archived_report(store: Store, stamp: str, *, data_dir: Path | None = None) -> list | None:
    """The tree content of one archived report -- see :func:`~src.report.read_tree_report`
    for the exact shape and its `stamp` validation. Degrades to `None` if `data_dir` can't be
    resolved."""
    resolved = try_resolve_data_dir(store, data_dir)
    return read_tree_report(resolved, stamp) if resolved is not None else None
