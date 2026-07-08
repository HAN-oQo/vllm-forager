"""Reports/Trends archive tab (T5.9): browse past tree reports (:mod:`src.report`'s own
``--tree`` output), newest first.

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
same optional, override-if-given way T5.6/T5.7 already established, and this module degrades
to an empty result rather than raising when it can't resolve one, matching every other read
endpoint's own established convention.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.store.base import Store

_REPORTS_SUBDIR = "reports"
_TREE_JSON_SUFFIX = ".tree.json"


def _reports_dir(store: Store, data_dir: Path | None) -> Path | None:
    resolved = data_dir if data_dir is not None else getattr(store, "data_dir", None)
    return Path(resolved) / _REPORTS_SUBDIR if resolved is not None else None


def list_archived_reports(store: Store, *, data_dir: Path | None = None) -> list[dict]:
    """Every archived tree report under ``<data_dir>/reports/*.tree.json``, newest (highest
    week-stamp) first, as ``{"stamp": week_stamp, "path": str}`` -- the path only, not the
    (potentially large) tree content itself; use :func:`open_archived_report` to read one.

    Sorting is a plain descending string sort on the stamp -- ``week_stamp``'s own
    ``"%G-W%V"`` format zero-pads the week number (``"W05"``, not ``"W5"``), so this sorts
    correctly by year-then-week without needing to parse each stamp back into a date.

    Degrades to ``[]`` (no `data_dir` resolvable, or the ``reports/`` directory doesn't exist
    yet) rather than raising -- a fresh KB that's never run ``python -m src.report --tree`` is
    a normal state for a read endpoint to degrade past.
    """
    reports_dir = _reports_dir(store, data_dir)
    if reports_dir is None or not reports_dir.is_dir():
        return []
    stamps = sorted(
        (p.name.removesuffix(_TREE_JSON_SUFFIX) for p in reports_dir.glob(f"*{_TREE_JSON_SUFFIX}")),
        reverse=True,
    )
    return [
        {"stamp": stamp, "path": str(reports_dir / f"{stamp}{_TREE_JSON_SUFFIX}")}
        for stamp in stamps
    ]


def open_archived_report(store: Store, stamp: str, *, data_dir: Path | None = None) -> list | None:
    """The tree content of one archived report (:meth:`~src.agents.reporter_v1.TreeNode.
    to_dict`'s own list-of-nodes shape, per :func:`~src.report.generate_tree`'s own
    ``.tree.json`` format), or `None` if `stamp` has no archived report (an unknown stamp, a
    corrupt file, or no `reports/` directory at all) -- mirrors this module's own
    list-side "degrade past a data gap, don't raise" convention.
    """
    reports_dir = _reports_dir(store, data_dir)
    if reports_dir is None:
        return None
    path = reports_dir / f"{stamp}{_TREE_JSON_SUFFIX}"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
