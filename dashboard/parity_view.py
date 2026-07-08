"""Parity diagram (T5.4): the engine x capability heatmap's own data layer -- a full grid,
not just the cells with recorded activity, so every gap is guaranteed a visible cell to color
red.

:func:`~dashboard.api.parity` (T5.1) already returns ``cells`` (:class:`~src.parity.ParityCell`,
one per (engine, capability) pair that has at least one *classified* item) and ``gaps``
(:class:`~src.parity.Gap`, one per capability missing on some primary target). Naively merging
a gap flag onto that sparse ``cells`` list would silently drop most real gaps from a heatmap: a
capability that's **never even been discussed** on the target engine (the most common, most
important kind of gap) has no `ParityCell` at all to attach a flag to -- by construction, a gap
means "not present here," and "not present" includes "no cell exists yet," not just "a cell
exists with `present=False`." :func:`parity_matrix` closes that gap (no pun intended) by
enumerating the full engine x capability grid and synthesizing an absent cell (``present=False,
evidence=None``) wherever no real one exists, so a gap always has *some* cell to mark
``is_gap=True`` on -- this todo's own worked example directly: "red cells = upstream gaps."
"""

from __future__ import annotations

from src.store.base import Store

from . import api


def parity_matrix(store: Store) -> dict:
    """``{"engines": [...], "capabilities": [...], "cells": [{"engine", "capability",
    "present", "evidence", "is_gap"}, ...]}`` -- the full grid a heatmap renders directly,
    one row per engine, one column per capability, every cell present exactly once.

    `engines`/`capabilities` are derived from real signal only (every engine that appears in
    at least one cell or is cited as some gap's `target_engine`; every capability that appears
    in at least one cell) -- not :data:`~src.config.REPOS`'s full tracked-repo list, which can
    include engines with zero classified items yet, whose all-blank row would add noise
    without showing "who has what, who lags" anything real.

    `is_gap` is `True` for exactly the ``(target_engine, capability)`` pairs
    :func:`~dashboard.api.parity`'s own ``gaps`` list names -- a capability shipped on *some*
    other tracked engine but not this one. A cell can be both ``present=False`` and
    ``is_gap=False`` (nothing classified for that engine/capability at all, and no other engine
    ships it either -- not a gap, just genuinely irrelevant to that engine).
    """
    result = api.parity(store)
    cells_by_key = {(cell["engine"], cell["capability"]): cell for cell in result["cells"]}
    gap_targets = {(gap["target_engine"], gap["capability"]) for gap in result["gaps"]}

    engines = sorted({cell["engine"] for cell in result["cells"]} | {t for t, _ in gap_targets})
    capabilities = sorted({cell["capability"] for cell in result["cells"]})

    cells = []
    for engine in engines:
        for capability in capabilities:
            cell = cells_by_key.get((engine, capability))
            cells.append(
                {
                    "engine": engine,
                    "capability": capability,
                    "present": cell["present"] if cell else False,
                    "evidence": cell["evidence"] if cell else None,
                    "is_gap": (engine, capability) in gap_targets,
                }
            )
    return {"engines": engines, "capabilities": capabilities, "cells": cells}
