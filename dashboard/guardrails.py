"""Guardrail panels (T5.7): "trust the numbers" made visible -- the T0.11 data-quality
reconciliation history and the T1.8 RAG-eval score history, each already carrying its own
threshold flag, surfaced as one clean read.

Data-shaping only, matching T5.1-T5.6's own established "data layer first" precedent (actual
drift-line/threshold-marker chart rendering is a further step, not this one -- see
:mod:`dashboard.trend_charts`'s own docstring for the same scope boundary on a sibling todo).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from src import audit, rag_eval
from src.store.base import Store


def data_quality_series(store: Store, *, data_dir: Path | None = None) -> list[dict]:
    """Every recorded ``data_quality`` check (T0.11 reconciliation + T0.12 cursor-stall), oldest
    first, as plain dicts -- thin re-export of :func:`~src.audit.list_records`. Each record
    already carries its own ``flagged`` boolean (computed at write time), so no threshold logic
    is re-derived here; this panel doesn't decide what counts as bad, it just surfaces the
    signal `src.audit` already computed.
    """
    return audit.list_records(store, data_dir=data_dir)


def rag_eval_series(store: Store) -> list[dict]:
    """Every recorded RAG-eval run (T1.8), oldest first, each with its own ``passed`` threshold
    flag included -- :attr:`~src.rag_eval.RagEvalScore.passed` is a computed ``@property``, not
    a dataclass field, so ``dataclasses.asdict`` alone would silently drop it (mirrors
    :func:`dashboard.api._candidate_dict`'s own identical fix for
    :attr:`~src.agents.scout.Candidate.priority`).
    """
    return [{**asdict(score), "passed": score.passed} for score in rag_eval.list_scores(store)]
