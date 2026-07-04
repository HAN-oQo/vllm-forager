"""Forecaster agent (T1.5): emit calibrated, timestamped predictions via ``llm.complete``.

The self-evolution loop's other half of "grade your own calls" (CONTEXT.md decision 6): log
a falsifiable prediction now ("this will become important"), timestamped and with a concrete
resolution rule, so a later grading pass (T2.1) can compare it against what actually happened
and score how well-calibrated this pipeline's judgment is.

Storage: predictions are NOT bolted onto item records — T1.4's own review found that writing
extra fields onto collector-owned item records gets silently clobbered the next time the
collector re-fetches that item (``Store.upsert_items`` is a full replace, not a merge, on
every backend). Predictions instead live in their own append-only log in the KB's generic
state map, the same append-only-sequence pattern :mod:`src.taxonomy`/:mod:`src.policy` use for
versions: ``prediction@1``, ``prediction@2``, ... plus one ``prediction_count`` index.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import llm
from ..store.base import Store

_PREDICTION_KEY_PREFIX = "prediction@"
_COUNT_KEY = "prediction_count"

# The format every other timestamp in this codebase already uses (collector.py's cursor,
# GitHub's own API) — keeping predictions' timestamps in the same shape avoids a second,
# incompatible date convention in the KB.
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "resolution_rule": {"type": "string"},
        "prob": {"type": "number"},
        "due_date": {"type": "string"},
    },
    "required": ["claim", "resolution_rule", "prob", "due_date"],
}


class ForecastError(RuntimeError):
    """Any forecaster failure: an out-of-range/malformed prediction, or a corrupt KB record."""


def _parse_ts(raw: str) -> datetime:
    """Parse a ``_TS_FORMAT`` timestamp; raise :class:`ForecastError` (not ValueError)."""
    try:
        return datetime.strptime(raw, _TS_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ForecastError(f"{raw!r} is not a valid {_TS_FORMAT!r} timestamp") from exc


@dataclass(frozen=True)
class Prediction:
    """One calibrated, timestamped prediction, validated at construction time.

    ``evidence`` cites the source item(s) the claim is about (the evidence principle: every
    KB record carries a source link) — a tuple so a future multi-item/trend forecast can cite
    more than one, without changing the shape.

    Raises:
        ForecastError: `prob` isn't in ``[0, 1]``, or `due_date`/`created_at` aren't valid
            timestamps, or `due_date` isn't after `created_at`.
    """

    claim: str
    resolution_rule: str
    prob: float
    due_date: str
    evidence: tuple[str, ...]
    created_at: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ForecastError(f"prob must be in [0, 1], got {self.prob}")
        created = _parse_ts(self.created_at)
        due = _parse_ts(self.due_date)
        if due <= created:
            raise ForecastError(f"due_date {self.due_date!r} must be after {self.created_at!r}")

    def to_json(self) -> str:
        """Serialize for storage in the KB's state map."""
        return json.dumps(
            {
                "claim": self.claim,
                "resolution_rule": self.resolution_rule,
                "prob": self.prob,
                "due_date": self.due_date,
                "evidence": list(self.evidence),
                "created_at": self.created_at,
            }
        )

    @staticmethod
    def from_json(raw: str) -> Prediction:
        """Deserialize a value previously produced by :meth:`to_json`.

        Raises:
            ForecastError: `raw` isn't valid JSON, isn't shaped like a prediction record, or
                fails the same validation :meth:`__post_init__` enforces at creation time.
        """
        try:
            data = json.loads(raw)
            return Prediction(
                claim=data["claim"],
                resolution_rule=data["resolution_rule"],
                prob=data["prob"],
                due_date=data["due_date"],
                evidence=tuple(data["evidence"]),
                created_at=data["created_at"],
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ForecastError(f"corrupt prediction record: {exc}") from exc


def forecast_item(item: dict, *, now: datetime | None = None) -> Prediction:
    """Ask the model for one calibrated prediction about `item`'s future.

    `now` fixes the "made at" timestamp (defaults to the current time, UTC) — tests pass a
    fixed clock so due-date validation isn't flaky.

    Raises:
        llm.LLMError: the completion call failed (transport error, timeout, non-JSON reply).
        ForecastError: the model's reply doesn't shape into a valid, calibrated prediction
            (see :meth:`Prediction.__post_init__`) — a malformed reply is a loud failure here,
            not a silent fallback, since a bad prediction would corrupt the grading signal
            T2.1 depends on.
    """
    when = now or datetime.now(timezone.utc)
    title = item.get("title") or ""
    body = (item.get("body") or "")[:2000]
    prompt = (
        "Make one falsifiable, calibrated prediction about this GitHub issue/PR's future "
        "(e.g. will it be merged/closed/widely adopted, and by when). Reply with a `claim` "
        "(what you predict), a `resolution_rule` (exactly how to check later whether it came "
        "true), a `prob` (your confidence it resolves true, 0.0-1.0), and a `due_date` "
        f"(UTC timestamp in {_TS_FORMAT!r} format, after {when.strftime(_TS_FORMAT)}) by "
        "which it should be resolvable.\n\n"
        f"Title: {title}\n\nBody: {body}"
    )
    reply = llm.complete(prompt, json_schema=_PREDICTION_SCHEMA)
    if not isinstance(reply, dict):
        raise ForecastError(f"expected a JSON object reply, got {type(reply).__name__}")
    try:
        return Prediction(
            claim=reply["claim"],
            resolution_rule=reply["resolution_rule"],
            prob=reply["prob"],
            due_date=reply["due_date"],
            evidence=(item.get("url") or "",),
            created_at=when.strftime(_TS_FORMAT),
        )
    except KeyError as exc:
        raise ForecastError(f"model reply missing required field: {exc}") from exc


def record_prediction(store: Store, prediction: Prediction) -> int:
    """Append `prediction` to the KB's prediction log; return its 1-based sequence number."""
    raw = store.get_state(_COUNT_KEY)
    next_id = (int(raw) if raw else 0) + 1
    store.set_state(f"{_PREDICTION_KEY_PREFIX}{next_id}", prediction.to_json())
    store.set_state(_COUNT_KEY, str(next_id))
    return next_id


def list_predictions(store: Store) -> list[Prediction]:
    """Return every recorded prediction, oldest first."""
    raw = store.get_state(_COUNT_KEY)
    count = int(raw) if raw else 0
    predictions = []
    for i in range(1, count + 1):
        stored = store.get_state(f"{_PREDICTION_KEY_PREFIX}{i}")
        if stored is not None:
            predictions.append(Prediction.from_json(stored))
    return predictions


def forecast_store(store: Store, *, now: datetime | None = None) -> list[Prediction]:
    """Forecast every classified item that doesn't already have a recorded prediction.

    "Delta" here is by evidence, not an item-record field (see the module docstring on why):
    an item is skipped once any of its own URLs appears in an existing prediction's evidence.
    A failing item (``llm.LLMError``/``ForecastError``) is skipped and logged rather than
    losing every other item's already-produced prediction in the same run. `now` fixes the
    "made at" timestamp for every prediction in this run (see :func:`forecast_item`).

    Returns the newly-recorded predictions (``[]`` if there was nothing to do).
    """
    already_forecast = {url for p in list_predictions(store) for url in p.evidence}
    pending = [
        item
        for item in store.query()
        if item.get("category") and item.get("url") not in already_forecast
    ]
    predictions = []
    for item in pending:
        try:
            prediction = forecast_item(item, now=now)
        except (llm.LLMError, ForecastError) as exc:
            print(
                f"forecaster: skipping {item.get('repo')}#{item.get('number')}: {exc}",
                file=sys.stderr,
            )
            continue
        record_prediction(store, prediction)
        predictions.append(prediction)
    return predictions
