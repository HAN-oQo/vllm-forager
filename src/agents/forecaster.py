"""Forecaster agent (T1.5): emit calibrated, timestamped predictions via ``llm.complete``.

The self-evolution loop's other half of "grade your own calls" (CONTEXT.md decision 6): log
a falsifiable prediction now ("this will become important"), timestamped and with a concrete
resolution rule, so a later grading pass (T2.1) can compare it against what actually happened
and score how well-calibrated this pipeline's judgment is.

Storage: predictions are NOT bolted onto item records — at the time this module was written,
T1.4's own review had found that writing extra fields onto collector-owned item records gets
silently clobbered the next time the collector re-fetches that item (``Store.upsert_items``
was a full replace, not a merge, on every backend — since fixed in T1.10). Predictions still
belong in their own append-only log rather than on item records: unlike a classification (one
value per item, naturally overwritten as re-classified), a prediction log is inherently
history — many entries can exist per item over time, which a per-item field could never
represent. State map: ``prediction@1``, ``prediction@2``, ... plus one ``prediction_count``
index.

Known limitation (not fixed here): unlike :mod:`src.taxonomy`/:mod:`src.policy`, which hold a
*small, curated* number of versions of one object, this log grows without bound (one entry
per classified item). ``Store.get_state``/``set_state`` reload/rewrite the *entire* shared
state map on every call, so :func:`list_predictions` (an unconditional full-log read on every
:func:`forecast_store` run) and :func:`record_prediction` get more expensive as the log grows
— and neither backend can filter by ``due_date``/resolution status server-side. The read-then
-write in :func:`record_prediction` also has no locking, a race that matters more here than
for taxonomy/policy since predictions are written continuously, not rarely. The real fix is a
Store-level primitive for a large, independent, queryable record collection (distinct from
the small state map and the GitHub-item-shaped ``items`` bucket) — this is a *different* gap
than T1.4's item-field-clobbering issue (fixed by T1.10's merge-on-upsert semantics): that one
was about *how* a write lands on an existing record, this one is about the state map's own
read/write shape not scaling to an ever-growing collection. Recommend a dedicated follow-up
when the log's cost actually bites, not bundled into a merge-semantics fix that doesn't touch
this axis at all.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import llm
from ..store.base import Store

_PREDICTION_KEY_PREFIX = "prediction@"
_COUNT_KEY = "prediction_count"

# The format every other timestamp in this codebase already uses (collector.py's cursor,
# GitHub's own API) — keeping predictions' timestamps in the same shape avoids a second,
# incompatible date convention in the KB. Only used to *render* timestamps we generate
# ourselves (created_at); see parse_ts for what we accept back from the model. Public (with
# parse_ts below) — T2.1's grader parses/renders the same due_date/created_at timestamps.
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# What we accept when *parsing* a timestamp: the base second-precision form, with optional
# fractional seconds and a "Z"/"+00:00"-style UTC marker — an LLM told to use TS_FORMAT
# routinely still adds milliseconds or spells the offset differently, so exact-string
# matching (a plain strptime) would silently reject most real (non-mocked) replies.
_TS_RE = re.compile(r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:Z|\+00:?00)?$")

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


def parse_ts(raw: str) -> datetime:
    """Parse a UTC timestamp (tolerating fractional seconds and a "Z"/"+00:00" marker).

    Raises:
        ForecastError: `raw` doesn't match :data:`_TS_RE` (not a bare ValueError) — a non-UTC
            offset (e.g. ``+05:00``) is rejected rather than silently misread as UTC.
    """
    match = _TS_RE.match(raw.strip())
    if not match:
        raise ForecastError(f"{raw!r} is not a valid UTC timestamp")
    return datetime.strptime(match.group("base"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def _parse_count(raw: str | None) -> int:
    """Parse the prediction-count index; raise :class:`ForecastError` (not ValueError)."""
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError as exc:
        raise ForecastError(f"corrupt prediction_count {raw!r}") from exc


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
        created = parse_ts(self.created_at)
        due = parse_ts(self.due_date)
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
        f"(UTC timestamp in {TS_FORMAT!r} format, after {when.strftime(TS_FORMAT)}) by "
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
            created_at=when.strftime(TS_FORMAT),
        )
    except KeyError as exc:
        raise ForecastError(f"model reply missing required field: {exc}") from exc


def record_prediction(store: Store, prediction: Prediction) -> int:
    """Append `prediction` to the KB's prediction log; return its 1-based sequence number."""
    next_id = _parse_count(store.get_state(_COUNT_KEY)) + 1
    store.set_state(f"{_PREDICTION_KEY_PREFIX}{next_id}", prediction.to_json())
    store.set_state(_COUNT_KEY, str(next_id))
    return next_id


def iter_predictions(store: Store) -> list[tuple[int, Prediction]]:
    """Every recorded prediction with its 1-based log index, oldest first.

    The index is the stable identity :mod:`~src.agents.grader` (T2.1) keys a grade against
    (``grade@<same index>``) — predictions are an append-only log with no other identifier, so
    "which prediction does this grade resolve" has to be this position, not e.g. equality on
    the `Prediction` value itself (two genuinely different predictions could coincidentally
    have identical fields).
    """
    count = _parse_count(store.get_state(_COUNT_KEY))
    result = []
    for i in range(1, count + 1):
        stored = store.get_state(f"{_PREDICTION_KEY_PREFIX}{i}")
        if stored is not None:
            result.append((i, Prediction.from_json(stored)))
    return result


def list_predictions(store: Store) -> list[Prediction]:
    """Return every recorded prediction, oldest first."""
    return [prediction for _, prediction in iter_predictions(store)]


def forecast_store(store: Store, *, now: datetime | None = None) -> list[Prediction]:
    """Forecast every classified item that doesn't already have a recorded prediction.

    "Delta" here is by evidence, not an item-record field (see the module docstring on why):
    an item is skipped once any of its own URLs appears in an existing prediction's evidence.
    An item with no ``url`` is skipped entirely — it can't carry real evidence (the evidence
    principle), and matching it against `already_forecast` by absence would either wrongly
    treat it as forecast-already or re-forecast it forever, depending on how "missing" and
    "empty" compare. A failing item (``llm.LLMError``/``ForecastError``) is skipped and
    logged rather than losing every other item's already-produced prediction in the same run.
    `now` fixes the "made at" timestamp for every prediction in this run (see
    :func:`forecast_item`).

    Returns the newly-recorded predictions (``[]`` if there was nothing to do).
    """
    already_forecast = {url for p in list_predictions(store) for url in p.evidence}
    pending = [
        item
        for item in store.query()
        if item.get("category") and item.get("url") and item["url"] not in already_forecast
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
