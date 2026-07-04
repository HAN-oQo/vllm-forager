"""Grader agent (T2.1): resolve matured predictions against reality; score precision/recall +
Brier — the self-evolution signal T2.2's policy update reads from.

A prediction (T1.5, :mod:`~src.agents.forecaster`) is only useful once its ``due_date`` has
passed and someone checks whether it actually came true. This module is that check: for every
*matured* (due_date ≤ now), *not-yet-graded* prediction, it looks up the cited evidence
item(s) by parsing their GitHub URL back into ``(repo, number)`` (O(1) :meth:`Store.get_item`
lookups, not a full :meth:`Store.query` scan — the KB can be tens of thousands of items, per
the production dataset behind this project), then asks ``llm.complete`` to judge true/false
against that item's *current* state, since a free-text ``resolution_rule`` ("will merge by Q3")
isn't reducible to a fixed field comparison the way item state alone would be.

Grades are stored keyed by the **prediction's own log index** (``grade@<same index>``), a 1:1
correspondence rather than a separate incrementing counter — a grade only ever resolves one
specific prediction, so there's no independent identity for it to have. This also makes
"already graded" a single ``get_state`` check per prediction, with no separate delta-tracking
structure to keep in sync.

Aggregate metrics (:func:`compute_metrics`) threshold each prediction's ``prob`` at 0.5 to
call it "predicted true" or "predicted false", then compute standard precision/recall against
the resolved outcome, plus the Brier score (mean squared error between ``prob`` and the 0/1
outcome) — the calibration half precision/recall can't see (a run that's always confidently
wrong scores 0 precision either way, but only Brier penalizes overconfidence specifically).

Known limitation, not fixed here: like :mod:`~src.agents.forecaster`'s own prediction log,
grades add a second unbounded key range to the state map, with the same "no server-side
filtering, whole-map read/write" cost profile documented there — see that module's docstring.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import llm
from ..store.base import Store
from . import forecaster
from .forecaster import TS_FORMAT, Prediction, parse_ts

_GRADE_KEY_PREFIX = "grade@"

_RESOLUTION_SCHEMA = {
    "type": "object",
    "properties": {"outcome": {"type": "boolean"}},
    "required": ["outcome"],
}

# Matches both the `.../issues/N` and `.../pull/N` forms reporter.evidence_url ever produces
# (GitHub itself redirects between them) — this is the inverse of that function: URL back to
# (repo, number), so a grade can look its cited item up in O(1) via Store.get_item rather than
# scanning the whole store for a URL match.
_GITHUB_URL_RE = re.compile(r"^https://github\.com/([^/]+/[^/]+)/(?:issues|pull)/(\d+)$")


class GradeError(RuntimeError):
    """A prediction couldn't be resolved: a malformed LLM reply, or a corrupt KB grade record."""


def _parse_repo_number(url: str) -> tuple[str, int] | None:
    """`url` -> `(repo, number)`, or `None` if it isn't a recognized GitHub issue/PR URL."""
    match = _GITHUB_URL_RE.match(url.strip())
    if not match:
        return None
    return match.group(1), int(match.group(2))


def is_matured(prediction: Prediction, *, now: datetime | None = None) -> bool:
    """True once `now` has reached `prediction.due_date` — a prediction isn't gradable before
    the point it committed to being checkable by."""
    when = now or datetime.now(timezone.utc)
    return parse_ts(prediction.due_date) <= when


@dataclass(frozen=True)
class Grade:
    """The resolved outcome of one matured prediction, validated at construction time.

    Raises:
        GradeError: `graded_at` isn't a valid UTC timestamp (see `forecaster.parse_ts`).
    """

    prediction: Prediction
    outcome: bool
    graded_at: str

    def __post_init__(self) -> None:
        try:
            parse_ts(self.graded_at)
        except forecaster.ForecastError as exc:
            raise GradeError(str(exc)) from exc

    @property
    def brier_term(self) -> float:
        """This grade's contribution to the Brier score: squared error between the original
        `prob` and the actual 0/1 outcome."""
        return (self.prediction.prob - (1.0 if self.outcome else 0.0)) ** 2

    def to_json(self) -> str:
        """Serialize for storage in the KB's state map."""
        return json.dumps(
            {
                "prediction": json.loads(self.prediction.to_json()),
                "outcome": self.outcome,
                "graded_at": self.graded_at,
            }
        )

    @staticmethod
    def from_json(raw: str) -> Grade:
        """Deserialize a value previously produced by :meth:`to_json`.

        Raises:
            GradeError: `raw` isn't valid JSON, isn't shaped like a grade record, or its
                embedded prediction/timestamp fails its own validation.
        """
        try:
            data = json.loads(raw)
            return Grade(
                prediction=Prediction.from_json(json.dumps(data["prediction"])),
                outcome=bool(data["outcome"]),
                graded_at=data["graded_at"],
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise GradeError(f"corrupt grade record: {exc}") from exc
        except forecaster.ForecastError as exc:
            raise GradeError(f"corrupt grade record: {exc}") from exc


def _resolution_line(item: dict) -> str:
    return (
        f"{item.get('repo')}#{item.get('number')} [{item.get('state')}]: {item.get('title') or ''}"
    )


def _resolution_prompt(prediction: Prediction, items: list[dict]) -> str:
    listing = "\n\n".join(_resolution_line(item) for item in items) or (
        "(none of the cited item(s) could be found in the KB anymore)"
    )
    return (
        "A prior prediction was made about the GitHub issue/PR(s) below. Judge whether it "
        "resolved TRUE or FALSE, based on their CURRENT state.\n\n"
        f"Claim: {prediction.claim}\n"
        f"Resolution rule: {prediction.resolution_rule}\n\n"
        f"Current state of the cited item(s):\n{listing}\n\n"
        "Reply with `outcome`: true if the claim resolved true, false otherwise."
    )


def resolve_prediction(
    prediction: Prediction, store: Store, *, now: datetime | None = None
) -> Grade | None:
    """Resolve `prediction` against current reality, or `None` if it hasn't matured yet.

    Raises:
        llm.LLMError: the completion call failed (transport error, timeout, non-JSON reply).
        GradeError: the model's reply didn't include a boolean ``outcome``.
    """
    when = now or datetime.now(timezone.utc)
    if not is_matured(prediction, now=when):
        return None
    items = []
    for url in prediction.evidence:
        parsed = _parse_repo_number(url)
        if parsed is None:
            continue
        item = store.get_item(*parsed)
        if item is not None:
            items.append(item)
    reply = llm.complete(_resolution_prompt(prediction, items), json_schema=_RESOLUTION_SCHEMA)
    if not isinstance(reply, dict) or not isinstance(reply.get("outcome"), bool):
        raise GradeError(f"expected a boolean `outcome` reply, got {reply!r}")
    return Grade(
        prediction=prediction, outcome=reply["outcome"], graded_at=when.strftime(TS_FORMAT)
    )


def grade_store(store: Store, *, now: datetime | None = None) -> list[Grade]:
    """Resolve every matured, not-yet-graded prediction in `store`; write results back.

    A prediction not yet due, or already graded (``grade@<its index>`` already set), is
    skipped without touching the KB. A failing prediction (``llm.LLMError``/``GradeError``) is
    skipped and logged to stderr — left pending for the next run — rather than losing every
    other grade already produced in this batch.

    Returns the newly-recorded grades (``[]`` if nothing matured/pending).
    """
    when = now or datetime.now(timezone.utc)
    graded = []
    for index, prediction in forecaster.iter_predictions(store):
        key = f"{_GRADE_KEY_PREFIX}{index}"
        if store.get_state(key) is not None:
            continue
        try:
            grade = resolve_prediction(prediction, store, now=when)
        except (llm.LLMError, GradeError) as exc:
            print(f"grader: skipping prediction #{index}: {exc}", file=sys.stderr)
            continue
        if grade is None:
            continue
        store.set_state(key, grade.to_json())
        graded.append(grade)
    return graded


def list_grades(store: Store) -> list[Grade]:
    """Every recorded grade, in prediction order (oldest first)."""
    grades = []
    for index, _ in forecaster.iter_predictions(store):
        stored = store.get_state(f"{_GRADE_KEY_PREFIX}{index}")
        if stored is not None:
            grades.append(Grade.from_json(stored))
    return grades


@dataclass(frozen=True)
class GradeMetrics:
    """Aggregate calibration metrics over a set of grades."""

    precision: float
    recall: float
    brier: float
    n: int


def compute_metrics(grades: list[Grade], *, threshold: float = 0.5) -> GradeMetrics:
    """Precision/recall (each prediction's `prob` thresholded at `threshold` as "predicted
    true") plus the Brier score (mean squared error between `prob` and the actual outcome)
    over `grades`.

    `n=0` (no grades yet) returns all-zero metrics rather than raising — a fresh KB with no
    matured predictions yet is a normal, expected state, not an error condition.
    """
    n = len(grades)
    if n == 0:
        return GradeMetrics(precision=0.0, recall=0.0, brier=0.0, n=0)
    true_positive = false_positive = false_negative = 0
    brier_sum = 0.0
    for grade in grades:
        predicted_true = grade.prediction.prob >= threshold
        if predicted_true and grade.outcome:
            true_positive += 1
        elif predicted_true and not grade.outcome:
            false_positive += 1
        elif not predicted_true and grade.outcome:
            false_negative += 1
        brier_sum += grade.brier_term
    denom_p = true_positive + false_positive
    denom_r = true_positive + false_negative
    precision = true_positive / denom_p if denom_p else 0.0
    recall = true_positive / denom_r if denom_r else 0.0
    return GradeMetrics(precision=precision, recall=recall, brier=brier_sum / n, n=n)
