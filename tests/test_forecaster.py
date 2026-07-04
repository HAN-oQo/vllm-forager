"""Tests for the Forecaster agent (T1.5) — offline & deterministic.

Per the DEVPLAN todo: mock ``llm.complete`` → stored prediction validates against schema
(``prob`` in ``[0, 1]``, ``due_date`` after ``created_at``). Also covers the validation
failure paths, the evidence-based delta filter, per-item failure isolation, and the KB
round-trip.
"""

from datetime import datetime, timezone

import pytest

from src import llm
from src.agents import forecaster
from src.store.jsonl_store import JsonlStore

pytestmark = pytest.mark.m1

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _item(repo: str, number: int, title: str, **overrides) -> dict:
    rec = {
        "repo": repo,
        "number": number,
        "type": "issue",
        "title": title,
        "state": "open",
        "labels": [],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "url": f"http://x/{repo}/{number}",
        "body": "",
    }
    rec.update(overrides)
    return rec


def _reply(**overrides) -> dict:
    base = {
        "claim": "this will be merged",
        "resolution_rule": "resolved true if the PR is merged",
        "prob": 0.7,
        "due_date": "2026-02-01T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_forecast_item_valid_reply_produces_stored_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())
    item = _item("o/r", 1, "hipBLAS build fails on gfx90a")

    prediction = forecaster.forecast_item(item, now=_NOW)

    assert prediction.claim == "this will be merged"
    assert 0.0 <= prediction.prob <= 1.0
    assert prediction.due_date == "2026-02-01T00:00:00Z"
    assert prediction.created_at == "2026-01-01T00:00:00Z"
    assert prediction.evidence == (item["url"],)  # evidence citation


@pytest.mark.parametrize("bad_prob", [-0.1, 1.1, 2.0])
def test_forecast_item_prob_out_of_range_raises(
    monkeypatch: pytest.MonkeyPatch, bad_prob: float
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply(prob=bad_prob))
    with pytest.raises(forecaster.ForecastError, match=r"prob must be in \[0, 1\]"):
        forecaster.forecast_item(_item("o/r", 1, "x"), now=_NOW)


def test_forecast_item_due_date_not_after_created_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply(due_date="2025-06-01T00:00:00Z"))
    with pytest.raises(forecaster.ForecastError, match="must be after"):
        forecaster.forecast_item(_item("o/r", 1, "x"), now=_NOW)


def test_forecast_item_malformed_due_date_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply(due_date="not-a-date"))
    with pytest.raises(forecaster.ForecastError, match="not a valid"):
        forecaster.forecast_item(_item("o/r", 1, "x"), now=_NOW)


def test_forecast_item_missing_field_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = _reply()
    del reply["claim"]
    monkeypatch.setattr(llm, "complete", lambda *a, **k: reply)
    with pytest.raises(forecaster.ForecastError, match="missing required field"):
        forecaster.forecast_item(_item("o/r", 1, "x"), now=_NOW)


def test_forecast_item_non_dict_reply_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "not json")
    with pytest.raises(forecaster.ForecastError, match="expected a JSON object"):
        forecaster.forecast_item(_item("o/r", 1, "x"), now=_NOW)


@pytest.mark.parametrize(
    "due_date",
    [
        "2026-02-01T00:00:00Z",
        "2026-02-01T00:00:00.123Z",
        "2026-02-01T00:00:00+00:00",
        "2026-02-01T00:00:00.123456+0000",
    ],
)
def test_forecast_item_tolerates_iso8601_variants(
    monkeypatch: pytest.MonkeyPatch, due_date: str
) -> None:
    """Regression: a plain strptime used to reject anything but the exact TS_FORMAT string,
    which real (non-mocked) LLM replies routinely deviate from (millis, "+00:00" vs "Z")."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply(due_date=due_date))
    prediction = forecaster.forecast_item(_item("o/r", 1, "x"), now=_NOW)
    assert prediction.due_date == due_date


def test_forecast_item_rejects_non_utc_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero offset is rejected rather than silently misread as UTC."""
    monkeypatch.setattr(
        llm, "complete", lambda *a, **k: _reply(due_date="2026-02-01T00:00:00+05:00")
    )
    with pytest.raises(forecaster.ForecastError, match="not a valid UTC timestamp"):
        forecaster.forecast_item(_item("o/r", 1, "x"), now=_NOW)


def test_prediction_from_json_malformed_raises() -> None:
    with pytest.raises(forecaster.ForecastError, match="corrupt prediction record"):
        forecaster.Prediction.from_json("not valid json")


def test_prediction_from_json_missing_field_raises() -> None:
    with pytest.raises(forecaster.ForecastError, match="corrupt prediction record"):
        forecaster.Prediction.from_json('{"claim": "c"}')


def test_prediction_json_roundtrip() -> None:
    original = forecaster.Prediction(
        claim="c",
        resolution_rule="r",
        prob=0.5,
        due_date="2026-02-01T00:00:00Z",
        evidence=("http://x/1",),
        created_at="2026-01-01T00:00:00Z",
    )
    assert forecaster.Prediction.from_json(original.to_json()) == original


def test_record_and_list_predictions_roundtrip(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    p1 = forecaster.Prediction(
        claim="a",
        resolution_rule="r",
        prob=0.1,
        due_date="2026-02-01T00:00:00Z",
        evidence=("http://x/1",),
        created_at="2026-01-01T00:00:00Z",
    )
    p2 = forecaster.Prediction(
        claim="b",
        resolution_rule="r",
        prob=0.9,
        due_date="2026-03-01T00:00:00Z",
        evidence=("http://x/2",),
        created_at="2026-01-01T00:00:00Z",
    )
    assert forecaster.record_prediction(store, p1) == 1
    assert forecaster.record_prediction(store, p2) == 2
    assert forecaster.list_predictions(store) == [p1, p2]


def test_forecast_store_skips_uncategorized_and_already_forecast_items(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    already = forecaster.Prediction(
        claim="c",
        resolution_rule="r",
        prob=0.5,
        due_date="2026-02-01T00:00:00Z",
        evidence=("http://x/o/r/2",),
        created_at="2026-01-01T00:00:00Z",
    )
    forecaster.record_prediction(store, already)
    store.upsert_items(
        [
            _item("o/r", 1, "uncategorized"),  # no category yet -> skipped
            _item("o/r", 2, "already forecast", category="rocm-build"),  # skipped
            _item("o/r", 3, "ready to forecast", category="rocm-build"),
        ]
    )
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())

    predictions = forecaster.forecast_store(store, now=_NOW)

    assert len(predictions) == 1
    assert predictions[0].evidence == ("http://x/o/r/3",)
    assert len(forecaster.list_predictions(store)) == 2  # the pre-existing one + this one


def test_forecast_store_skips_failing_item_and_persists_the_rest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlStore(tmp_path)
    store.upsert_items(
        [
            _item("o/r", 1, "good item", category="rocm-build"),
            _item("o/r", 2, "bad item", category="rocm-build"),
        ]
    )

    def flaky_complete(prompt: str, **kwargs) -> dict:
        if "bad item" in prompt:
            raise llm.LLMError("simulated transient failure")
        return _reply()

    monkeypatch.setattr(llm, "complete", flaky_complete)

    predictions = forecaster.forecast_store(store, now=_NOW)

    assert len(predictions) == 1
    assert predictions[0].evidence == ("http://x/o/r/1",)


def test_forecast_store_no_pending_returns_empty(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    assert forecaster.forecast_store(store) == []


def test_forecast_store_skips_items_with_no_url(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a url-less item used to be re-forecast forever (None never matched the
    stored "" evidence) and would have persisted an empty-string, non-citing evidence."""
    store = JsonlStore(tmp_path)
    item = _item("o/r", 1, "no url", category="rocm-build")
    del item["url"]
    store.upsert_items([item])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: _reply())

    assert forecaster.forecast_store(store, now=_NOW) == []
    assert forecaster.list_predictions(store) == []
    # a second run doesn't behave any differently — it's not a transient "not yet" state
    assert forecaster.forecast_store(store, now=_NOW) == []


def test_record_prediction_corrupt_count_raises(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.set_state("prediction_count", "not-a-number")
    p = forecaster.Prediction(
        claim="c",
        resolution_rule="r",
        prob=0.5,
        due_date="2026-02-01T00:00:00Z",
        evidence=("http://x/1",),
        created_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(forecaster.ForecastError, match="corrupt prediction_count"):
        forecaster.record_prediction(store, p)


def test_list_predictions_corrupt_count_raises(tmp_path) -> None:
    store = JsonlStore(tmp_path)
    store.set_state("prediction_count", "not-a-number")
    with pytest.raises(forecaster.ForecastError, match="corrupt prediction_count"):
        forecaster.list_predictions(store)
