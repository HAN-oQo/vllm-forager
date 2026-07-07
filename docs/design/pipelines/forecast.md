# Pipeline: `forecast` (calibrated predictions)

> Entry: `python -m src.forecast` → `src.agents.forecaster.forecast_store(store)`
> Code: `src/forecast.py` (CLI), `src/agents/forecaster.py` (logic)

The other half of the self-evolution loop's "grade your own judgment" thesis (CONTEXT decision
6): for each classified item, log a **falsifiable, timestamped, calibrated prediction now**, so a
later grading pass (`T2.1`) can score how well-calibrated this pipeline's judgment actually is.

## Where it sits
```
analyze → [classified items] → FORECAST → [append-only prediction log] → grade (T2.1) → policy/taxonomy self-evolution
```

## Logic, step by step
1. **Store select** — same `resolve_store` contract as `analyze`.
2. **Delta by evidence** — `forecast_store` builds `already_forecast = {every URL in every existing
   prediction's evidence}`, then selects items with **`category` set AND a `url` AND that url not
   already forecast**. An item with no `url` is skipped entirely (can't carry real evidence).
3. **Per item, `forecast_item`** — one `llm.complete(prompt, _PREDICTION_SCHEMA)` asking for a
   `claim`, a `resolution_rule` (exactly how to check it later), a `prob` (0–1), and a `due_date`
   (UTC, after "now"). Body truncated to 2000 chars.
4. **Strict construction** — the reply is built into a frozen `Prediction`, validated at
   construction: `prob ∈ [0,1]`, `due_date`/`created_at` parse as UTC, `due_date > created_at`. A
   malformed reply is a **loud `ForecastError`, not a silent fallback** — a bad prediction would
   corrupt the grading signal.
5. **Append** — `record_prediction` writes `prediction@N` + bumps `prediction_count` (an
   append-only log; the 1-based index is the stable identity the grader keys `grade@N` against).
6. **Failure isolation** — a per-item `llm.LLMError`/`ForecastError` is logged and skipped; other
   items' predictions in the run are preserved.

## KB writes
`prediction@N` (JSON: claim, resolution_rule, prob, due_date, evidence, created_at) + a
`prediction_count` index, in the state map. Predictions are **history** (many per item over time),
so they live in a log, not on the item record.

## Cost shape
**One `llm.complete` per classified-but-not-yet-forecast item.** Because it forecasts *every*
classified item, volume ≈ classification volume.

## Known limitations (from the code)
- The log **grows without bound** (one entry per item); `get_state`/`set_state` reload/rewrite the
  *entire* state map per call, so `list_predictions` (a full read every run) and `record_prediction`
  get slower over time; neither backend can filter by `due_date`/status server-side.
- `record_prediction`'s read-then-write has **no locking** (a race, worse here than
  taxonomy/policy since predictions are written continuously).
- The real fix is a Store-level primitive for a large, queryable record collection.

## 고도화 (advancement)
1. **Forecast selectively, not everything** *(biggest cost + value lever).* Most items (a routine
   good-first-issue) aren't worth a prediction. Gate on an importance signal — trend momentum
   (`trends.py`), engagement, ROCm/speech relevance, or the scout's own boosts — and only forecast
   the forecast-*worthy* tail. Cuts volume dramatically and raises the average prediction's value.
2. **Close the calibration loop — the project's actual thesis.** The grader (`T2.1`) produces
   realized outcomes but nothing feeds them *back into the forecaster*. Add it: compute per-category
   **Brier / log-loss**, then either (a) post-hoc **calibrate** raw LLM probabilities (Platt /
   isotonic on historical `prob → realized`), or (b) inject calibration feedback into the prompt
   ("recently you said 0.8 but only 55% resolved true"). LLMs are systematically overconfident;
   this makes `prob` mean something and *is* the self-improvement the product promises.
3. **Auto-resolution from GitHub state.** Many `resolution_rule`s ("merged by <date>") are
   checkable from data the collector already has (state/merged/labels). Auto-grade those instead of
   waiting for a manual pass — closes the loop in hours, not weeks, and lets calibration converge.
4. **Reference-class / base-rate prior.** Compute a cheap historical merge-rate per repo/label as a
   prior the LLM adjusts (reference-class forecasting). A prediction anchored to a base rate is far
   better calibrated than a cold LLM guess.
5. **Dedicated queryable store** for the log (the documented real fix) — enables "predictions due
   this week", status filters, and drops the full-map rewrite race.
6. **Update, don't one-shot.** Re-forecast an item as it materially evolves (new comments, label
   changes) rather than freezing the first prediction forever — keyed by item content hash so
   unchanged items cost nothing.
