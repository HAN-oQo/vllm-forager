# Pipeline: `candidates` (contribution queue — the Scout)

> Entry: `python -m src.candidates` → `src.agents.scout.discover_from_store(store)`
> Code: `src/candidates.py` (CLI), `src/agents/scout.py` (logic), `src/parity.py` (gap source),
> `src/novelty.py` (dedup — built, **not yet wired**)

M2's *"where can I contribute?"* output — turns parity gaps and raw GitHub signal into one ranked
queue a human (or M3's Engineer) works top-to-bottom. Each candidate is scored on three
independent dimensions (risk / effort / impact) plus deterministic boosts.

## Where it sits
```
analyze → parity.build_matrix → PARITY GAPS ┐
                     good-first-issue label ─┼→ SCOUT (score + boost + rank) → ranked queue → human pick / M3 Engineer
                     ROCm-reproducible issue ┘
```

## Logic, step by step
1. **Gaps** — `parity.find_gaps_for_all_targets(build_matrix(items))`: a capability (= flattened
   `category`) *shipped* (merged PR) on any tracked engine but **not** on a `"primary"` target
   (vllm / vllm-omni / vime), domain-scoped so an rl capability isn't flagged against a speech
   target. A `ParityError` (no primary configured) degrades to "no gaps", not a crash.
2. **Three sources**, in `discover_candidates`:
   - **parity-gap** — skipped if it has no resolvable evidence (evidence principle).
   - **good-first-issue** — open issue whose labels (hyphen/underscore normalized) contain
     "good first issue".
   - **ROCm-reproducible** — open issue that `reporter.categorize` files under `"ROCm / AMD"`.
   Open **issues only** for the latter two; within-run dedup by `(repo, number)`; an item matching
   two sources is scored once (first source wins).
3. **Score** — one `llm.complete(_SCORE_SCHEMA)` per candidate for `risk`/`effort`/`impact`, each
   `low`/`medium`/`high` (title + 2000-char body). A failed/malformed score skips that candidate.
4. **Deterministic boosts** (`_boost_for`, no LLM — cheap, offline-testable):
   - **ROCm∩speech** (+2): repo domain `speech` **and** text matches both `ROCM_HINTS` and
     `SPEECH_HINTS`.
   - **Edge-applicability** (+1): domain ∈ {speech, omni, engine} (not `rl`, not untracked).
   - **Merge-velocity** (+1): repo merges at/faster than the run's median repo.
5. **Rank** — `priority = (3·impact − risk − effort) · 10 + boost`, sorted desc. The `·10` scaling
   keeps `boost` a pure tiebreak among identical risk/effort/impact combos (never overturns a
   genuine difference).

## KB writes
None — the CLI prints the queue. Nothing is persisted (no memory of picks/attempts).

## Cost shape
**One `llm.complete` per candidate**, recomputed from scratch every run (four passes over the
items). Cost scales with the number of open ROCm-relevant/good-first issues, **not** with what's
new since the last run.

## Known limitations (from the code)
- **Recompute-every-run, no persistence, no dedup.** `novelty.py` (embedding + LLM-judge dedup,
  `T2.7`) is fully built but **not wired in** — so the same candidates are re-ranked (and would be
  re-attempted) forever.
- Parity capability match is the **exact flattened `category` string**; two items at different path
  depths for the same real capability → false-positive/false-negative gaps.
- "ROCm-reproducible" (word-boundary title+labels+body) is inconsistent with `stats.py`'s own
  looser labels-only notion of "ROCm relevant".
- First-source-wins evidence for a gap is a REPOS-order artifact, not a judgment.

## 고도화 (advancement)
1. **Wire in novelty + persist queue state** *(the single biggest sample-efficiency lever —
   ShinkaEvolve, already argued in `novelty.py`'s own docstring).* Dedup each candidate against a
   **prior-attempt log** before ranking/attempting, and persist what a human picked / what was
   tried, so the queue is **incremental** (rank only what's new) instead of recomputed. This also
   makes cost scale with the delta, not the open-issue count.
2. **Reference each repo's own roadmap as a first-class source** *(user request).* Target repos
   publish intent — vLLM's quarterly roadmap issues / RFCs, **vllm-omni's ROCm roadmap**
   (`config.py` already notes it), milestones, "help wanted". Add a **roadmap-derived** source:
   detect roadmap/RFC/milestone items (label or pinned-issue or a `ROADMAP.md` parse) among already
   collected items, and surface each unstarted line that matches our domain — especially
   **ROCm∩speech** — as a candidate. An officially-wanted item has a **high merge probability**, so
   give it a roadmap boost (+1/+2) alongside the existing ones. **Sibling corpus:** our *own* org's
   Confluence/Jira roadmaps (first-party demand) — same machinery, different source — are captured in
   `docs/IDEAS.md` → "Internal-demand-driven contributions" (e.g. an internal agentic-gateway need for session
   affinity → infer a vLLM session-id/routing feature); privacy-gated, so deferred to IDEAS until settled.
3. **Feasibility gate — only surface what we can actually build + verify** *(the "그거 구현이 가능하다면"
   part).* A roadmap item is only a candidate if it's implementable on **our** oracle: cross-check
   against MI250 capability (gfx90a), a bounded effort estimate, and dependency availability
   (does it need hardware/features we don't have?). Score each roadmap/gap candidate for
   **feasibility** and drop or down-rank the infeasible — a candidate we can't verify can't reach
   the human gate anyway. (This reuses M3's Engineer/`runner.py` verification oracle as a
   pre-filter, not just a post-hoc check.)
4. **Learn the ranking from outcomes** — replace the hand-tuned `3·impact − risk − effort` weights
   with a model predicting `P(merge)` / realized value from features (source, domain, boosts,
   scores, roadmap-endorsed?) once contribution outcomes exist. Closes the RSI loop for *selection*
   the way calibration closes it for forecasting; the cost-aware `llm_bandit.py` is a natural fit.
5. **Path-prefix parity matching** — compare capabilities by rolling up the taxonomy `path` prefix,
   not the exact flattened string (fixes the documented false gaps; shared fix with `trends`).
6. **Confidence + evidence on scores, and a cheap pre-filter.** Have the scorer cite *why* (which
   part of the issue implies high effort) and abstain when unsure — feeds the attempt report
   (`T3.12`). Deterministically pre-screen (boosts + rules), then spend the LLM only on the top-K.
7. **Richer sources** — perf-regression (a trend reversal in `trends.py`), high-engagement-but-stale
   issues, CI-failure clusters; and reconcile the two "ROCm-relevant" definitions.
