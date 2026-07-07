# Intelligence-plane pipelines — logic & advancement

Detailed, code-grounded write-ups of the four intelligence-plane CLIs, each with its own
`고도화 (advancement)` section:

| Stage | Entry | What it produces | Doc |
|---|---|---|---|
| **analyze** | `python -m src.analyze` | items classified into the versioned taxonomy tree | [analyze.md](analyze.md) |
| **forecast** | `python -m src.forecast` | append-only log of calibrated predictions | [forecast.md](forecast.md) |
| **report** | `python -m src.report [--tree\|--v0]` | dated, cited weekly digest (+ taxonomy tree) | [report.md](report.md) |
| **candidates** | `python -m src.candidates` | ranked contribution queue (risk/effort/impact) | [candidates.md](candidates.md) |

Pipeline order: `collect → analyze → (summarizer) → forecast · report · trends · parity → candidates`.
All four select a store via the shared `resolve_store(--data-dir)` contract and isolate per-item
LLM failures (skip + log, never abort the batch).

## Cross-cutting 고도화 (read this first)

The per-stage docs list local improvements; these five themes cut across all four and are where
the real leverage is.

**A. Delta + cache everywhere.** Only `analyze` processes a delta today — `forecast`, `report`, and
`candidates` all recompute from the **whole KB every run**, so cost grows with KB size and repeats
work on unchanged items. Content-hash-keyed caching + incremental processing is the highest-leverage
*cost* fix across the plane (claims in report, scores in scout, predictions in forecast).

**B. Close the self-improvement loop into `forecast` and `candidates`.** The product's thesis is
"grade your own judgment and evolve." Today grading feeds **taxonomy + policy** only. It does *not*
feed the **forecaster's calibration** (probabilities stay overconfident) or the **scout's ranking**
(hand-tuned weights). Wiring realized outcomes back into those two is the advancement most aligned
with what makes this project different from a dashboard.

**C. Cost cascade: rules → embeddings → local vLLM → Claude.** The same pattern applies to every
LLM call in the plane — classification (`analyze`), per-item claims (`report`), risk/effort/impact
(`scout`), and *which* items to forecast (`forecast`). Cheap-first + escalate-the-tail + distill.
Tracks `docs/IDEAS.md` "Cheap issue classification" and DEVPLAN `T4.11`.

**D. Trust guardrails before the human gate.** Each stage should self-check before a human spends
attention: **faithfulness** on report claims (the `T1.8` RAG-trust discipline), **calibration** on
forecasts, **novelty** on candidates (`novelty.py`, built but unwired). "Trust the numbers" means
the numbers are guarded.

**E. Align to each target repo's own roadmap — and only what we can verify.** *(owner request)*
The strongest contribution signal is what a repo **says it wants**: vLLM's roadmap issues / RFCs,
**vllm-omni's ROCm roadmap**, milestones, "help wanted". Make **roadmap-derived** a first-class
candidate source (high merge probability → a priority boost), gated by **feasibility on our own
oracle** — only surface roadmap items an MI250 (gfx90a) can actually build + verify, so a candidate
that reaches the human gate is one we can really carry to a PR. See [candidates.md](candidates.md)
§2–3.

## Suggested sequencing (if promoted to work)
1. **Now / cheap:** batch + cache the LLM calls (C, A) — immediate cost relief, no architecture change.
2. **Next:** wire `novelty` + persist queue state (candidates D/A); roadmap source + feasibility gate (E).
3. **Then (the differentiator):** calibration loop into forecast, learned ranking into scout (B) —
   needs a few graded outcomes first, so it lands after the loop has run end-to-end.

These are analysis notes, not committed todos — promote the chosen ones into `docs/IDEAS.md`
(shaping) or `docs/DEVPLAN.md` (a numbered, test-backed todo) before building.
