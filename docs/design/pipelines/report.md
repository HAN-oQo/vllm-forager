# Pipeline: `report` (weekly cited digest + taxonomy tree)

> Entry: `python -m src.report [--v0 | --tree]` → `src.report.generate` / `generate_tree`
> Code: `src/report.py` (CLI), `src/agents/reporter_v1.py` (default + tree), `src/agents/reporter.py`
> (v0 keyword fallback + shared `_cite`), `src/agents/summarizer.py` (node summaries), `src/trends.py`
> (`week_stamp`)

Turns the classified KB into a dated, **fully cited** Markdown digest on disk. Three renderers:
the default LLM-written per-category digest (v1), an opt-in taxonomy-**tree** digest (`--tree`),
and an offline keyword-only fallback (`--v0`). Every line carries a source link (evidence
principle).

## Where it sits
```
analyze (→ summarizer for tree nodes) → REPORT → data/reports/<ISO-week>.md | .tree.md + .tree.json
```

## Logic, step by step

### Default (v1) — `reporter_v1.build_report_v1`
1. Group items by their flat `category` (first-occurrence order; `Other` last; items with no
   category counted and surfaced with a note pointing at `python -m src.analyze`).
2. Per category, **batch** items into chunks of `_CHUNK_SIZE = 25` and ask
   `llm.complete(_CLAIMS_SCHEMA)` for a one-sentence claim per item (`item_index`-aligned,
   `_BODY_CHARS = 500` each). Chunking bounds context/output/argv size and confines a chunk
   failure (caught broadly, incl. bare `OSError`) to that chunk.
3. Render every item through `reporter._cite` — LLM claim where covered, plain title where not
   (completeness guarantee) — with whitespace collapsed and a URL-or-synthesized citation.

### Tree (`--tree`) — `reporter_v1.tree_from_store` → `build_tree` + `render_tree_markdown`
1. Group items by their classified **`path`** (not flat category); unclassified/`OTHER` → one flat
   `Other` root. `count` rolls up each subtree; empty branches can't exist (structural).
2. Attach each node's **summary** via `summarizer.get_node_summary` — read from the KB, where a
   separate `summarize_store` pass wrote a 1–2 line cited synthesis per node (its own batched LLM
   call, capped `_MAX_ITEMS_PER_NODE = 25`). Building the tree itself makes **no** LLM call.
3. Render `대 → 소(summary) → 소소 → PRs` Markdown (heading per depth to H6, then bold) + write
   `<week>.tree.json` (`{name, summary, count, gaps, children, prs}`) for the dashboard. `gaps` is
   a placeholder 0 until parity data is wired in.

### v0 (`--v0`) — `reporter.build_report`
Fixed ordered `(category, keyword-regex)` taxonomy, first-match-wins, **no LLM** — the offline
fallback for an LLM outage/cost/bad-output incident.

## KB writes / outputs
Writes files under `data/reports/`. Reads item records + `node_summary@*` state. The dashboard
recomputes the tree live from the Store (never reads the stale `.tree.json`).

## Cost shape
v1: `ceil(items_in_category / 25)` calls per category — i.e. **re-summarizes essentially the whole
KB on every run**. Tree summaries: one call per taxonomy node per `summarize_store` run, also full
recompute. v0: zero.

## Known limitations (from the code)
- **No delta / no cache**: a "weekly" report re-LLMs every item's claim every run; cost grows with
  KB size and repeats work for unchanged items.
- Node summaries are **recomputed every run** with no skip-if-unchanged cache, and each node
  re-reads *raw items* (call count ≈ items × avg depth) rather than rolling up child summaries.
- v1 groups by flat `category`, not `path` (only the tree uses `path`); trends are computed
  separately and never woven into the narrative.

## 고도화 (advancement)
1. **Delta digest + claim cache** *(highest leverage).* Cache each item's claim keyed by a content
   hash; a run re-LLMs only items new/changed since last week and reuses the rest. This turns
   "re-summarize the whole KB weekly" into "summarize the delta", and naturally reframes the report
   as **"what changed this week"** (new items, state flips, newly-closed gaps) instead of a full
   dump.
2. **Hierarchical (map-reduce) summaries.** Synthesize a parent node from its **child summaries**,
   not from re-read raw items — the standard long-document summarization pattern. Fixes the
   documented O(items × depth) cost and makes ancestor summaries more coherent.
3. **Editorial executive summary.** Open with a synthesized top-line built from the *other* stages —
   trend inflections (`trends.py`), newly-opened/closed parity gaps (`parity.py`), top-ranked
   candidates (`scout`), and notable predictions/their resolutions (`forecast`+grader). One
   integrated digest, not four disconnected CLIs. Include **roadmap deltas** where a target repo's
   roadmap is tracked (see `candidates.md`): "this landed / this is now unblocked / this slipped".
4. **Faithfulness guardrail before the human reads it** (the `T1.8` RAG-trust discipline applied to
   the report): NLI / LLM-judge each generated claim against its cited item; flag or drop
   unsupported claims, threshold it, track drift. A cited-but-hallucinated claim is worse than a
   plain title.
5. **Group v1 by `path`** (roll up by prefix) so the flat report and the tree agree, and so a
   growing taxonomy doesn't fragment into opaque full-path buckets.
