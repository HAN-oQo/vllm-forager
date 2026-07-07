# Pipeline: `analyze` (classification)

> Entry: `python -m src.analyze` → `src.agents.analyst.analyze_store(store)`
> Code: `src/analyze.py` (CLI), `src/agents/analyst.py` (logic), `src/taxonomy.py` (the tree it validates against)

The first LLM-backed stage of the intelligence plane. It buckets every newly collected
issue/PR into the KB's **current, versioned taxonomy tree**, writing the label back onto the
item's own record so every later stage (report, trends, parity, scout) reads one record per
item, not a join.

## Where it sits
```
collector → [items in KB, no `path`] → ANALYZE → [items with `path`+`category`] → forecast · report · trends · parity · scout
```

## Logic, step by step
1. **Store select** — `resolve_store(--data-dir)`: explicit `--data-dir` reads a `JsonlStore`;
   otherwise `get_store()` picks the backend from `STORE` (jsonl/firestore).
2. **Delta** — `analyze_store` selects only items with **no `path` key** (`"path" not in item`).
   A rerun therefore touches only what an earlier run / the latest collection hasn't labeled;
   items classified by the pre-T1.5.2 flat-`category` analyst get re-classified once to backfill
   `path`. If nothing is pending → returns `[]` (no taxonomy load, no LLM).
3. **Active taxonomy** — `get_active(store)`; raises `TaxonomyError` if items are pending but no
   taxonomy exists yet.
4. **Per item, `classify_item(item, taxonomy)`** — deterministic **domain rooting** first (T3.15):
   - `domain = _domain_of_item(item)` reads the repo's `config.REPOS` `domain`
     (`speech`/`rl`/`omni`/`engine`) via `parity._domain_of`. The domain is a **known KB fact,
     never guessed by the model**.
   - `domain is None` (untracked/retired repo, e.g. `ROCm/vllm`) → fall back to a depth-1
     `(OTHER,)` path, **no LLM call**.
   - `taxonomy.children((domain,))` empty (domain not seeded yet) → classify as just `(domain,)`,
     **no LLM call** ("don't ask when there's nothing to validate the reply against").
   - Otherwise → **one** `llm.complete(_path_prompt, _PATH_SCHEMA)` returning a whole path *beneath*
     the domain in a single call; `_validated_subpath` keeps the **longest prefix that validates**
     level-by-level against `taxonomy.children(...)` (casefold match, stops at first off-taxonomy
     level — never rejects the whole item). Final path = `(domain, *validated_sublevels)`.
5. **Write** — `_classified_record` adds `path` (list), `category` (path flattened by
   `LEVEL_SEPARATOR`, for the flat-label consumers), and `taxonomy_version`; `store.upsert_items`
   **merges** these onto the record (T1.10) so a later collector re-fetch can't wipe them.
6. **Failure isolation** — a per-item `llm.LLMError` is logged to stderr and skipped (item stays
   pending for next run); every other item already classified in the batch is still written.

## KB writes
`path`, `category`, `taxonomy_version` (merged onto each item record). No separate table.

## Cost shape
**One `llm.complete` per not-yet-classified item** (skipping the two no-LLM fallbacks). Body
truncated to 2000 chars. Cost scales with *new* items per run (good — it deltas).

## Known limitations (from the code)
- **Guess-then-validate**: the model never sees the exact per-level `children()` options; it
  guesses a whole path and `_validated_subpath` prunes post-hoc. Fine at today's tiny taxonomy;
  weaker as the tree grows.
- `_path_prompt` lists every known subpath under a domain as one unbounded comma-joined hint.
- **Curator coupling**: a domain-only item is `(domain,)`, not `OTHER`. Curator's
  new-category discovery clusters `category == OTHER`, so until a human seeds one subcategory per
  domain, discovery is silently disabled for that domain.
- No reserved-domain enforcement on the write side (`add_category` can create an unreachable root).
- Downstream, `trends` buckets by the *full flattened path*, not rolled up by prefix.

## 고도화 (advancement)
1. **Zero-LLM first pass + confidence-gated cascade** *(highest leverage; see `docs/IDEAS.md` →
   "Cheap issue classification" and DEVPLAN `T4.11`).* Embeddings are already computed for RAG
   (`src/embed.py`) — classify most items with a **local classifier on those vectors** (centroid /
   kNN / SetFit) that returns a **confidence**, and only escalate low-confidence items to an LLM
   (local vLLM first, Claude for the hard tail). **Distill** Claude's labels back into the cheap
   classifier so its share shrinks over time. kNN/SetFit update by *adding examples*, so they
   track the **evolving taxonomy** without retraining — the key reason a frozen fine-tuned BERT is
   the wrong shape here.
2. **Constrained decoding instead of guess-then-validate.** Put the exact `children()` set for the
   next level into the JSON-schema as an `enum` (or use grammar-guided decoding — xgrammar/outlines
   — on local vLLM). The model can then only emit valid levels: `_validated_subpath`'s post-hoc
   pruning becomes unnecessary, drift → 0, and a much smaller/cheaper model becomes viable.
3. **Confidence + abstain.** Have the model (or the local classifier) return a confidence; below a
   threshold, stop the walk at the higher level rather than guessing deeper. This *is* the routing
   signal the cascade in (1) needs.
4. **Batch N items per call** (quick win, `T4.11`) — the single classify prompt is small; batching
   20–25 items amortizes the per-request overhead massively at the 24h cadence.
5. **Re-classify on `taxonomy@v` bump.** Today the delta is "no `path`", so items classified under
   an old taxonomy version are never revisited when the taxonomy self-evolves. Add a cheap
   re-embed-nearest pass on a version bump and only LLM the items whose nearest label actually
   changed — keeps labels fresh as the taxonomy grows, without a full re-LLM.
6. **Fix the curator coupling** — cluster per-domain (not only `OTHER`) so new-category discovery
   isn't silently disabled for a freshly seeded domain.
