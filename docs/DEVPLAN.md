# DEVPLAN — Resumable Development Checklist

> **Purpose:** the single source of truth for *what to build next*. Every task is a checkbox with a named
> test. A fresh session (human or AI) can open this file, run the suite, find the first unchecked box, and
> continue — no other context required beyond `docs/CONTEXT.md` (decisions) and `docs/PLAN.md` (roadmap +
> architecture).

## Resume protocol — do this first, every new session

1. Read `docs/CONTEXT.md` (why) and `docs/PLAN.md` (architecture), then this file (what next).
2. Set up env once:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # activate! the tools/hooks live in here
   pip install -r requirements-dev.txt                 # runtime + test + lint/format/type deps
   pre-commit install                                  # install the git commit hook
   ```
3. Run the suite + checks — **everything already checked below must stay green**:
   ```bash
   pytest                       # full offline suite
   pytest -m m0                 # just one milestone's tests (markers: m0, m0_6, m1 … m5)
   pre-commit run --all-files   # black + ruff + mypy + hygiene (venv must be activated)
   ```
4. Find the **first unchecked `[ ]`** top-to-bottom — that's the next task. Read its milestone's
   **Expected output / Demo / Acceptance** block so you know what "done" looks like.
5. Work one task: **write code (with docstrings/comments) → write/adjust its named test → run its Demo command
   to watch it actually work → `pytest` + `pre-commit` green → check the box → commit.**
6. **HARD RULE: never check a box unless its named test passes** and `pre-commit` is clean. Partial work stays
   `[ ]` with a `> note:`. Integration tasks that can't be unit-tested (real PR, hardware) are checked only when
   the named evidence (a PR URL, a captured log) is pasted under the task.

## Conventions

- **Tests are offline & deterministic by default.** Mock network/subprocess (`monkeypatch`), use `tmp_path`.
  Anything hitting real GitHub / Firestore emulator / MI250 / a live LLM is marked `@pytest.mark.integration`
  and skipped in the default run (`pytest -m integration` to run explicitly).
- **Evidence principle:** every KB record and every report claim carries source issue/PR links. Tests assert this.
- **One test file per module:** `src/foo.py` → `tests/test_foo.py`; tag it with its milestone marker
  (`pytestmark = pytest.mark.m1`).
- **Runnable verification, not just unit tests.** Every milestone has a **Demo** command you can actually run to
  see the behavior, plus an **Acceptance** check (`pytest -m <milestone>`). Prefer driving the real thing over
  trusting green unit tests alone.
- **Readable code:** module + public-function **docstrings** (what/why), inline comments on non-obvious logic,
  and **type hints** on public functions. Match the comment density of the existing `src/collector.py`.
- **Todo format (human-readable):** under each todo put **Why** / **e.g.** / **Test** as their own sub-bullets
  (`  - **Why:** …`) — a one-line rationale, a concrete result example, and the test — so a person can scan it,
  not just an agent. Fill these in when the todo is concrete; don't fabricate examples for undesigned work.

## Python conventions & tooling

- **Format:** `black` (line length 100). **Lint + import-sort:** `ruff` (`E,W,F,I,UP,B,C4,SIM`). **Types:**
  `mypy` (lenient for now: `ignore_missing_imports`, annotations encouraged not forced). Config in `pyproject.toml`.
- **pre-commit** (`.pre-commit-config.yaml`) runs all of the above on commit. `pre-commit install` once, then
  `pre-commit run --all-files` to check everything. **Activate the venv first** — the hooks call the venv's tools.
- **Layout:** code in `src/` (one module per responsibility), agents in `src/agents/`, tests in `tests/`
  (`test_<module>.py`), the web UI in `dashboard/`.
- **Commit green:** `pytest` + `pre-commit` must pass before you check a box or commit.

## Guardrails (trust the numbers)

Two guardrails make the pipeline's outputs trustworthy. Treat them as first-class acceptance — their thresholds
gate merges, and their scores are written to the KB and shown on the dashboard (T5.7):

1. **Collection data-quality** (T0.10–T0.11): collect *without silent loss* — robust fetch (per-repo isolation,
   incremental state, retry/backoff, secondary-rate-limit) **plus reconciliation** against GitHub's own totals and
   an issue-number gap scan, surfaced as a `data_quality` metric.
2. **RAG trust score** (T1.8): a hand-labeled golden set + **retrieval** metrics (Recall@k / MRR / nDCG) and
   **groundedness** metrics (faithfulness / hallucination-rate), thresholded and tracked for drift — so a report's
   citations are numbers you can trust, not hope.

## Developing with Claude agents

Default to a **single main session** working `DEVPLAN.md` top-to-bottom — the checklist + `pytest`/`pre-commit`
are the "team" (manager = the plan, QA = the gates, developer = the session). This work is mostly sequential and
shares one context — the wrong shape for a standing multi-agent team (3–4× tokens + coordination overhead, and
against the project's sample-efficiency ethos).

Use **ephemeral subagents** (not teams) for the "fetch me a result" pattern that keeps the main context clean:
research fan-out (e.g. the RSI survey), a **reviewer pass** on the diff after each todo (`/code-review`), and
running tests / triaging failures (return only the summary).

**Graduate to an agent team only at M3/M5**, when work splits into cleanly separable directories (`dashboard/` vs
`src/agents/engineer.py` vs `src/store/`): git worktrees for isolation, 3–5 agents max, tight per-agent scope, a
validation step before merge. Keep the human review gate for anything upstream.

> Rationale + sources: Anthropic *When to use multi-agent systems* and *How we built our multi-agent research
> system*; Cognition *Don't Build Multi-Agents* (context engineering); Claude Code *agent-teams* docs.

## Git & PR workflow

- **Unit of PR = one todo** (or a small cluster of tightly-related tiny todos). A milestone is an *epic*
  (a GitHub Milestone / tracking issue), never a single PR.
- **Code-todo flow:** branch `t<id>-slug` → implement + its named test → open PR → **CI green**
  (`pytest` + `pre-commit`) + **`/code-review --comment`** (Claude posts its review as inline PR comments) →
  **a human merges** → check the box. Full rules: `docs/CONTRIBUTING.md`.
- **Direct to `main`:** pure docs / tooling / typo edits only.
- **Human-in-the-loop merge (required):** CI and `/code-review` are *gates, not approvers*. **Every merge to
  `main` is performed by a human.** Agents (subagents/teams) may open PRs and push to branches but **never
  self-merge**, and auto-merge stays off. This mirrors the upstream contribution rule inward.
- **Upstream vLLM PRs** (the product's output, T3.x) are a separate flow: fork → draft PR → **human gate** →
  upstream — never conflated with internal dev PRs.
- **Recommended branch protection on `main`:** require the CI status check to pass, require a PR (no direct agent
  pushes), block force-push. (Set once in GitHub → Settings → Branches.)
- CI = `.github/workflows/ci.yml`; a box is checked only when CI is green **and** a human has merged.

## Infrastructure (fixed)

- **`ce-master`** — CPU node. The agent runs here, long-running under **tmux**. Collection · KB · intelligence
  plane · orchestration all run on CPU.
- **`mi250-051` · `mi250-052` · `mi250-053`** — AMD MI250 (gfx90a) nodes. Used **only** for vLLM
  build / bug reproduction / patch verification (the M3 "oracle") and optionally hosting a local LLM (see below).
  Reached over ssh via these aliases.

## LLM backend — pluggable (three providers)

All reasoning goes through one wrapper, `src/llm.py`, selected by env `LLM_PROVIDER`:

| `LLM_PROVIDER` | Backend | Notes |
|---|---|---|
| `claude_cli` (default) | `claude -p` (Claude Code headless) | no API key mgmt; inherits local Claude Code auth |
| `claude_api` | Anthropic Messages API | needs `ANTHROPIC_API_KEY`; best for high-throughput/parallel |
| `local` | OpenAI-compatible endpoint (**vLLM** server) | `LLM_BASE_URL` + model; dogfoods vLLM, can run on MI250 |

The wrapper exposes one contract — `complete(prompt, *, system=None, json_schema=None) -> str | dict` — so
every agent is provider-agnostic. Unit tests mock `llm.complete`; a live smoke test per provider is `integration`.

---

## Status

- **Tooling (T0.0 ✅):** black + ruff + mypy + pre-commit wired; `pytest` green (9 tests, marker `m0`).
- **M0 data plane:** collector implemented + test-backed (T0.1–T0.5 ✅). Collection cadence = **24h / daily**
  (`config.COLLECT_INTERVAL_HOURS`). Next: **T0.6** (pluggable store).
- Everything below M0.6 is designed but unbuilt.

---

## M0 — Foundation & data plane

> **Expected output:** `data/<owner>__<repo>.jsonl` for all 5 repos + `data/state.json`; a v0 weekly Markdown
> report with cited links; pluggable store + LLM wrapper; full lint/format/type tooling.
> **Demo:** `python -m src.collector` → JSONL in `data/`; `python -m src.report` → `data/reports/YYYY-Www.md`.
> **Acceptance:** `pytest -m m0` green · `pre-commit run --all-files` clean · a collector run produces non-empty
> JSONL and a report whose lines carry issue/PR URLs.

- [x] **T0.0 Dev tooling** — black + ruff + mypy + pre-commit + pytest (`pyproject.toml`, `.pre-commit-config.yaml`, `pytest.ini`, `requirements-dev.txt`).
  - **Why:** enforce one formatting/lint/type/test standard across every session + CI, so nothing drifts.
  - **e.g.:** `pre-commit run --all-files` → black/ruff/mypy/hygiene all green in one shot.
  - **Test:** `pre-commit run --all-files` clean · `pytest` green.
- [x] **T0.1 Collector: incremental GitHub issue/PR fetch** — `src/collector.py::fetch_repo`.
  - **Why:** pull issues+PRs updated only since the last run (not the whole history) — saves time + rate limit.
  - **e.g.:** `fetch_repo("vllm-project/vllm", "2026-01-01T…")` → normalized issue/PR dicts updated since then.
  - **Test:** `tests/test_collector.py::test_fetch_repo_pagination`, `::test_fetch_repo_404` — mocks `requests`; paging stops at `< PER_PAGE`, 404 → `[]`.
- [x] **T0.2 Normalization schema** — `src/collector.py::_normalize`.
  - **Why:** flatten GitHub's varied payloads into one minimal schema so later stages consume a consistent shape.
  - **e.g.:** raw issue → `{repo, number, type:"issue|pr", title, labels, url, updated_at, body[:4000]}`.
  - **Test:** `tests/test_collector.py::test_normalize_issue_vs_pr`, `::test_normalize_body_truncated_and_defaults`.
- [x] **T0.3 JSONL upsert store** — `src/collector.py::_merge_jsonl`.
  - **Why:** re-fetched items overwrite by number (no duplicates, always latest), persisted to a per-repo JSONL.
  - **e.g.:** #123 already stored + an updated copy arrives → that one line is replaced, the rest untouched.
  - **Test:** `tests/test_collector.py::test_merge_jsonl_upsert_and_order`.
- [x] **T0.4 Incremental state cursor** — `src/collector.py::_load_state/_save_state`.
  - **Why:** remember each repo's last-collected time so the next run only fetches newer items (incremental).
  - **e.g.:** `data/state.json` = `{"vllm-project/vllm": "2026-07-03T…Z"}` → next run uses that as `since`.
  - **Test:** `tests/test_collector.py::test_state_roundtrip`.
- [x] **T0.5 Rate-limit handling + auth headers** — `src/collector.py::_headers/_sleep_for_rate_limit`.
  - **Why:** a token lifts the limit 60→5000/hr, and on exhaustion we wait for reset instead of crashing.
  - **e.g.:** `403` + `X-RateLimit-Remaining: 0` → sleep until reset then retry; `GITHUB_TOKEN` → `Bearer` header.
  - **Test:** `tests/test_collector.py::test_headers_token`, `::test_rate_limit_no_wait_on_ok`, `::test_rate_limit_waits_on_403`.
- [x] **T0.6 Pluggable store interface** — extract `src/store/base.py` (`upsert_items`, `get_item`, `query`, `get_state`, `set_state`); move JSONL logic into `src/store/jsonl_store.py`; collector writes via the interface.
  - **Why:** swap JSONL → Firestore (M0.6) later without touching the collector/agents — one interface, many backends.
  - **e.g.:** `get_store().upsert_items(recs)` writes JSONL today, Firestore tomorrow, same call.
  - **Test:** `tests/test_store_jsonl.py` — upsert + query(by repo/label/state) + state round-trip on `tmp_path`.
- [x] **T0.7 LLM wrapper (pluggable)** — `src/llm.py::complete` dispatching on `LLM_PROVIDER` (`claude_cli` → `claude -p`; `claude_api`; `local`/vLLM OpenAI-compatible); JSON-mode, timeout, error handling; returns call metadata (tokens/latency/cost) for the T2.6 bandit.
  - **Why:** every agent needs an LLM; one wrapper lets us switch claude_cli / API / local-vLLM without editing agents.
  - **e.g.:** `complete("classify: <issue>", json_schema=TAXONOMY)` → `{"category": "rocm-build"}` — same call whether it hits `claude -p` or a local vLLM server.
  - **Test:** `tests/test_llm.py` — each provider mocked (prompt passed, response parsed, `json_schema` → dict, metadata populated, error path raises); live smoke = `@pytest.mark.integration`.
- [x] **T0.8 Baseline weekly report v0 (fixed taxonomy, no LLM)** — `src/agents/reporter.py`: read items from store, bucket by fixed-taxonomy keyword match, emit Markdown with cited links.
  - **Why:** first human-readable deliverable — raw JSONL → a weekly digest, and sets the "every claim cites a link" bar before any LLM is involved.
  - **e.g.:** `## ROCm builds (3)` → `- [vllm#123] hipBLAS build fails on gfx90a — https://github.com/vllm-project/vllm/issues/123`.
  - **Test:** `tests/test_reporter.py` — synthetic items → report contains every item URL + correct per-section counts.
- [x] **T0.9 Report CLI** — `python -m src.report` writes `data/reports/YYYY-Www.md`.
  - **Why:** one command to produce the weekly report on demand / on a schedule.
  - **e.g.:** `python -m src.report` → writes `data/reports/2026-W27.md` and prints its path.
  - **Test:** `tests/test_report_cli.py` — `main()` on a tmp store creates a non-empty file.
- [x] **T0.10 Collector robustness (guardrail 1a: collect without error)** — per-repo `try/except` (one repo's failure doesn't abort the run); save state incrementally after each repo; retry+backoff on 5xx/timeouts; honor secondary rate limits (`Retry-After`); validate required fields (`number,url,updated_at,type`) + log/skip malformed; configurable `body` cap (raise for RAG).
  - **Why:** a 24h unattended collector must survive one flaky repo / transient 5xx without losing the other repos' progress — partial success beats an all-or-nothing crash.
  - **e.g.:** repo #2 throws mid-run → repo #1's cursor is already saved, so the next run resumes there instead of re-fetching everything.
  - **Test:** `tests/test_collector_robust.py` — repo #2 raises ⇒ repo #1 cursor persisted; `5xx,5xx,200` ⇒ succeeds; `403 + Retry-After` waits then continues; malformed record skipped+logged; cursor monotonic.
  - **Note:** partly shipped in #3 (cursor-windowing past the ~1000-item pagination cap + per-repo isolation + incremental state save, with tests). Remaining: retry/backoff, secondary-rate-limit, schema validation, configurable `body` cap.
- [x] **T0.11 Collection data-quality guardrail (guardrail 1b: reconciliation)** — `src/audit.py`: compare local counts vs GitHub **GraphQL** `issues.totalCount + pullRequests.totalCount` over the window; scan collected `number`s for gaps (alert on gap *ratio* — deleted/transferred allowed); write a `data_quality` record (count delta, gap ratio, error count) to the KB each run.
  - **Why:** "the collector ran without error" ≠ "we got everything" — reconciling against GitHub's own totals is the only way to *trust the numbers* every later stage builds on.
  - **e.g.:** GitHub reports 420 issues in the window, we stored 400 → `data_quality.delta = -20` flagged, so a silent gap surfaces instead of poisoning the report.
  - **Test:** `tests/test_audit.py` — offline: synthetic local vs remote → delta computed, gap-ratio flagged over threshold; live GraphQL compare on a small repo = `@pytest.mark.integration`.
- [x] **T0.12 Collector review follow-ups** (from the #3 review) — log the cursor-stall case to the `data_quality` metric (not just stderr) + GraphQL fallback for single-timestamp clusters >1000; make non-network failures in `main` loud (narrow the `except`, log the traceback) instead of looking like a transient skip; add `pytest-timeout` so the stall-guard test fails fast.
  - **Why:** the #3 fix worked but could hide real failures (a silent stall / a swallowed non-network bug reads like "nothing new") — make every failure mode visible + testable.
  - **e.g.:** a single timestamp holds >1000 items (pagination cap) → GraphQL fallback fetches them + records a stall event, instead of silently truncating.
  - **Test:** `tests/test_audit.py::test_stall_recorded`; `tests/test_collector_robust.py` (timeout marker).

## M0.6 — Storage: Firestore KB backend

> **Expected output:** the same collected data readable/writable via Firestore; `STORE=firestore` works; a
> jsonl→firestore migration script.
> **Demo:** `STORE=firestore python -m src.collector` (against the Firestore emulator) · `python -m src.store.migrate`.
> **Acceptance:** `pytest -m m0_6` green — the store contract test passes for **both** jsonl and firestore backends.

- [x] **T0.6.1 Firestore store** — `src/store/firestore_store.py` implementing `store/base.py` (collection `items` keyed `repo#number`; collection `state`).
  - **Why:** the real shared KB — concurrent-safe reads/writes (scheduler writes while the dashboard reads), queryable by field, no whole-file rewrites like JSONL.
  - **e.g.:** `STORE=firestore` → `upsert_items` writes one doc per issue keyed `vllm-project/vllm#123`; the dashboard reads it live.
  - **Test:** `tests/test_store_contract.py` — **one contract test parametrized over jsonl + firestore** so both satisfy identical assertions; firestore param uses the **Firestore emulator**, marked `integration`.
- [x] **T0.6.2 Store factory** — `src/store/__init__.py::get_store()` selects impl via env `STORE=jsonl|firestore`.
  - **Why:** callers ask for "the store" and get the configured backend — no agent hardcodes JSONL vs Firestore.
  - **e.g.:** `STORE=firestore python -m src.collector` → same collector code, Firestore backend.
  - **Test:** `tests/test_store_factory.py` — env selects the right class (firestore import mocked).
- [x] **T0.6.3 Migration** — `python -m src.store.migrate` (jsonl → firestore).
  - **Why:** carry the data already collected under JSONL into Firestore without re-fetching from GitHub.
  - **e.g.:** `python -m src.store.migrate` → every `data/*.jsonl` item becomes a Firestore `items` doc.
  - **Test:** `tests/test_store_migrate.py` (`integration`) — sample jsonl → docs present in emulator.

## M1 — Intelligence plane (inner loop)

> **Expected output:** LLM-classified items (taxonomy + evidence) in the KB; versioned `taxonomy@v` + `policy@v`;
> timestamped calibrated forecasts; per-category trend series; an LLM-written **cited** weekly report.
> **Demo:** `python -m src.analyze` (classify new items) · `python -m src.report` (cited report) ·
> `python -m src.forecast` (log predictions) · `python -m dashboard` (thin read-only web view).
> **Acceptance:** `pytest -m m1` green · the report's every claim line carries ≥1 evidence URL ·
> the dashboard renders the latest report + trend charts (superseded by T1.5.5's tree UI, below).

- [x] **T1.1 Embeddings + vector index** — `src/embed.py` (embed text/labels; NN search; backend TBD).
  - **Why:** semantic retrieval is the backbone of the cited report + candidate discovery — find related issues by meaning, not exact keywords.
  - **e.g.:** query "hipBLAS build failure" → nearest neighbors surface the relevant ROCm build issues even without word overlap.
  - **Test:** `tests/test_embed.py` — with a deterministic fixture/mock model, NN of a query returns the semantically closer of two docs.
- [x] **T1.2 Taxonomy schema + versioning** — `src/taxonomy.py` (`taxonomy@vN` in KB, active pointer).
  - **Why:** the classification vocabulary must *evolve* (M2 adds/retires categories) while old labels stay interpretable — so it's versioned, not mutated.
  - **e.g.:** `taxonomy@v1` → add "disaggregated-prefill" → `taxonomy@v2`; both retrievable, `active` points to v2.
  - **Test:** `tests/test_taxonomy.py` — v1 → add category → v2; both retrievable; `active` returns v2.
- [x] **T1.3 Policy object (versioned)** — `src/policy.py` (scoring weights, prompt templates, active taxonomy ref).
  - **Why:** the policy is the agent's *learning state* — M2's grading updates it (`policy@v+1`); versioning makes "what changed and why" auditable.
  - **e.g.:** `policy@v3` = {per-category weights, prompt templates, taxonomy@v2 ref}; the grader later proposes v4.
  - **Test:** `tests/test_policy.py` — versions are append-only/immutable; `get_active()` returns latest.
- [x] **T1.4 Analyst agent** — classify delta items into taxonomy via `llm.complete`; write labels + evidence to KB.
  - **Why:** turns raw collected items into the classified, evidence-linked signal every later stage (report, trends, candidates) reads.
  - **e.g.:** issue #123 → `{category: "rocm-build", evidence: [url]}` written back to its KB record.
  - **Test:** `tests/test_analyst.py` — mock `llm.complete` → item updated with category + citation preserved.
- [x] **T1.5 Forecaster agent** — emit calibrated predictions `{claim, resolution_rule, prob, due_date, evidence}`.
  - **Why:** the self-evolution loop needs falsifiable, timestamped predictions to grade later — that's what makes the agent's judgment *measurable*.
  - **e.g.:** `{claim: "spec-decoding lands in ROCm by Q4", prob: 0.7, due: 2026-12-31, evidence:[…]}` logged now, graded when it matures.
  - **Test:** `tests/test_forecaster.py` — mock llm → stored prediction validates against schema (prob∈[0,1], due>now).
- [x] **T1.6 Reporter v1 (LLM, cited)** — weekly report written from classified items.
  - **Why:** the human-facing deliverable — an LLM-written digest of what moved this week, every claim backed by a source link (evidence principle).
  - **e.g.:** "Spec-decoding activity doubled ([#a](url), [#b](url)); ROCm builds saw 3 new failures ([#c](url))."
  - **Test:** `tests/test_reporter_v1.py` — mock llm → **every claim line has ≥1 evidence URL** (evidence principle).
- [x] **T1.7 Trend series** — `src/trends.py`: per-category activity time series from KB.
  - **Why:** momentum over time (not a snapshot) is what reveals *direction* — which techniques are heating up/cooling — and feeds the dashboard charts.
  - **e.g.:** `trends("quantization")` → weekly counts `[3,5,4,9,12]`, a rising topic.
  - **Test:** `tests/test_trends.py` — synthetic items across weeks → correct bucketed counts per category.
- [x] **T1.8 RAG evaluation guardrail (guardrail 2: a trustworthy score)** — `src/rag_eval.py` + `tests/rag_eval/golden.jsonl` (hand-labeled query → relevant ids): **retrieval** metrics (Recall@k, MRR, nDCG@k) + **generation** metrics (faithfulness/groundedness via LLM-judge, citation-accuracy, hallucination-rate); enforce thresholds (e.g. Recall@10 ≥ 0.8, hallucination_rate = 0) + write scores to KB each run for drift.
  - **Why:** the report is only worth trusting if retrieval/citation quality is *measured* — this is the gate that catches hallucination/drift before a bad report ships.
  - **e.g.:** a run scores Recall@10 = 0.72 (< 0.8) → flagged; an absent-topic query must return "no evidence", not a fabricated cite.
  - **Test:** `tests/test_rag_eval.py` — offline: metric math on a fixed ranked list (known Recall@k/MRR/nDCG), every claim carries a citation, an absent-topic query ⇒ "no evidence"; live retrieval + LLM-judge = `@pytest.mark.integration`.
- [x] **T1.9 Thin read-only dashboard** — early slice of the M5 dashboard pulled forward to right after M1; a local `dashboard/` web view over the KB (read via the store interface, so JSONL now / Firestore after M0.6 both work).
  - **Why:** view M1 outputs (report · trends · forecasts) in one screen instead of running CLI commands — the "follow-along" tool; PLAN.md sanctions an early thin version as soon as there's a loop to watch.
  - **e.g.:** `python -m dashboard` → localhost shows the latest cited report + per-category trend charts + the forecast log. Read-only, no auth. (T1.5.5 later replaces the flat report section with a collapsible taxonomy tree — see below.)
  - **Test:** `tests/test_dashboard.py` — seed a store fixture on `tmp_path`, assert the render functions return the report body + correct trend series (offline); a live server smoke = `@pytest.mark.integration`.
- [x] **T1.10 Store merge-on-upsert semantics** — `src/store/base.py` + both backends: `upsert_items` merges the given fields onto an existing `(repo, number)` record instead of fully replacing it, across `JsonlStore` and `FirestoreStore` (flagged as a known limitation in T1.4's `analyst.py` docstring).
  - **Why:** the collector's `_normalize()` never carries `category`/`taxonomy_version` forward, so re-fetching an already-classified item (any new comment/label bumps `updated_at` back into the incremental window) silently wiped its classification on the next collection cycle.
  - **e.g.:** an item classified `category="build"` gets re-collected after a new comment; the re-normalized record has no `category` key → the stored record keeps `category="build"`, only the fields the new record actually specifies (title/state/labels/etc.) are overwritten.
  - **Test:** `tests/test_store_contract.py` — a second upsert that omits a previously-set field preserves it; a second upsert that specifies a field overwrites it; both backends (JsonlStore offline, FirestoreStore `@pytest.mark.integration`) pass the same contract.
- [x] **T1.11 Shared store-selection CLI helper** — extract the `--data-dir → JsonlStore(data_dir) else get_store()` branch (duplicated across `src/analyze.py`, `src/forecast.py`, `src/report.py`, `dashboard/__main__.py`) into one shared helper.
  - **Why:** past the codebase's own rule-of-three — a 4th copy landed with T1.9's dashboard; a future change to store selection now needs 4 hand-edited call sites.
  - **e.g.:** a shared `resolve_store(args.data_dir) -> (store_or_factory, data_dir)` (or similar) helper in `src/store/__init__.py`, called from all 4 CLIs.
  - **Test:** existing CLI tests (`test_analyze.py`/`test_forecast.py`/`test_report.py`/`test_dashboard.py`) continue to pass unchanged against the shared helper — a refactor, not a behavior change.

## M1.5 — Hierarchical report tree (readable 大 → 소 → 소소 → PRs)

> **Why this milestone:** the current report/dashboard lists PRs flat under one big category — hard to scan.
> This makes classification **hierarchical** and renders it as a collapsible tree with per-node summaries, e.g.
> `ROCm/AMD → DeepSeek-V4 → performance → attention → cited PRs`.
> **Design reference:** [`docs/design/report-tree-mockup.html`](design/report-tree-mockup.html) — open it in a
> browser; that is the approved look (collapsible nodes · per-node summary · count + `gap` chips · PR-state chips
> · filter · light/dark). Build the UI to match it. Extends M1; tests carry `pytest.mark.m1`.
> **Expected output:** items classified to a **taxonomy path** (not one flat label); an LLM **summary per internal
> node**; the report + dashboard render the tree matching the mockup.
> **Demo:** `python -m src.report` (tree-structured) · `python -m dashboard` → collapsible tree with node
> summaries + filter (view via the ssh tunnel, same as before).
> **Acceptance:** `pytest -m m1` green · the dashboard shows the `AMD → DeepSeek → performance → attention → PRs`
> tree with per-node summaries · every leaf carries an evidence URL.

- [x] **T1.5.1 Hierarchical taxonomy (path)** — evolve `src/taxonomy.py` so a category is a **path** `[level0, level1, …]` (engine/vendor → area → topic), still versioned; keep back-compat so an existing flat label reads as a depth-1 path.
  - **Why:** a single flat label can't express `AMD → DeepSeek-V4 → performance → attention`; a path is what lets the report nest into a tree.
  - **e.g.:** an item carries `path=["ROCm/AMD","DeepSeek-V4","performance","attention"]` instead of `category="rocm"`.
  - **Test:** `tests/test_taxonomy.py` — a path round-trips through the store; versioning still holds; a legacy flat label still reads as a depth-1 path.
- [x] **T1.5.2 Analyst assigns a path** — evolve `src/agents/analyst.py` to classify each item into a taxonomy **path** via `llm.complete` (each level from a controlled per-level label set to prevent drift), writing `path` + evidence to the KB.
  - **Why:** this is what actually fills the tree — without a per-item path every node stays a flat bucket.
  - **e.g.:** an "MLA decode on MI300" issue → the path above + its source URL, written back onto the item.
  - **Test:** `tests/test_analyst.py` — mock `llm.complete` → item gets a valid path (each level from the allowed set), citation preserved; an off-taxonomy answer is rejected/normalized.
- [x] **T1.5.3 Node summarizer** — new `src/agents/summarizer.py`: for each internal tree node, `llm.complete` writes a **1–2 line synthesis** of that node's items (cited), keyed/cached by node path.
  - **Why:** the "소분류 요약" — turns a bucket of PRs into a scannable "what's happening here"; the biggest readability lever after nesting.
  - **e.g.:** node `ROCm/AMD > DeepSeek-V4 > performance` → "MLA + MoE kernels are the frontier; the theme is closing decode-parity vs SGLang."
  - **Test:** `tests/test_summarizer.py` — mock llm → a node summary is produced and every claim cites ≥1 item URL; an empty node yields no summary (no hallucinated content).
- [x] **T1.5.4 Tree-structured report** — evolve `src/agents/reporter.py` (or a report builder) to emit the **nested tree** grouped by path: `{name, summary, count, gaps, children[], prs[]}` as JSON + Markdown, with counts rolling up.
  - **Why:** both the CLI report and the dashboard must consume one tree structure; counts/gaps roll up so a parent shows its subtree totals.
  - **e.g.:** `data/reports/…tree.json` = nested nodes; the Markdown renders indented `大 → 소(summary) → 소소 → PRs`.
  - **Test:** `tests/test_reporter_v1.py` — fixtures → correct nesting + rolled-up counts; every leaf PR has an evidence URL; empty branches pruned.
- [x] **T1.5.5 Dashboard tree UI (match the approved mockup)** — upgrade `dashboard/render.py` (+ `server.py`) to render the tree: **collapsible** nodes, per-node summary line, count + `gap` chips, PR-state chips (merged/open/issue), a **filter** box, light/dark — matching `docs/design/report-tree-mockup.html`.
  - **Why:** the readability win itself; the same renderer displays the real hierarchy T1.5.1–1.5.4 produce (the mockup is the design spec, not throwaway).
  - **e.g.:** `python -m dashboard` → the collapsible tree (like the mockup) over live KB data; the filter narrows to matches and auto-expands ancestors.
  - **Test:** `tests/test_dashboard.py` — a seeded tree → render produces nested nodes + summaries + evidence links; filtering to a term keeps only matching leaves.

## M2 — Outer loop: grading + candidate discovery

> **Expected output:** grading metrics (precision/recall/Brier) on matured forecasts → `policy@v+1`; taxonomy
> evolution; an engine×capability parity matrix; a candidate queue ranked on **risk, implementation effort, and
> impact** (all three, not risk alone); cost-aware provider bandit; novelty filter.
> **Demo:** `python -m src.grade` (score past predictions) · `python -m src.candidates` (ranked queue with
> risk/effort/impact).
> **Acceptance:** `pytest -m m2` green · every candidate carries risk, effort, and impact scores + evidence links.

- [x] **T2.1 Grader** — `src/agents/grader.py`: resolve matured predictions vs reality (merged / in release / adopted); compute precision/recall + Brier.
  - **Why:** grading its own past predictions is *the* self-evolution signal — without it the agent can't tell if its judgment is any good.
  - **e.g.:** a Q3 forecast "X will merge" is now merged → scored a hit; aggregate → precision 0.68, Brier 0.19.
  - **Test:** `tests/test_grader.py` — synthetic predictions + outcomes → known metric values.
- [x] **T2.2 Policy update from grades** — propose `policy@vN+1` from grading results.
  - **Why:** closes the loop — grading is useless unless the scores actually reweight the policy that drives the next round.
  - **e.g.:** a category with 0.3 precision → its scoring weight drops in `policy@v+1`; a reliable one gains.
  - **Test:** `tests/test_policy_update.py` — a low-precision category → its weight decreases in the new version.
- [x] **T2.3 Curator** — `src/agents/curator.py`: propose new / retire dead taxonomy categories from activity.
  - **Why:** the taxonomy must track a moving field — new techniques appear, old ones die; a static vocabulary goes stale.
  - **e.g.:** no activity in a category for N weeks → flagged retire; a novel issue cluster → proposed new category.
  - **Test:** `tests/test_curator.py` — category with no activity for N weeks → flagged retire; a novel cluster → proposed new category.
- [x] **T2.4 Parity matrix** — `src/parity.py`: engine × capability with evidence + gap flags.
  - **Why:** the contribution strategy is "find what exists elsewhere but is missing upstream" — the matrix makes those gaps explicit.
  - **e.g.:** (ROCm/vllm fork, "fp8 kv-cache") = present, (upstream vllm, same) = missing → gap flagged as a port candidate.
  - **Test:** `tests/test_parity.py` — synthetic capability signals → matrix cell populated + "present in fork, missing upstream" gap flagged.
- [x] **T2.5 Candidate discovery + risk/effort/impact ranking** — `src/agents/scout.py`: candidates (ROCm-reproducible / good-first-issue / parity gap), each scored on **risk, implementation effort, and impact** — three independent dimensions, not one collapsed "risk" tier — ranked from that combined read.
  - **Why:** this is the "*where can I contribute?*" output — a human (or M3's engineer) picking what to work on needs more than "how risky": a low-risk/low-impact candidate isn't obviously worth doing before a medium-risk/high-impact one, and an effort estimate is what makes a candidate actually schedulable soon vs. someday.
  - **e.g.:** ranked queue `[#c risk=low effort=low impact=low (docs), #d risk=med effort=med impact=high (ROCm build fix blocking several downstream issues), #e risk=high effort=high impact=high (parity port)]`, each with evidence links and all three scores shown, not a single tier.
  - **Test:** `tests/test_scout.py` — fixture items → every ranked candidate carries risk, effort, and impact scores + evidence present; ranking reflects the combination of all three, not risk alone (e.g. a low-risk/low-impact item doesn't outrank a medium-risk/high-impact one).
- [x] **T2.6 Cost-aware LLM provider selection (bandit)** — `src/llm_bandit.py`: a UCB-style bandit over `LLM_PROVIDER` (claude_cli / claude_api / local-vLLM) using per-call reward (task success) vs cost/latency from T0.7's metadata; agents ask the policy which provider to use. *(Borrowed from ShinkaEvolve.)*
  - **Why:** different tasks warrant different models — spend big-model budget only where it pays off, cheap/local elsewhere — automatically, from measured reward-per-cost.
  - **e.g.:** classification runs fine on local vLLM (cheap) while patch-writing routes to a stronger provider — the bandit learns this from outcomes.
  - **Test:** `tests/test_llm_bandit.py` — synthetic reward/cost history → bandit prefers the best reward-per-cost provider; an unseen provider still gets explored.
- [x] **T2.7 Novelty / dedup filter before expensive evaluation** — `src/novelty.py`: reject a candidate *before* a costly MI250 build if it near-duplicates a prior attempt (embedding similarity ≥ threshold) or an LLM-as-novelty-judge rules it redundant. Gates T3.2/T3.3. *(Borrowed from ShinkaEvolve — the biggest sample-efficiency lever.)*
  - **Why:** MI250 build+verify is the most expensive step — not re-attempting a near-duplicate candidate is the single biggest way to save that budget.
  - **e.g.:** a new candidate 0.95-similar to a failed prior attempt → rejected before any build; a genuinely new one passes.
  - **Test:** `tests/test_novelty.py` — near-duplicate candidate rejected; a genuinely new one passes (embedding + judge mocked).

## M3 — Contribution plane (MI250 verification oracle) — human-gated

> **Expected output:** a reproduced bug signal on MI250; a **verified** patch (signal flips) on a fork branch;
> ensemble self-review votes; a human-gate artifact; a **draft PR** (only after approval).
> **Demo:** `python -m src.engineer --candidate <id>` (repro→patch→verify on mi250-05x) prints verified=true/false;
> the human gate opens a draft PR only with `--approve`.
> **Acceptance:** `pytest -m m3` green · (integration) a real MI250 run yields a verified patch · **T3.6 = a real
> draft PR URL pasted in the checklist.**

- [x] **T3.1 Remote runner** — `src/runner.py`: run a command on `mi250-05x` over ssh, stream logs, capture exit code + artifacts.
  - **Why:** the CPU agent needs hands on the GPU box — every repro/build/verify step is a command executed on MI250 with its output captured.
  - **e.g.:** `run("mi250-051", "pytest test_rocm.py")` → streams logs, returns `{exit: 1, artifacts: […]}`.
  - **Test:** `tests/test_runner.py` — mock subprocess/ssh → correct command composed + result parsed. Real ssh = `integration` (runs only when `MI250_HOST` set).
- [x] **T3.2 Repro harness** — given a candidate, run repro on MI250, capture failing signal → KB `runs`.
  - **Why:** a fix is only credible if the bug was first *reproduced on real hardware* — the failing signal is the "before" half of the proof.
  - **e.g.:** candidate #d → run its repro on mi250-051 → capture the failing assertion/log as the baseline signal.
  - **Test:** `tests/test_repro.py` — mock runner returns a failing log → failing signal recorded.
- [x] **T3.3 Engineer patch loop** — `llm.complete` generates a patch on a fork branch → rebuild/test on MI250 → confirm signal flips.
  - **Why:** the core contribution act — and MI250 is the empirical oracle: a patch counts as verified only when the failing signal actually flips to passing.
  - **e.g.:** patch applied → rebuild on mi250-051 → the T3.2 failing test now passes ⇒ `verified=True`; still fails ⇒ no PR.
  - **Test:** `tests/test_engineer.py` — mock llm+runner: fail→patch→pass ⇒ `verified=True`; fail→patch→fail ⇒ `verified=False` and **no PR**.
- [x] **T3.4 Ensemble self-review gate** — before the human gate, run N independent adversarial self-critiques of the verified patch (multi-sample vote) via `llm.complete`; require a majority "looks correct" **in addition to** the MI250 pass. *(Borrowed from The AI Scientist's ensemble reviewer.)*
  - **Why:** a passing test can still hide a bad patch (reward-hacking, side effects) — an adversarial vote catches what the hardware check can't, before spending the human's attention.
  - **e.g.:** 5 critiques, 4 say "correct" → advance to human gate; a 3–2 split → hold, don't gate.
  - **Test:** `tests/test_self_review.py` — mock llm votes: majority-approve ⇒ advance; split/reject ⇒ hold (no gate).
- [x] **T3.5 Human gate** — assemble `{diff, risk badge, repro evidence, MI250 logs, self-review votes}`; `gh pr create --draft` **only** after an explicit approve flag.
  - **Why:** the mandatory human checkpoint — nothing reaches upstream vLLM without a person seeing the full evidence bundle and approving (reputation safety).
  - **e.g.:** `engineer --candidate d` prints the bundle; only `--approve` triggers `gh pr create --draft`.
  - **Test:** `tests/test_gate.py` — unapproved ⇒ `gh` never called; approved ⇒ `gh` invoked (subprocess mocked). **HARD: nothing reaches upstream without approval.**
- [ ] **T3.6 First real PR** (lowest risk: docs/typing/test-only) through the gate.
  - **Why:** the project's first actual upstream deliverable — proves the whole pipeline end-to-end on a low-risk change before attempting harder fixes.
  - **e.g.:** a docs/typing fix flows repro→patch→verify→self-review→approve→ a real draft PR URL on vllm-project/vllm.
  - **Test:** manual/`integration` — **draft PR URL pasted here**; checked only then.
  - **Note:** PR URL = …

## M4 — Orchestration / always-on (on ce-master, tmux)

> **Expected output:** an always-on loop on `ce-master` under tmux; per-stage run events **+ live
> heartbeats/intermediate output** in the KB; run-locking.
> **Demo:** `python -m src.orchestrator --once` (one dry-run tick) · tmux runbook in `docs/`.
> **Acceptance:** `pytest -m m4` green · a dry-run tick completes and writes a run event **+ a heartbeat**.

- [ ] **T4.1 Orchestrator** — `src/orchestrator.py`: pin active `policy@v`, route deltas → agents, enforce cadences (data plane **daily**, `config.COLLECT_INTERVAL_HOURS=24` / intel daily+weekly / contribution triggered).
  - **Why:** the conductor that turns a pile of agents into one always-on pipeline — right thing, right cadence, under one pinned policy version.
  - **e.g.:** a tick pins `policy@v4`, runs collect (daily), classify+report (weekly), skips contribution unless a candidate triggers it.
  - **Test:** `tests/test_orchestrator.py` — fake clock + fake agents → correct routing per cadence; gate respected.
- [ ] **T4.2 Scheduler + tmux runbook** — launch on `ce-master` under tmux (cron/systemd); runbook in `docs/`.
  - **Why:** "always-on" is the essence of the project — it must survive disconnects/restarts, not depend on a laptop staying open.
  - **e.g.:** cron kicks the orchestrator daily in a tmux session on ce-master; `docs/RUNBOOK.md` says how to attach/restart.
  - **Test:** `tests/test_schedule_dryrun.py` — a dry-run tick runs end-to-end without error (agents stubbed).
- [ ] **T4.3 Locking / idempotency** — overlapping runs don't double-write.
  - **Why:** a scheduled run + a manual run (or a slow run overlapping the next tick) must not corrupt the KB with duplicates.
  - **e.g.:** a second run starts while the first is mid-flight → it backs off; no item written twice.
  - **Test:** `tests/test_locking.py` — second concurrent run backs off; no duplicate KB writes.
- [ ] **T4.4 Run events** — per-stage run records (stage, status, counts, duration) to KB for the dashboard.
  - **Why:** the audit trail — "what ran, when, how much, how long" — that the dashboard and any debugging read from.
  - **e.g.:** a tick writes `{stage: collect, status: ok, items: 42, dur_s: 31}` to the `runs` collection.
  - **Test:** `tests/test_events.py` — a pipeline tick writes a run event with the expected fields.
- [ ] **T4.5 Liveness: heartbeat + intermediate output (know what's running)** — every stage/agent writes its lifecycle to KB `runs`: `started` → periodic `heartbeat` (current step + a rolling tail of intermediate output) → `finished` / `failed`, each stamped with the active `policy@v`; a run is **stalled** if its last heartbeat is older than `N × expected_interval`. Powers the T5.8 health panel.
  - **Why:** a crash must never look like "still running" — the dashboard has to distinguish alive / stalled / failed (with partial output) or you can't trust what you see.
  - **e.g.:** the engineer stage dies mid-build → its record shows `failed` (or `stalled` if silent), not a frozen "running".
  - **Test:** `tests/test_liveness.py` — a stage emits started→heartbeat→finished with monotonic timestamps; a stale heartbeat is classified `stalled`; an exception path records `failed` (nothing left silently "running").
- [ ] **T4.6 Collector scheduler + health** — `scripts/collect.sh` runs `python -m src.collector` on a **cron** schedule (ce-master); writes `data/last_run.json` (status / exit / timestamps / error tail) + `data/logs/`. *(Scaffold shipped; follow-ups: real alerting, optional systemd timer.)*
  - **Why:** the collector is the data lifeline — it needs its own scheduled run + a health record so a silent collection failure is visible.
  - **e.g.:** `data/last_run.json` = `{status: ok, items: 42, ts: …}`; on failure it carries the exit code + error tail.
  - **Test:** `tests/test_collect_health.py` — a stubbed run writes a well-formed `last_run.json` (ok + error cases).
- [ ] **T4.7 Self-heal triage (human-gated)** — `scripts/triage.sh`: on a collector failure run `claude -p` to diagnose and — only for a code bug, clean tree, no existing `triage/*` PR — fix on a branch, add a test, and **open a PR** (never merges). *(Scaffold shipped; follow-ups: dedupe by error signature.)*
  - **Why:** an unattended pipeline should attempt to fix its own bugs — but as a *proposed PR*, keeping the human merge gate intact (no autonomous self-modification of main).
  - **e.g.:** collector crashes on a new GitHub payload shape → triage opens a fix PR with a regression test for you to review.
  - **Test:** integration (needs `claude`+`gh`); the skip-guards are shell-checkable.

## M5 — Dashboard (Firestore-backed; monitoring + trends + parity)

> **Expected output:** a local web dashboard reading the KB — a **live health / "what's running now" view**
> (which stage is active, alive/stalled, current step, intermediate output), monitoring panels, trend charts,
> engine×capability parity heatmap, pipeline data-flow diagram, guardrail panels, and the patch review pane.
> **Demo:** `python -m dashboard` (or `streamlit run dashboard/app.py`) → open the printed localhost URL.
> **Acceptance:** `pytest -m m5` green · panels/charts render from a seeded KB · the health view shows a running
> stage as active and a stale one as `stalled`.

- [ ] **T5.1 Read layer** — `dashboard/api.py`: read-only access the UI consumes (items/trends/predictions/candidates/parity/runs).
  - **Why:** one clean read API keeps the UI decoupled from the KB backend (JSONL/Firestore) and keeps the dashboard strictly read-only.
  - **e.g.:** `api.trends("quantization")` / `api.candidates()` return plain shapes the panels render.
  - **Test:** `tests/test_dashboard_api.py` — seeded store → each endpoint returns the expected shape.
- [ ] **T5.2 Monitoring panels** — collection stats, taxonomy timeline, prediction scoreboard, candidate queue.
  - **Why:** the per-stage "is it working + what did it decide" view — the pipeline made legible at a glance.
  - **e.g.:** a taxonomy-timeline panel shows when each category was added/retired; a scoreboard shows prediction precision over time.
  - **Test:** `tests/test_dashboard_panels.py` — panel data functions return expected shapes from fixtures.
- [ ] **T5.3 Trend visualizations** — category momentum over time.
  - **Why:** the market-reading half of the dashboard — turn T1.7's series into charts that show where the field is heading.
  - **e.g.:** a line chart of "spec decoding" vs "disaggregated prefill" activity across the last 12 weeks.
  - **Test:** `tests/test_dashboard_trends.py` — series endpoint returns bucketed points.
- [ ] **T5.4 Parity diagram** — engine × capability heatmap/matrix; each cell links to evidence.
  - **Why:** the visual counterpart to M2's parity gaps — "who has what, who lags" legible instantly, every cell traceable to sources.
  - **e.g.:** a heatmap (vllm · ROCm/vllm · SGLang · Dynamo · llm-d) × (paged-KV, spec-decode, fp8 …); red cells = upstream gaps.
  - **Test:** `tests/test_dashboard_parity.py` — matrix endpoint returns cells + evidence + gap flags.
- [ ] **T5.5 Pipeline data-flow diagram** — architecture view of the running system.
  - **Why:** an at-a-glance map of the live pipeline (the rendered counterpart to PLAN.md's diagrams) so onlookers grasp the system fast.
  - **e.g.:** the 3-plane flow rendered with each stage's current status pulled from live metadata.
  - **Test:** `tests/test_dashboard_diagram.py` — diagram data builds from live stage metadata (smoke).
- [ ] **T5.6 Review pane (folds in M3 gate)** — diff + risk + approve/hold in the same UI.
  - **Why:** merge monitoring + the human review gate into one place — approve/hold a patch without leaving the dashboard.
  - **e.g.:** a candidate card shows the diff + risk badge + MI250 logs; an "approve" button flips it toward a draft PR.
  - **Test:** `tests/test_dashboard_review.py` — approve action flips candidate status (gate mocked).
- [ ] **T5.7 Guardrail panels** — data-quality (reconciliation delta, gap ratio, collector error-rate over time) and RAG-eval scores (Recall@k, faithfulness, hallucination-rate) with drift lines + threshold markers.
  - **Why:** "trust the numbers" made visible — surface the T0.11 + T1.8 guardrails so drift/threshold breaches are seen, not buried.
  - **e.g.:** a Recall@10 line dipping under the 0.8 marker turns the panel red before a bad report is trusted.
  - **Test:** `tests/test_dashboard_guardrails.py` — seeded metrics → panels return series + threshold flags.
- [ ] **T5.8 Live health / "what's running now" panel** — reads T4.5 events: per stage/agent show **running / idle / stalled / failed** (from heartbeat age), the current step, elapsed time, and a live tail of intermediate output; auto-refresh; a top-level green/red health badge.
  - **Why:** the "is it alive right now?" view — the whole reason liveness (T4.5) exists; a stall must show red, never a frozen "running".
  - **e.g.:** classifier = 🟢 running "item 40/120", collector = ⚪ idle, engineer = 🔴 stalled (heartbeat 20m old).
  - **Test:** `tests/test_dashboard_health.py` — seeded run events → a running stage renders active with its step + output tail; a stale heartbeat renders `stalled`; a `failed` event renders red.

---

## Test index (quick run)

```bash
.venv/bin/python -m pytest                 # full offline suite (must be green)
.venv/bin/python -m pytest tests/test_collector.py   # one module
.venv/bin/python -m pytest -m integration  # live: GitHub / emulator / MI250 / LLM (needs creds/hardware)
```
