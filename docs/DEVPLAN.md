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
- [x] **T0.6.4 Migration must also carry `runs.jsonl`** — `migrate()` only copies items + `state.json`; it never reads/writes run records (`stage="verify"`/`"self_review"`/`"pr_author"`/`"pr_quality"`/`"gate"`, etc.).
  - **Why:** found via code review on T3.10.6 (PR #71): a jsonl→Firestore cutover would silently drop every candidate's entire pipeline history, including `stage="gate"` records showing an already-submitted `pr_url` — defeating T3.10.6's idempotency check with zero warning right at the moment of a KB migration.
  - **e.g.:** `source.list_runs()` (no filters — every run across every repo/candidate/stage) → each written to Firestore via `dest.record_run(run)`, skipping any run that's an exact content match for one the destination already has under the same `(repo, number, stage)`; summary gains `runs_migrated`. Real run against the live Firestore emulator: a `stage="gate"` run with `submitted=True, pr_url="https://github.com/o/r/pull/99"` survived the migration and was correctly readable back via `FirestoreStore.list_runs(...)`.
  - **Test:** `tests/test_store_migrate.py` (`integration`, run against the live emulator on `localhost:8081`) — a jsonl source with `verify`/`gate` runs recorded → after `migrate()`, `FirestoreStore.list_runs(...)` returns them, including the `submitted=True, pr_url=...` gate run; re-running against an unchanged source is a true no-op (`runs_migrated == 0`) while a genuinely new run added since the last pass still gets copied; one bad/oversized run doesn't abort migration of the rest.
  - **Note:** `record_run` has **no natural identity key** on either backend (both are pure appends) — `migrate()` works around this with a content-equality dedup against the destination's existing runs per `(repo, number, stage)`, making re-running safe for an unchanged source at the cost of one `dest.list_runs` call per distinct scope. This is a heuristic, not a real key (a re-recorded run with even one different field would count as new — not currently possible, since both stores are append-only). **Also found and fixed in this same PR (surfaced by this todo's own integration test against real data):** `JsonlStore.query()`'s "scan every repo JSONL in the dir" glob (`*.jsonl`) also matched `runs.jsonl` itself, so any run record with a `number` field (nearly all of them) was misread as a pseudo-item — confirmed live against this project's own real `forager-data/` (one real `stage="gate"` run was being returned by `query()` and would have been migrated into Firestore's `items` collection as a fake item). Fixed by excluding `runs_path` from the glob.

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
> the human gate writes a local draft with `--approve` and opens a real PR only with `--approve --submit`
> on a *later, separate* invocation (see T3.7 — supersedes this line's original "`--approve` opens a draft
> PR" wording, which was the exact behavior the T3.6 incident showed was unsafe).
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
  - **Superseded by T3.7:** the "approved ⇒ `gh` invoked" line above described a single flag that, in
    practice, opened a real public PR (see T3.6's incident note) — `gate.py` now requires an explicit,
    separate `--submit` flag *and* a pre-existing draft from an earlier call before `gh` is ever invoked.
- [x] **T3.6 First real PR** (lowest risk: docs/typing/test-only) through the gate.
  - **Why:** the project's first actual upstream deliverable — proves the whole pipeline end-to-end on a low-risk change before attempting harder fixes.
  - **e.g.:** a docs/typing fix flows repro→patch→verify→self-review→**human confirm**→ a real PR URL on vllm-project/vllm.
  - **Test:** manual/`integration` — **PR URL: https://github.com/vllm-project/vllm/pull/47678** (`[Bugfix] Ship missing tool_chat_template_gemma4.jinja in packaged chat_templates`, fixes `vllm-project/vllm#47600`, opened as a draft on the fork `HAN-oQo/vllm` → `vllm-project/vllm:main`).
  - **Note:** candidate selection took real effort — three earlier candidates (`#43364`, `#37967`, `#47600`'s sibling packaging-doc issues) turned out to be either environment-drift-dependent (couldn't reproduce today), already resolved elsewhere, or already claimed by other contributors; the "good first issue"/typo/docs pools on `vllm-project/vllm` are heavily contested. `#47600` (an `examples/*.jinja` chat-template file referenced by vLLM's own docs but never shipped in the installed package) was uncontested, deterministically reproducible, and root-caused via a real published-wheel inspection. Full repro→patch→verify cycle run for real on `mi250-051` (editable install, gfx90a); self-review 5/5 and quality-gate 5/5 (twice, after a factual slip — "Gemma-3" instead of "Gemma-4" — was caught and corrected by hand, since neither automated stage flagged it). Two real pipeline defects found and fixed live during this run: (1) the composed body didn't follow the target repo's actual fetched `pr_template` structure at all — see the new **T3.10.7** — worked around by hand for this PR; (2) `gate.py`'s `_create_draft_pr` hardcoded a generic PR title, discarding T3.9's composed one entirely — fixed for real in `gate.py` (PR #78) before this submission, so it's not a one-off workaround.
  - **⚠ Previously BLOCKED — now resolved via M3.5.** The M3 test flow had auto-opened a low-quality draft on the **public**
    `vllm-project/vllm#47645` (now withdrawn) because `gh pr create` from a fork branch defaults base=upstream. M3.5's fork-first + human-confirm + maintainer-grade authoring (T3.7–T3.11) landed first; this submission went through `--fork-owner HAN-oQo` and two separate, human-confirmed `gate.py` invocations, with zero PRs ever opened against upstream until this one, explicitly approved.
  - **Scope boundary (clarified during this run):** the submit task's job ends once the PR is open with every check that's actually within our control passing — a real DCO trailer on the commit (`git commit -s`, not just body text), no lint/format issues in the diff, a title/body matching the target repo's own template. `vllm-project/vllm#47678` hit one check outside that control entirely: `pre-run-check`'s first-time-contributor gate (`hasReadyLabel || hasVerifiedLabel || mergedCount >= 4`, verified directly against the workflow source) — only a maintainer or 4+ prior merged PRs can satisfy it, not a real failure and not ours to fix. Once `REVIEW_REQUIRED` is reached with our own checks green, this task is **done**; watching/responding to the maintainer review thread from here on is T3.11's (review-response loop) job, not a re-invocation of `gate.py`. T5.14 (dashboard, M5) gives this ongoing state its own tracked view.

## M3.5 — Maintainer-grade upstream PRs (fork-first, human-confirmed)

> **Why this milestone:** a PR only gets merged if it reads like careful human work — clear problem statement, root
> cause, reproduction, on-hardware verification, and adherence to the target repo's norms — **and** nothing reaches
> a public repo without an explicit human OK. Incident: a test draft auto-opened on `vllm-project/vllm#47645`
> (withdrawn) because `gh pr create` from a fork branch defaults its base to upstream. This milestone makes the safe
> path the default *and* raises PRs to acceptance quality. Extends M3; tests carry `pytest.mark.m3`.
> **Expected output:** PRs prepared on a **fork branch** to maintainer standard; a human explicitly confirms before
> any upstream submission; a review-response loop.
> **Demo:** `python -m src.engineer --candidate <id>` → pushes a fork branch + writes a PR draft file, **no upstream
> PR**; `--submit --approve` opens the upstream PR only after the human OK.
> **Acceptance:** `pytest -m m3` green · **no code path opens a PR against `vllm-project/vllm` without an explicit
> human-confirm flag** · a sample PR body carries problem / root-cause / repro / MI250-verification sections + DCO
> `Signed-off-by` + `Fixes #`.

- [x] **T3.7 Fork-first, human-confirmed submission (HARD safety fix)** — the engineer pushes the branch to **the fork** and writes a PR draft artifact (title + body) locally; it must **not** run `gh pr create` against `vllm-project/vllm`. The upstream PR opens only via a separate, explicit `--submit --approve` step after the human reviews the draft.
  - **Why:** `gh pr create` from a fork branch defaults base=upstream → a public PR (that is how #47645 escaped). Make the safe path the default so it can't recur.
  - **e.g.:** `python -m src.engineer --push-branch --host mi250-051 --branch forager/o-r-X --expected-owner HAN-oQo` pushes to the fork (refusing if the remote doesn't look like the fork); `python -m src.gate --candidate X --approve` writes `data/pr_drafts/o-r-X.md` — zero upstream PRs. The human reads it, runs the *same* `--candidate X --approve --submit` command **again, as a separate invocation** → then (and only then) the upstream PR opens.
  - **Test:** `tests/test_gate.py` — without both flags, `gh pr create` is never invoked (subprocess mocked); with both flags on the *first* invocation for a candidate, still never invoked (the draft must predate the submit call); only on a later, separate invocation is it invoked exactly once. `tests/test_engineer.py` — `push_branch` composes/runs the right `git push` command (runner mocked) and refuses to push when `--expected-owner` doesn't match the remote.
  - **Note:** `gh pr create` stays in `gate.py` (where its own tests already lived) rather than moving into `engineer.py`'s CLI — the DEVPLAN's own `engineer --candidate d` phrasing in T3.5 was already loose shorthand for "the M3 pipeline," not literally `src/engineer.py`. `engineer.py` gained the new `push_branch` helper (a separate function, not folded into `run_engineer`, so its already-tested behavior didn't change) plus a minimal CLI to actually invoke it. `gate.py`'s single `--approve` flag was split into `--approve` (writes a local draft, never touches `gh`) and `--submit` (only meaningful together with `--approve`) — and, per a code-review finding on this very fix, `--submit` is additionally inert unless the draft already existed from an *earlier*, separate call (`GateResult.draft_is_new`), so passing both flags together on one command line still can't open a PR on the first try. `push_branch` isn't wired into `run_engineer` or any orchestrator yet (there is no M4 orchestrator to wire it into) — it's a standalone, directly-runnable step for now.
- [x] **T3.8 Contribution-norms adapter (repo profile)** — fetch + cache the target repo's `CONTRIBUTING`, PR template, DCO/sign-off requirement, title conventions, and a few recent **merged** PRs as style exemplars; expose as a repo profile the author uses.
  - **Why:** a PR that ignores the template / lacks DCO / uses the wrong title style gets bounced regardless of code quality.
  - **e.g.:** `src/pr_profile.py`'s `get_profile("vllm-project/vllm")` (real run) → `contributing` found (a short pointer to docs.vllm.ai), `pr_template` found (809 chars), `requires_dco=False`, `title_pattern="[Category] ... (observed categories: [Bugfix], [MRV2], [Feature])"` from real recent merged PR titles — an earlier version of this heuristic looked for one single repeated literal prefix and returned `None` for this exact repo (vLLM uses several different category tags, not one), caught and fixed by code review.
  - **Test:** `tests/test_pr_profile.py` — mock fetch → profile has template + sign-off flag + title pattern + N exemplars.
  - **Note:** cached as JSON under `data/pr_profiles/{repo-slug}.json` (`get_profile`, refresh via
    `refresh=True`) — mirrors `gate.py`'s own `_pr_draft_path`/`_PR_DRAFTS_DIR` None-sentinel
    pattern so tests can isolate cleanly. Auth reuses `collector._headers()` rather than a second
    GitHub-auth mechanism, same choice `gate.py` made reusing `scout._score`.
- [x] **T3.9 PR-author agent (maintainer-grade body)** — from the evidence bundle, `llm.complete` composes a title + body following the profile: **problem → root cause → fix rationale → reproduction (before/after) → MI250 verification (logs/benchmarks) → limitations**, with `Fixes #NNNN`, DCO sign-off, checklist ticked.
  - **Why:** the "reads like a human did it" step — structured, specific, evidence-backed, template-compliant (vs the `# Candidate … Risk badge` + empty-Repro + raw-diff dump that got #47645 withdrawn).
  - **e.g.:** body opens with the user-facing problem + linked issue, explains *why* the bug happens, then the fix, a copy-pasteable repro, and the MI250 pass log. Real run against a reconstructed #21948-shaped bundle: `src/pr_author.py`'s `compose_pr_body` produced `"[V1] Add direct LLMEngine test coverage for parallel sampling (n>1) add_request/step path"` + a full problem/root-cause/fix-rationale/repro/MI250-log/limitations/checklist/`Signed-off-by` body — a real, evidence-grounded document, not the `# Candidate … Risk badge` shape that got #47645 withdrawn.
  - **Test:** `tests/test_pr_author.py` — mock llm + bundle → body has all required sections, links the issue, carries `Signed-off-by`, and every repro/perf claim cites the captured MI250 run.
  - **Note:** only `title`/`problem`/`root_cause`/`fix_rationale`/`limitations` are LLM-generated; the `Fixes #`, reproduction, and MI250-verification sections are spliced in verbatim from the bundle so a repro/perf claim can never be an LLM paraphrase of what MI250 actually printed (CLAUDE.md's evidence principle). `Signed-off-by` reuses the local git identity (`git config user.name`/`user.email`) rather than a new identity config; if unset, the body omits the trailer and leaves its checklist item unticked instead of fabricating one. `RepoProfile` (T3.8) is advisory prompt context only — this module doesn't itself enforce `title_pattern`/`requires_dco` compliance, that's T3.10's job. Not yet wired into `gate.py`'s draft-writing path (`_write_pr_draft` still uses `bundle.format()`) — left for T3.10/M4, same "standalone, not yet orchestrated" note T3.7 left for `engineer.push_branch`.
- [x] **T3.10 PR-quality gate ("would a maintainer accept this?")** — an adversarial ensemble scores the PR narrative against the profile + exemplars (clarity, completeness, claims-backed-by-evidence, focused diff, not boilerplate); must pass **before** the human-confirm step.
  - **Why:** catch thin / AI-looking PRs automatically before they reach a person or a maintainer.
  - **e.g.:** flags "empty Repro block" / "perf claim without numbers" / "raw diff dumped in body" → sends it back to T3.9. Real run (3 judges, real LLM, not mocked): a synthetic `# Candidate … Risk badge` body with an empty Repro block was unanimously rejected — one judge's verbatim reason: *"The Repro section is empty (no captured output/log), and the title 'Fix bug' is a vague placeholder..."*. A rich T3.9-style body (problem/root-cause/repro/MI250-log/checklist/DCO) was also caught when its narrative claimed a "bug fix" but the diff was test-only — a genuinely useful catch beyond the DEVPLAN's own worked examples.
  - **Test:** `tests/test_pr_quality.py` — a thin/boilerplate body fails; a complete, evidence-backed one passes (mock judges).
  - **Note:** mirrors `self_review.py`'s ensemble mechanics exactly, reusing `self_review.DEFAULT_VOTES`/`SUPERMAJORITY_THRESHOLD` directly rather than re-declaring them. Judges see the composed title/body, the verified diff, and `RepoProfile` (T3.8) context; `PRQualityResult.pr_author_recorded_at` traces which `stage="pr_author"` run was judged, matching `SelfReviewResult.verify_recorded_at`'s identical staleness-prevention purpose. Not wired into `gate.py`'s approve/submit path yet, and a `passes=False` verdict doesn't automatically re-invoke T3.9 — both left to M4's orchestrator, same "standalone, not yet orchestrated" note T3.7/T3.9 already left for their own downstream steps.
- [x] **T3.10.5 Wire the gate: composed narrative + quality-pass required for submit** — `gate.py` must actually submit T3.9's composed narrative (not `bundle.format()`'s raw evidence dump), and `--submit` must refuse unless a passing T3.10 verdict exists for that exact narrative.
  - **Why:** code review on T3.7/T3.9/T3.10 (PRs #66/#68/#69) each shipped as a standalone module with no caller in the real pipeline — three "not wired in yet, left to M4" notes in a row means DEVPLAN's own safety claim ("must pass before the human-confirm step") was a design intent, not an enforced code property, and `gate.py` was still one `--approve --submit` away from re-opening a `#47645`-shaped PR (the exact thin-body incident T3.9/T3.10 exist to prevent) with nothing in code stopping it.
  - **e.g.:** `python -m src.pr_author --candidate o/r#1` (composes + persists a narrative) → `python -m src.pr_quality --candidate o/r#1` (judges it, new CLI) → `python -m src.gate --candidate o/r#1 --approve` (writes a draft using the composed body, prints whether the quality gate passed) → `--approve --submit` (a *second*, separate call) opens the real PR only if that same narrative's quality verdict was `passes=True`; otherwise `gh pr create` is never invoked and the CLI explains why. Real run (real LLM for pr_author/pr_quality, `gh` mocked): a synthetic candidate's thin diff was correctly judged `0/3 passes=False`, and the follow-up `--submit` call read that verdict straight back from the KB (not a live recompute) and correctly kept `gh` uncalled; a second run with `quality_passed=True` confirmed `gh pr create`'s `--body` was exactly the composed narrative, not `bundle.format()`'s raw dump.
  - **Test:** `tests/test_gate.py` — `_finalize`/`run_gate` write the composed `pr_author` body (not `bundle.format()`) when one exists and use `bundle.format()` when none does; `submit` stays inert when `quality_passed` is `False`/`None` even with both flags on a later call (`draft_is_new=False`); `main()` reads the latest `pr_author`/`pr_quality` KB records (not live LLM calls) and refuses stale pairings. `tests/test_pr_quality.py` — new CLI prints the verdict and exits non-zero on failure/not-ready.
  - **Note:** `gate.py` does **not** import `pr_author`/`pr_quality` at module level (both already import `gate`, so that would be a real circular import, not just a lint nit) — `_finalize`/`run_gate` instead take `pr_body: str | None` and `quality_passed: bool | None` as plain parameters, and `main()` *reads* (never re-runs) the latest persisted `stage="pr_author"`/`stage="pr_quality"` KB records via the `Store` interface it already holds, cross-checked by `recorded_at` so a stale quality verdict for an older narrative is never read as covering the current one. Reading rather than re-invoking also avoids a subtler bug: re-running `pr_author.run_pr_author` on the second (`--submit`) call would make a fresh, non-deterministic LLM call and could compose a *different* body than the one the human actually read in the draft file from the first call.
- [x] **T3.10.6 Idempotency: refuse to re-submit an already-open PR** — `gate.py` must check prior `stage="gate"` history for an existing real `pr_url` before ever calling `gh pr create` again for the same candidate.
  - **Why:** code review on T3.10.5 (PR #70) flagged that `draft_is_new` resets to `True` (and so `submitted`'s gate reopens) whenever the local draft file is missing — a changed `--data-dir`, a deleted draft, or a fresh checkout — even for a candidate whose `stage="gate"` history already shows `submitted=True` with a real `pr_url`. Three PRs in this milestone just closed "deferred safety work" gaps on this exact file; leaving a known double-submission path open indefinitely is the same pattern, not a new tradeoff.
  - **e.g.:** a candidate submitted once (`pr_url="https://github.com/o/r/pull/99"` recorded in a prior `stage="gate"` run) → a human deletes the local `pr_drafts/` cache or points `--data-dir` elsewhere → `--approve --submit` must still refuse, printing the existing `pr_url` instead of opening a second PR. Real run (`gh` mocked): first submission opened `pull/42`; a second attempt against a completely fresh `--data-dir` (so `draft_is_new=True`) still correctly reported `already_open_pr_url` and never called `gh`.
  - **Test:** `tests/test_gate.py` — a `stage="gate"` history containing `submitted=True, pr_url=<real url>` blocks a fresh `_finalize`/`run_gate` call from calling `gh` even with `approve`/`submit`/`quality_passed` all `True` and `draft_is_new=True`; `main()` prints the existing PR URL instead of "first time" messaging.
  - **Note:** `_already_open_pr_url` scans *every* recorded `stage="gate"` run for the candidate, not just the latest one — a candidate could have several gate runs (approved-only, a failed `gh` call, a later successful submit), and a real PR opened at any point in that history must never be opened a second time regardless of what the most recent run shows. Checked with the highest priority of the four submission conditions (`already_open_pr_url` → `draft_is_new` → `quality_passed`), reflected in both `GateResult`'s field order and `main()`'s CLI messaging.
- [x] **T3.10.7 Enforce the fetched PR template's actual section structure** — `pr_author.py` must compose the body using the *exact* section headers `RepoProfile.pr_template` (T3.8) requires, not a fixed problem/root-cause/fix-rationale/reproduction/verification/limitations shape; `pr_quality.py`'s judges must check the composed body's headers against the template, not just narrative quality.
  - **Why:** found by hand during T3.6's first real attempt (`vllm-project/vllm#47600`) — `pr_author.py` fetches and stores `pr_template` (T3.8) but only ever hands it to the LLM as loose prompt context, never enforces it structurally; the composed body used entirely different section names (`## Problem`/`## Root cause`/`## Fix rationale`/`## Reproduction`/`## MI250 verification`/`## Checklist`) than vLLM's real template (`## Purpose`/`## Test Plan`/`## Test Result` + its own specific checklist item wording). `pr_quality.py`'s 5-judge ensemble passed the off-template body unanimously (5/5) both before and after an unrelated correction — none of the judges are prompted to check headers against the profile's template at all, so this class of defect is currently invisible to the automated gate and would have shipped to a real maintainer without a human manually diffing the two documents.
  - **e.g.:** for `vllm-project/vllm`, `pr_template` requires `## Purpose` / `## Test Plan` / `## Test Result`; a hand-corrected body restructured into that exact shape (Purpose = problem + root cause + fix rationale; Test Plan = commands; Test Result = before/after output) is what actually got submitted for `#47600` — this todo is to make that restructuring automatic instead of manual.
  - **Test:** `tests/test_pr_author.py` — given a `RepoProfile.pr_template` with named `## X`/`## Y` headers, `compose_pr_body`'s output uses those exact headers (not the current fixed set); a profile with no template falls back to the current shape. `tests/test_pr_quality.py` — a body missing a template-required header is flagged/fails even if the narrative quality is otherwise strong.
  - **Note:** `pr_author._template_sections` matches a template's own headers to three fixed concepts (purpose/test-plan/test-result) via **word-boundary-safe** keyword search (`"purpose"/"summary"/"problem"`, `"test plan"/"testing"/"how has this been tested"`, `"test result"/"results"/"verification"/"evidence"`), taking the *last* `---` divider (not the first — a template using `---` as a per-section rule, not just once before the checklist, would otherwise lose every section after the first one) and enforcing that one header can never satisfy two of the three concepts (a `claimed` set — without it, a header like "Test Plan and Results" could match both test-plan and test-result and get emitted as a duplicate heading). Only falls back to the fixed shape if it can't confidently match *all three* distinct headers, never a partial/guessed mapping. `pr_quality.py` enforces header presence as a **deterministic code check** (`_missing_template_headers`), not just an LLM prompt hint — found by hand that judges alone don't reliably catch this (the real off-template `#47600` body passed 5/5 twice); a missing header now fails `passes` outright regardless of vote outcome. Verified directly against the real cached `vllm-project/vllm` profile: `_template_sections` correctly recovers `Purpose`/`Test Plan`/`Test Result` from its actual fetched template. A `/code-review` pass on this PR found and fixed 5 more real bugs in the above (duplicate-header collision, divider truncation, naive substring matching, missing the common "How Has This Been Tested?" header, and inconsistent "no repro captured" messaging between the Test Plan/Test Result sections) plus one cleanup (redundant double-computation of `_template_sections`) — see PR #81's review thread for the full findings.
  - **Known limitation, deferred (not fixed here):** (1) `_template_sections` only ever recognizes these three hardcoded English-keyword concepts — a template requiring a genuinely different or larger section set (or one in another language) still silently falls back to the fixed shape, reproducing the same off-template mismatch this todo exists to prevent, just for a repo other than `vllm-project/vllm`; a fully general fix would need the LLM itself to populate whatever arbitrary headers `_extract_template_headers` finds, keeping only repro/verify evidence splicing pinned to a heuristic match when one exists. (2) `pr_quality.py`'s header check has no staleness guard analogous to `verify_recorded_at` — it re-resolves `RepoProfile` independently of whichever profile `pr_author` actually composed against, so a cached-profile refresh between the two separate CLI invocations could judge a body against headers it was never composed to match.
- [x] **T3.11 Review-response loop (human-gated)** — after a PR is (human-)submitted, watch maintainer comments and draft point-by-point replies + follow-up commits; the human approves each before anything is pushed to the public PR.
  - **Why:** PRs merge through the review conversation, not the first push; an unanswered review is a dead PR — but responses must never auto-push to a public PR.
  - **e.g.:** maintainer asks "add a test for the empty case" → agent drafts the test + a reply → human approves → pushed. Real run (real LLM, real comment, never posted): fed PR #71's own actual code-review summary comment into `_compose_response` — the composed reply correctly addressed all 5 numbered findings point-by-point, including citing the exact commit SHAs and the separate follow-up PR (#72) that closed one of them.
  - **Test:** `tests/test_review_loop.py` — a mock review comment → a response draft + a proposed diff are produced; nothing is pushed without approval.
  - **Note:** `run_review_loop` (compose) and `post_reply` (post) are two separate functions/CLI subcommands, not two flags on one call — unlike `gate.py`'s original design, there's no single call site where "compose" and "post" could be passed together, so `post_reply` structurally requires a `stage="review_response"` record from an earlier, separate call to exist first. Deliberately does **not** auto-apply/push a `proposed_diff`, even though one is drafted: it's LLM-composed, never run through `engineer.py`'s MI250 verification (the project's "empirical verification oracle"), so auto-pushing it onto a live public PR would bypass that oracle entirely — a materially worse risk than opening a draft PR a human approves first. The diff is a copy-pasteable starting point for a human (or a future MI250-gated automation) to verify and push through the existing pipeline. Reads GitHub's general issue-comment thread (`gh api repos/{repo}/issues/{number}/comments`), not inline line-level review comments — the more common case for a small project's PRs, with the inline feed left as a known gap.
- [ ] **T3.12 Attempt report per worked issue (success *or* fail)** — for every candidate the engineer works, write a human-readable report to the KB: **(1) issue overview · (2) how it tried to solve it (approach + what actually happened) · (3) exact steps for a human to reproduce the agent's approach** — plus the outcome (verified / failed + why). Written on **both** success and failure.
  - **Why:** transparency + learning + reproducibility — a failed attempt is still valuable (what was tried, why it failed), and a human must be able to re-run the agent's exact approach; feeds the M5 "Attempts" tab (T5.11).
  - **e.g.:** `data/attempts/<candidate>.md` — "Issue: … · Approach: patched X, rebuilt on mi250-051 · Reproduce: `git fetch fork <branch>; ./repro.sh` · Outcome: FAILED — signal didn't flip because …".
  - **Test:** `tests/test_attempt_report.py` — a mock verified run and a mock failed run each produce a report with all 3 sections + outcome + a runnable reproduce block; failure reports are still written.
  - **Note (added during T3.6's first real attempt):** for a **verified/gate-ready** candidate specifically, the report must carry (or link to) a fully **rendered** view of the gate draft — title + body + diff stats + evidence (repro/verify logs, self-review vote, quality-gate verdict) styled for human reading, not just the raw `pr_drafts/*.md` text — mirroring the ad hoc rendered draft produced for `vllm-project/vllm#47600`'s sign-off. The human should be able to review and approve straight from this rendering, the same way they would from the dashboard.
- [ ] **T3.13 PR follow-up steward (drive submitted PRs to merge)** — a standing agent that tracks **every** submitted upstream PR through to merge/close: on a schedule, poll each open PR's **CI status · new review comments · review decision (approved / changes-requested) · staleness (no activity ≥ N days) · merge conflicts · closed/merged**, and pick the next action per PR. Orchestrates T3.11 (compose replies) + the engineer (CI-fix / rebase, MI250-verified) + the human gate.
  - **Why:** opening a PR ≠ getting it merged — PRs die from unanswered reviews, CI rot, conflicts, or silence. This is the piece that keeps them *moving* toward the actual deliverable (a merge); reactive T3.11 handles one comment, this drives the whole lifecycle across all open PRs.
  - **e.g.:** PR #N — CI red → engineer prepares a verified fix commit; changes-requested → T3.11 drafts replies + a follow-up diff; approved → notify "ready, your merge"; 7 days silent → draft a polite nudge; **all human-approved before anything is pushed/posted** (fork-first, no auto-post).
  - **Test:** `tests/test_pr_followup.py` — mock PR states → the steward picks the correct action per state (ci_fail→engineer · comment→review-response · approved→notify · stale→draft-nudge · conflict→rebase); nothing is pushed/posted without an explicit approve (gh/subprocess mocked).
  - **Feeds:** a **"PRs in flight"** panel on the Agents/Ops dashboard (each open PR + its state + what it needs + who's blocked) + ntfy on state changes (extends the merge/open workflow to "PR needs your attention").

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
- [ ] **T4.8 Per-agent cost capture** — persist a cost record per `llm.complete` call to the KB (fold into the T4.4 run-event stream): `{agent, run_id, loop, model, provider, tokens_in/out/cache, cost, ts}`; price via a **maintained table** (borrow LiteLLM's `model_prices_and_context_window.json` / ccusage data) with **explicit prices for local-vLLM models**; a scheduled roll-up by day / agent / model.
  - **Why:** operating a multi-agent system needs per-agent spend visibility, and cost only accrues history if we record it from the start; the wrapper already returns the metadata (T0.7), so this is nearly free on top of T4.4.
  - **e.g.:** a `costs` doc `{agent:"analyst", model:"claude-…", cost:0.0021, run:"r-42", ts}` → the ops dashboard shows analyst = $X/day.
  - **Test:** `tests/test_cost.py` — mocked calls across agents → correct per-agent / per-day sums; a local-vLLM model priced via the custom table (offline).
  - **Ref:** `docs/research/cost-tracking.md` (recommendation #1).
- [ ] **T4.9 Claude Code session cost ingest** — the `claude -p` dev-loop / collect-loop sessions bypass `llm.complete`, so ingest **`ccusage --json`** (or Claude Code's OTel `claude_code.api_request` event) into the same cost schema, tagged by session/loop.
  - **Why:** those CLI sessions are a real chunk of spend (the "tokens burn fast" worry) and won't show up in T4.8 otherwise.
  - **e.g.:** a nightly `ccusage --json` per node → session cost rows alongside the agent costs, in one view.
  - **Test:** `tests/test_cost_ccusage.py` — a fixture ccusage JSON → session cost rows with the expected totals (offline).
  - **Note:** `ccusage` also works **today, zero-build**, retroactively (`npx ccusage@latest`) — use it for an immediate read before this lands.
- [ ] **T4.10 LiteLLM budget gateway (adopt when going always-on)** — route Anthropic API + local vLLM (+ `claude -p` via `ANTHROPIC_BASE_URL`) through a self-hosted **LiteLLM** proxy with per-key / per-tag (**= per-agent**) spend + a **hard budget cap**, so an unattended run can't overspend.
  - **Why:** T4.8/T4.9 give *visibility*; this adds *enforcement* — a bug/loop can't run up the bill overnight. **Trigger:** the first time the loop runs unattended, or the T2.6 bandit routes multiple providers; **mandatory** at the IDEAS agent-teams stage. Skip until then.
  - **e.g.:** the `analyst` key hits its daily cap → further calls refused, not a surprise invoice.
  - **Test:** `tests/test_llm_gateway.py` — client honors a budget-exceeded response (mocked); integration (proxy up) = a tagged request is costed + attributed. *(Needs Postgres; ⚠ pin a clean version — PyPI 1.82.7/8 were malware.)*
  - **Ref:** `docs/research/cost-tracking.md` (recommendation #2).

## M5 — Dashboard (Firestore-backed; monitoring + trends + parity)

> **Expected output:** a local web dashboard reading the KB — a **live health / "what's running now" view**
> (which stage is active, alive/stalled, current step, intermediate output), monitoring panels, trend charts,
> engine×capability parity heatmap, pipeline data-flow diagram, guardrail panels, and the patch review pane.
> Unified as a **tabbed operator console** — tabs: **Issues** (tree) · **Reports/Trends** (daily archive) ·
> **Candidates** (ranking + why-selected + human "work this" select) · **Attempts** (per-issue reports) ·
> **Agents/Ops** (what's running · problems/stalls · **cost per agent**) + **Guardrails**. Read-only **except two
> human decisions**: candidate selection (T5.10) and patch review (T5.6). *(Reports vs Agents/Ops are the two
> halves — the report dashboard and the agent-operations dashboard — living as tabs in one app.)*
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
- [ ] **T5.9 Reports/Trends archive tab (daily)** — browse past reports, **one per day**, newest first; open any day's cited tree report (M1.5) + its trend charts (T5.3).
  - **Why:** the trend story is a time series — you read today's report and compare it to prior days.
  - **e.g.:** a date list `2026-07-06, 07-05, …` → selecting a date opens that day's tree report + charts.
  - **Test:** `tests/test_dashboard_archive.py` — seeded daily reports → archive lists them newest-first; selecting a date returns that report.
- [ ] **T5.10 Candidate selection + human-in-the-loop signal** — the Candidates tab shows the risk-ranked queue **with *why* each was selected** (score breakdown + evidence) and a **"work this / skip" control**; the choice writes a `decision` record to the KB that the orchestrator/engineer reads — so the agent only works **human-selected** candidates.
  - **Why:** the human picks what's worth doing and the agent receives that signal **from the dashboard** (not a CLI) — human-in-the-loop selection *before* any MI250 work. (Distinct from T5.6, which approves the *result* after work.)
  - **e.g.:** click "Work #d" → writes `{candidate:d, decision:"selected", by, ts}`; the engineer's queue = selected-only.
  - **Test:** `tests/test_dashboard_select.py` — a select action writes a decision record; the engineer/orchestrator query returns only selected candidates (write path mocked; everything else stays read-only).
- [ ] **T5.11 Attempts tab (per-issue reports)** — browse the T3.12 attempt reports: outcome badge (🟢 verified / 🔴 failed), the issue overview, the approach, and the reproduce steps; link to the PR draft (if any).
  - **Why:** see exactly what the contribution agent did on each issue — success or fail — and reproduce it by hand.
  - **e.g.:** a list of worked issues with 🟢/🔴 → click → the full T3.12 attempt report rendered.
  - **Test:** `tests/test_dashboard_attempts.py` — seeded attempt reports → the tab lists them with outcome + renders all 3 sections.
  - **Note (added during T3.6's first real attempt):** for a 🟢 verified attempt, don't just link the raw draft file — render the same rich title/body/diff-stats/evidence view T3.12's note now requires directly in this tab, with the T5.6 patch-review approve action wired to it, so a human can go straight from "browsing attempts" to "approving this one" without leaving the console.
- [ ] **T5.12 Tabbed console shell** — unify the panels into one **tabbed** nav: **Issues** (tree, M1.5) · **Reports/Trends** (T5.9) · **Candidates** (select, T5.10) · **Attempts** (T5.11) · **Upstream PRs** (T5.14) · **Agents/Ops** (running · problems · cost — T5.8/T5.7/T5.13). Read-only except the T5.10 selection and T5.6 review actions.
  - **Why:** one operator console for the whole loop instead of scattered CLIs/panels — what the user asked for.
  - **e.g.:** `python -m dashboard` → top-nav tabs, each deep-linkable.
  - **Test:** `tests/test_dashboard_tabs.py` — each tab route renders its panel from seeded data.
- [ ] **T5.13 Cost panel (Agents/Ops view)** — in the **agent-ops** tab (next to the T5.8 live-health panel, *not* the report tabs): **per-agent spend** (today / 7d / total), broken down by model & provider, a burn-rate, and a **budget line**; includes the Claude Code session cost (T4.9). Reads T4.8/T4.9.
  - **Why:** operating the agents = "what's running · what's wrong · **what's it costing**" in one place; cost belongs with health, not with the reports.
  - **e.g.:** a bar per agent ($ today) + a total-vs-budget gauge that turns 🔴 when a budget line is crossed.
  - **Test:** `tests/test_dashboard_cost.py` — seeded cost records → per-agent/day series + a budget-exceeded flag.
- [ ] **T5.14 Upstream PRs tracker (live review-state sub-view)** — a dedicated tab listing only candidates with a real, currently **open** PR against an upstream repo (`stage="gate"` records with `submitted=True` and a live `pr_url`); for each, shows the maintainer review thread (T3.11's `_fetch_comments`), which comments still need a response (no matching `stage="review_post"` with `posted=True` yet — an "outstanding" count), live CI/check status (`gh pr checks`), and who's involved (comment authors, requested reviewers).
  - **Why:** requested after T3.6's first real PR (`vllm-project/vllm#47678`) — once the pipeline actually opens real PRs, the operator needs *ongoing* visibility into their review lifecycle (what came back, what's left, who's looking), not just T5.11's static "did this attempt work" snapshot. This is a live-tracking view (re-fetches GitHub state) rather than a KB-only report, since a PR's review state keeps changing after the attempt itself is long done.
  - **e.g.:** `vllm-project/vllm#47678` today would show: draft, DCO ✅, blocked on the maintainer trust gate (not a real failure — see T3.6's own note), zero review comments yet, zero outstanding. Once a maintainer comments, it'd show the comment, flag it "outstanding" until a `review_post` run with `posted=True` exists for it, and surface the requested reviewer.
  - **Test:** `tests/test_dashboard_upstream_prs.py` — seeded `stage="gate"` (submitted) + `stage="review_response"`/`review_post` records, `gh` mocked → the tab lists only open real PRs, correctly flags un-responded comments as outstanding, and renders live check status.
  - **Note:** distinct from T5.11 (all attempts, success or fail, static per-issue report) and from T3.11 itself (compose/post replies) — this is read-only visibility over the subset that's actually live upstream, pulling current GitHub state each time it's viewed rather than a frozen KB snapshot.

---

## Test index (quick run)

```bash
.venv/bin/python -m pytest                 # full offline suite (must be green)
.venv/bin/python -m pytest tests/test_collector.py   # one module
.venv/bin/python -m pytest -m integration  # live: GitHub / emulator / MI250 / LLM (needs creds/hardware)
```
