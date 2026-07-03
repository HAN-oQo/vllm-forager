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

- [x] **T0.0 Dev tooling** — black + ruff + mypy + pre-commit + pytest configured (`pyproject.toml`,
      `.pre-commit-config.yaml`, `pytest.ini`, `requirements-dev.txt`).
      Test: `pre-commit run --all-files` clean · `pytest` green.
- [x] **T0.1 Collector: incremental GitHub issue/PR fetch** — `src/collector.py::fetch_repo`.
      Test: `tests/test_collector.py::test_fetch_repo_pagination`, `::test_fetch_repo_404` (mocks `requests`,
      asserts paging stops at `< PER_PAGE` and 404 → `[]`).
- [x] **T0.2 Normalization schema** — `src/collector.py::_normalize`.
      Test: `tests/test_collector.py::test_normalize_issue_vs_pr`, `::test_normalize_body_truncated_and_defaults`.
- [x] **T0.3 JSONL upsert store** — `src/collector.py::_merge_jsonl`.
      Test: `tests/test_collector.py::test_merge_jsonl_upsert_and_order`.
- [x] **T0.4 Incremental state cursor** — `src/collector.py::_load_state/_save_state`.
      Test: `tests/test_collector.py::test_state_roundtrip`.
- [x] **T0.5 Rate-limit handling + auth headers** — `src/collector.py::_headers/_sleep_for_rate_limit`.
      Test: `tests/test_collector.py::test_headers_token`, `::test_rate_limit_no_wait_on_ok`,
      `::test_rate_limit_waits_on_403`.
- [ ] **T0.6 Pluggable store interface** — extract `src/store/base.py` (`upsert_items`, `get_item`, `query`,
      `get_state`, `set_state`); move JSONL logic into `src/store/jsonl_store.py`; collector writes via the
      interface.
      Test: `tests/test_store_jsonl.py` — upsert + query(by repo/label/state) + state round-trip on `tmp_path`.
- [ ] **T0.7 LLM wrapper (pluggable)** — `src/llm.py::complete` dispatching on `LLM_PROVIDER`
      (`claude_cli` shells out to `claude -p`; `claude_api`; `local`/vLLM OpenAI-compatible). JSON-mode parsing,
      timeout, non-zero-exit / HTTP-error handling. **Return call metadata** (tokens / latency / est. cost)
      alongside the result so the cost-aware bandit (T2.6) can route on reward-per-cost.
      Test: `tests/test_llm.py` — each provider path with subprocess/HTTP **mocked**: asserts prompt passed,
      stdout/response parsed, `json_schema` returns dict, metadata populated, error path raises cleanly. Live
      per-provider smoke = `@pytest.mark.integration`.
- [ ] **T0.8 Baseline weekly report v0 (fixed taxonomy, no LLM)** — `src/agents/reporter.py`: read items from
      store, bucket by fixed-taxonomy keyword match, emit Markdown with cited links.
      Test: `tests/test_reporter.py` — synthetic items → report contains every item URL + correct per-section counts.
- [ ] **T0.9 Report CLI** — `python -m src.report` writes `data/reports/YYYY-Www.md`.
      Test: `tests/test_report_cli.py` — `main()` on a tmp store creates a non-empty file.
- [ ] **T0.10 Collector robustness (guardrail 1a: collect without error)** — per-repo `try/except` so one repo's
      failure doesn't abort the run; **save state incrementally after each repo**; retry with backoff on 5xx /
      timeouts; honor secondary rate limits (`Retry-After`); validate each record has required fields
      (`number,url,updated_at,type`) and log+skip malformed; make the `body` cap configurable (raise for RAG).
      Test: `tests/test_collector_robust.py` — repo #2 raises ⇒ repo #1 cursor persisted; `5xx,5xx,200` ⇒ succeeds;
      `403 + Retry-After` waits then continues; malformed record skipped+logged; state cursor is monotonic.
- [ ] **T0.11 Collection data-quality guardrail (guardrail 1b: reconciliation)** — `src/audit.py`: compare local
      counts vs GitHub **GraphQL** `issues.totalCount + pullRequests.totalCount` over the collected window; scan
      collected `number`s for gaps (alert on gap *ratio* — deleted/transferred are allowed); write a `data_quality`
      record (count delta, gap ratio, error count) to the KB each run.
      Test: `tests/test_audit.py` — offline: synthetic local vs remote → delta computed, gap-ratio flagged over
      threshold; live GraphQL compare on a small repo = `@pytest.mark.integration`.

## M0.6 — Storage: Firestore KB backend

> **Expected output:** the same collected data readable/writable via Firestore; `STORE=firestore` works; a
> jsonl→firestore migration script.
> **Demo:** `STORE=firestore python -m src.collector` (against the Firestore emulator) · `python -m src.store.migrate`.
> **Acceptance:** `pytest -m m0_6` green — the store contract test passes for **both** jsonl and firestore backends.

- [ ] **T0.6.1 Firestore store** — `src/store/firestore_store.py` implementing `store/base.py` (collection
      `items` keyed `repo#number`; collection `state`).
      Test: `tests/test_store_contract.py` — **one contract test parametrized over jsonl + firestore** so both
      satisfy identical assertions; firestore param uses the **Firestore emulator**, marked `integration`.
- [ ] **T0.6.2 Store factory** — `src/store/__init__.py::get_store()` selects impl via env `STORE=jsonl|firestore`.
      Test: `tests/test_store_factory.py` — env selects the right class (firestore import mocked).
- [ ] **T0.6.3 Migration** — `python -m src.store.migrate` (jsonl → firestore).
      Test: `tests/test_store_migrate.py` (`integration`) — sample jsonl → docs present in emulator.

## M1 — Intelligence plane (inner loop)

> **Expected output:** LLM-classified items (taxonomy + evidence) in the KB; versioned `taxonomy@v` + `policy@v`;
> timestamped calibrated forecasts; per-category trend series; an LLM-written **cited** weekly report.
> **Demo:** `python -m src.analyze` (classify new items) · `python -m src.report` (cited report) ·
> `python -m src.forecast` (log predictions).
> **Acceptance:** `pytest -m m1` green · the report's every claim line carries ≥1 evidence URL.

- [ ] **T1.1 Embeddings + vector index** — `src/embed.py` (embed text/labels; NN search; backend TBD).
      Test: `tests/test_embed.py` — with a deterministic fixture/mock model, NN of a query returns the
      semantically closer of two docs.
- [ ] **T1.2 Taxonomy schema + versioning** — `src/taxonomy.py` (`taxonomy@vN` in KB, active pointer).
      Test: `tests/test_taxonomy.py` — v1 → add category → v2; both retrievable; `active` returns v2.
- [ ] **T1.3 Policy object (versioned)** — `src/policy.py` (scoring weights, prompt templates, active taxonomy ref).
      Test: `tests/test_policy.py` — versions are append-only/immutable; `get_active()` returns latest.
- [ ] **T1.4 Analyst agent** — classify delta items into taxonomy via `llm.complete`; write labels + evidence to KB.
      Test: `tests/test_analyst.py` — mock `llm.complete` → item updated with category + citation preserved.
- [ ] **T1.5 Forecaster agent** — emit calibrated predictions `{claim, resolution_rule, prob, due_date, evidence}`.
      Test: `tests/test_forecaster.py` — mock llm → stored prediction validates against schema (prob∈[0,1], due>now).
- [ ] **T1.6 Reporter v1 (LLM, cited)** — weekly report written from classified items.
      Test: `tests/test_reporter_v1.py` — mock llm → **every claim line has ≥1 evidence URL** (evidence principle).
- [ ] **T1.7 Trend series** — `src/trends.py`: per-category activity time series from KB.
      Test: `tests/test_trends.py` — synthetic items across weeks → correct bucketed counts per category.
- [ ] **T1.8 RAG evaluation guardrail (guardrail 2: a trustworthy score)** — `src/rag_eval.py` +
      `tests/rag_eval/golden.jsonl` (hand-labeled query → relevant issue/PR ids). Compute **retrieval** metrics
      (Recall@k, MRR, nDCG@k) and **generation** metrics (faithfulness / groundedness via LLM-as-judge,
      citation-accuracy, hallucination-rate on absent-topic queries). Enforce thresholds (e.g. Recall@10 ≥ 0.8,
      hallucination_rate = 0) and write scores to the KB each run for drift tracking.
      Test: `tests/test_rag_eval.py` — offline: metric math on a fixed ranked list (known Recall@k/MRR/nDCG),
      every claim carries a citation, an absent-topic query ⇒ "no evidence"; live retrieval + LLM-judge =
      `@pytest.mark.integration`.

## M2 — Outer loop: grading + candidate discovery

> **Expected output:** grading metrics (precision/recall/Brier) on matured forecasts → `policy@v+1`; taxonomy
> evolution; an engine×capability parity matrix; a **risk-ranked candidate queue**; cost-aware provider bandit;
> novelty filter.
> **Demo:** `python -m src.grade` (score past predictions) · `python -m src.candidates` (ranked queue with risk).
> **Acceptance:** `pytest -m m2` green · candidates come out ranked with risk tiers + evidence links.

- [ ] **T2.1 Grader** — `src/agents/grader.py`: resolve matured predictions vs reality (merged / in release /
      adopted); compute precision/recall + Brier.
      Test: `tests/test_grader.py` — synthetic predictions + outcomes → known metric values.
- [ ] **T2.2 Policy update from grades** — propose `policy@vN+1` from grading results.
      Test: `tests/test_policy_update.py` — a low-precision category → its weight decreases in the new version.
- [ ] **T2.3 Curator** — `src/agents/curator.py`: propose new / retire dead taxonomy categories from activity.
      Test: `tests/test_curator.py` — category with no activity for N weeks → flagged retire; a novel cluster →
      proposed new category.
- [ ] **T2.4 Parity matrix** — `src/parity.py`: engine × capability with evidence + gap flags.
      Test: `tests/test_parity.py` — synthetic capability signals → matrix cell populated + "present in fork,
      missing upstream" gap flagged.
- [ ] **T2.5 Candidate discovery + risk ranking** — `src/agents/scout.py`: candidates (ROCm-reproducible /
      good-first-issue / parity gap), risk-tiered.
      Test: `tests/test_scout.py` — fixture items → ranked candidates with risk tier + evidence present.
- [ ] **T2.6 Cost-aware LLM provider selection (bandit)** — `src/llm_bandit.py`: a UCB-style bandit over the
      `LLM_PROVIDER` options (claude_cli / claude_api / local-vLLM) using per-call reward (task success) vs
      cost/latency from T0.7's metadata; agents ask the policy which provider to use. *(Borrowed from ShinkaEvolve.)*
      Test: `tests/test_llm_bandit.py` — synthetic reward/cost history → bandit prefers the best reward-per-cost
      provider; an unseen provider still gets explored.
- [ ] **T2.7 Novelty / dedup filter before expensive evaluation** — `src/novelty.py`: reject a candidate *before*
      a costly MI250 build if it is a near-duplicate of a prior attempt (embedding similarity ≥ threshold) or an
      LLM-as-novelty-judge rules it redundant. Gates T3.2/T3.3. *(Borrowed from ShinkaEvolve — the biggest
      sample-efficiency lever.)*
      Test: `tests/test_novelty.py` — near-duplicate candidate rejected; a genuinely new one passes (embedding +
      judge mocked).

## M3 — Contribution plane (MI250 verification oracle) — human-gated

> **Expected output:** a reproduced bug signal on MI250; a **verified** patch (signal flips) on a fork branch;
> ensemble self-review votes; a human-gate artifact; a **draft PR** (only after approval).
> **Demo:** `python -m src.engineer --candidate <id>` (repro→patch→verify on mi250-05x) prints verified=true/false;
> the human gate opens a draft PR only with `--approve`.
> **Acceptance:** `pytest -m m3` green · (integration) a real MI250 run yields a verified patch · **T3.6 = a real
> draft PR URL pasted in the checklist.**

- [ ] **T3.1 Remote runner** — `src/runner.py`: run a command on `mi250-05x` over ssh, stream logs, capture
      exit code + artifacts.
      Test: `tests/test_runner.py` — mock subprocess/ssh → correct command composed + result parsed. Real ssh =
      `integration` (runs only when `MI250_HOST` set).
- [ ] **T3.2 Repro harness** — given a candidate, run repro on MI250, capture failing signal → KB `runs`.
      Test: `tests/test_repro.py` — mock runner returns a failing log → failing signal recorded.
- [ ] **T3.3 Engineer patch loop** — `llm.complete` generates a patch on a fork branch → rebuild/test on MI250
      → confirm signal flips.
      Test: `tests/test_engineer.py` — mock llm+runner: fail→patch→pass ⇒ `verified=True`; fail→patch→fail ⇒
      `verified=False` and **no PR**.
- [ ] **T3.4 Ensemble self-review gate** — before the human gate, run N independent adversarial self-critiques of
      the verified patch (multi-sample vote) via `llm.complete`; require a majority "looks correct" **in addition
      to** the MI250 pass. Only patches passing **both** the hardware verify (T3.3) and self-review advance.
      *(Borrowed from The AI Scientist's ensemble reviewer — beat single-reviewer reliability.)*
      Test: `tests/test_self_review.py` — mock llm votes: majority-approve ⇒ advance; split/reject ⇒ hold (no gate).
- [ ] **T3.5 Human gate** — assemble `{diff, risk badge, repro evidence, MI250 logs, self-review votes}`;
      `gh pr create --draft` **only** after an explicit approve flag.
      Test: `tests/test_gate.py` — unapproved ⇒ `gh` never called; approved ⇒ `gh` invoked (subprocess mocked).
      **HARD: nothing reaches upstream without approval.**
- [ ] **T3.6 First real PR** (lowest risk: docs/typing/test-only) through the gate.
      Test: manual/`integration` — **draft PR URL pasted here**; checked only then.
      > note: PR URL = …

## M4 — Orchestration / always-on (on ce-master, tmux)

> **Expected output:** an always-on loop on `ce-master` under tmux; per-stage run events **+ live
> heartbeats/intermediate output** in the KB; run-locking.
> **Demo:** `python -m src.orchestrator --once` (one dry-run tick) · tmux runbook in `docs/`.
> **Acceptance:** `pytest -m m4` green · a dry-run tick completes and writes a run event **+ a heartbeat**.

- [ ] **T4.1 Orchestrator** — `src/orchestrator.py`: pin active `policy@v`, route deltas → agents, enforce
      cadences (data plane **daily**, `config.COLLECT_INTERVAL_HOURS=24` / intel daily+weekly / contribution
      triggered).
      Test: `tests/test_orchestrator.py` — fake clock + fake agents → correct routing per cadence; gate respected.
- [ ] **T4.2 Scheduler + tmux runbook** — launch on `ce-master` under tmux (cron/systemd); runbook in `docs/`.
      Test: `tests/test_schedule_dryrun.py` — a dry-run tick runs end-to-end without error (agents stubbed).
- [ ] **T4.3 Locking / idempotency** — overlapping runs don't double-write.
      Test: `tests/test_locking.py` — second concurrent run backs off; no duplicate KB writes.
- [ ] **T4.4 Run events** — per-stage run records (stage, status, counts, duration) to KB for the dashboard.
      Test: `tests/test_events.py` — a pipeline tick writes a run event with the expected fields.
- [ ] **T4.5 Liveness: heartbeat + intermediate output (know what's running)** — every stage/agent writes its
      lifecycle to KB `runs`: `started` → periodic `heartbeat` (current step + a rolling tail of intermediate
      output) → `finished` / `failed`, each timestamped with the active `policy@v`. A run is **stalled** if its
      last heartbeat is older than `N × expected_interval`; a crash must leave a `failed`/`stalled` record (never
      silent). Powers the T5.8 health panel.
      Test: `tests/test_liveness.py` — a stage emits started→heartbeat→finished with monotonic timestamps; a stale
      heartbeat is classified `stalled`; an exception path records `failed` (nothing left silently "running").

## M5 — Dashboard (Firestore-backed; monitoring + trends + parity)

> **Expected output:** a local web dashboard reading the KB — a **live health / "what's running now" view**
> (which stage is active, alive/stalled, current step, intermediate output), monitoring panels, trend charts,
> engine×capability parity heatmap, pipeline data-flow diagram, guardrail panels, and the patch review pane.
> **Demo:** `python -m dashboard` (or `streamlit run dashboard/app.py`) → open the printed localhost URL.
> **Acceptance:** `pytest -m m5` green · panels/charts render from a seeded KB · the health view shows a running
> stage as active and a stale one as `stalled`.

- [ ] **T5.1 Read layer** — `dashboard/api.py`: read-only access the UI consumes (items/trends/predictions/
      candidates/parity/runs).
      Test: `tests/test_dashboard_api.py` — seeded store → each endpoint returns the expected shape.
- [ ] **T5.2 Monitoring panels** — collection stats, taxonomy timeline, prediction scoreboard, candidate queue.
      Test: `tests/test_dashboard_panels.py` — panel data functions return expected shapes from fixtures.
- [ ] **T5.3 Trend visualizations** — category momentum over time.
      Test: `tests/test_dashboard_trends.py` — series endpoint returns bucketed points.
- [ ] **T5.4 Parity diagram** — engine × capability heatmap/matrix; each cell links to evidence.
      Test: `tests/test_dashboard_parity.py` — matrix endpoint returns cells + evidence + gap flags.
- [ ] **T5.5 Pipeline data-flow diagram** — architecture view of the running system.
      Test: `tests/test_dashboard_diagram.py` — diagram data builds from live stage metadata (smoke).
- [ ] **T5.6 Review pane (folds in M3 gate)** — diff + risk + approve/hold in the same UI.
      Test: `tests/test_dashboard_review.py` — approve action flips candidate status (gate mocked).
- [ ] **T5.7 Guardrail panels** — data-quality (reconciliation delta, gap ratio, collector error-rate over time)
      and RAG-eval scores (Recall@k, faithfulness, hallucination-rate) with drift lines + threshold markers.
      Test: `tests/test_dashboard_guardrails.py` — seeded metrics → panels return series + threshold flags.
- [ ] **T5.8 Live health / "what's running now" panel** — reads T4.5 events: per stage/agent show
      **running / idle / stalled / failed** (from heartbeat age), the current step, elapsed time, and a live tail
      of intermediate output; auto-refresh; a top-level green/red health badge.
      Test: `tests/test_dashboard_health.py` — seeded run events → a running stage renders active with its step +
      output tail; a stale heartbeat renders `stalled`; a `failed` event renders red.

---

## Test index (quick run)

```bash
.venv/bin/python -m pytest                 # full offline suite (must be green)
.venv/bin/python -m pytest tests/test_collector.py   # one module
.venv/bin/python -m pytest -m integration  # live: GitHub / emulator / MI250 / LLM (needs creds/hardware)
```
