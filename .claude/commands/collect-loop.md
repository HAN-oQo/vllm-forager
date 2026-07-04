---
description: One collection cycle — run the collector, report health, and on failure self-heal by opening a fix PR. Schedule with `/loop <interval> /collect-loop`.
---
You are the **Collector operator** for this repo. Do ONE collection cycle, then stop (the
scheduler or `/loop` calls you again next interval). Stay in Accept-Edits mode. Follow
`CLAUDE.md` and `docs/CONTRIBUTING.md`.

1. `git checkout main && git pull --ff-only`; activate `.venv`. If `GITHUB_TOKEN` isn't set,
   warn me (rate limit 60/hr) but continue.
2. **Ping start**, then run the collector and watch the streamed progress:
   `bash scripts/notify.sh "collect started" "vllm-forager:collect"` → `python -m src.collector`.
3. **On success:** send an end ping, then report one line — new items + total (`python -m src.stats`):
   `bash scripts/notify.sh "collect done: <new items + total>" "vllm-forager:collect"`. Then stop.
4. **On failure,** send an end ping
   (`bash scripts/notify.sh "collect FAILED: <transient | fix PR #n>" "vllm-forager:collect"`),
   then classify from the traceback / log:
   - **Transient** (rate limit, network, 5xx, GitHub outage, missing token): change nothing;
     note it and let the next cycle retry.
   - **Data the code should tolerate** (e.g. a corrupt/partial JSONL line): prefer a
     *deterministic code fix* (guard/skip) over a one-off — it will recur.
   - **A real code bug:** on a `triage/collector-fix` branch, fix it + add/adjust a test, get
     `pytest` and `pre-commit` green, commit, open a PR (`gh`, title `fix(collector): …`),
     then run `/code-review --comment` on it. **Do NOT merge.** One open `triage/*` PR at a
     time — if one already exists, don't open another.
5. **Never** merge (`gh pr merge` is denied) and **never** hot-patch then silently keep
   running — fixes land as PRs I review, and the next cycle picks them up once I merge.
   Report what you did in ≤3 lines.

The collector is incremental (resumes from `data/state.json`) and first-run windows to
`INITIAL_LOOKBACK_DAYS`, so a normal cycle is cheap.

> `scripts/notify.sh` is best-effort — a silent no-op if `NOTIFY_URL` isn't set (configure it in `.env`;
> use the same ntfy topic as the dev-loop merge pings so all alerts land in one place).
