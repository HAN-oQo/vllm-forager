# Contributing — Git & code-review rules

The short version of how change lands in this repo. Architecture: `docs/PLAN.md` · what to build next:
`docs/DEVPLAN.md` · day-to-day operation: `docs/RUNBOOK.md`.

## Unit of change
- **One PR = one todo** (`Txx`), or a small cluster of tightly-related tiny todos.
- A **milestone is an epic**, never a single PR (use a GitHub Milestone / tracking issue).
- **Direct to `main`** is allowed only for pure docs / tooling / typo edits.

## The review gate (3 layers)
1. **Automated — CI, always.** `.github/workflows/ci.yml` runs `pre-commit` (black / ruff / mypy / hygiene) +
   `pytest` on every PR. Must be green.
2. **Code review — every code PR.** Run **`/code-review --comment`** so findings are **posted on the PR as inline
   comments**; fix or acknowledge each. Use `/security-review` for security-sensitive diffs.
3. **Human merge.** CI and code review are **gates, not approvers** — a **human reads and merges**. Agents open
   PRs and push branches but **never self-merge** (`.claude/settings.json` denies `gh pr merge`; auto-merge off).

A checklist box in `DEVPLAN.md` is ticked only when its named test passes, `pre-commit` is clean, **and a human
has merged**.

## Code review — who & how
- The reviewer is **Claude + you**: Claude runs `/code-review --comment` on the PR (posts inline findings); you
  read them and merge.
- Default effort: **`medium`** for routine todos; **`high`** for M3 patch code and anything touching the collector
  or the MI250 oracle.
- `--comment` posts inline comments; `--fix` applies fixes to the working tree (re-review after).

## Upstream vLLM PRs (the product's output)
Separate, stricter flow: fork → **draft** PR → ensemble self-review (T3.4) + **mandatory human gate** → upstream.
Never conflated with internal dev PRs.

## Branch protection (set once — GitHub → Settings → Branches → `main`)
- ✅ Require status checks to pass before merging → select **CI**.
- ✅ Require a pull request before merging (blocks direct agent pushes).
- ✅ Do not allow force pushes.

This makes the human-in-the-loop merge **structural**, not just convention.
