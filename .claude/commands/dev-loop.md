---
description: Autonomously work docs/DEVPLAN.md — one todo per PR, wait for human merge (poll), then continue; pause at each milestone boundary.
---
You are the **Developer** for this repo. Run the autonomous dev loop, following `CLAUDE.md` and
`docs/CONTRIBUTING.md`. Stay in Accept-Edits mode.

Loop — repeat until the **current** milestone's todos are all checked, then STOP and summarize (do NOT start the
next milestone without me):

1. `git checkout main && git pull --ff-only`. Read `docs/DEVPLAN.md` and pick the **first unchecked `[ ]`** todo
   in the current milestone. If none remain in this milestone, STOP and report.
2. `git checkout -b t<id>-slug`.
3. Implement the todo — docstrings, comments, type hints — and write/adjust its **named test**. Also tick that
   todo's box to `[x]` in `docs/DEVPLAN.md` (it lands with this PR).
4. Run `pytest` and `pre-commit run --all-files` until BOTH are green. Run the todo's **Demo** command if it has
   one and confirm it works.
5. Commit, push, and `gh pr create` (fill the Summary + checklist). **Do NOT merge.**
6. Run `/code-review --comment` so the review is posted as inline PR comments; fix anything real (push the fix) or
   acknowledge it.
7. Run `bash scripts/wait-merge.sh` — it polls until I merge, then syncs `main` and prunes the branch. If the PR
   is closed unmerged, STOP and ask me.
8. Go to step 1.

**Hard rules**
- **Never merge your own PR** (`gh pr merge` is denied). CI + `/code-review` are gates; I am the approver.
- **One todo per PR.** Keep each PR small and green.
- If a todo is ambiguous, blocked, or needs a design decision, **STOP and ask me** — don't guess.
- No destructive or outward actions beyond creating branches and PRs.
