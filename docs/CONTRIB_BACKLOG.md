# Contribution backlog — human-seeded, top-priority

> Operator-authored contribution ideas for the **vLLM-PR contribution agent** (the Engineer that works the
> candidate queue) — **not** the project dev-loop. Everything here **outranks every auto-discovered candidate**:
> the agent works these **first, top-to-bottom** (order = priority), then fills spare capacity with the scout's
> auto-discovered queue. Mechanism: `docs/IDEAS.md` → "Human-seeded candidates rank first".

## How to use
- Append an entry under **## Backlog**. **Order is priority** — highest at the top; reorder freely.
- Keep each entry **concrete**: a specific PR-worthy target on a specific repo, with a rationale and evidence links
  (not a system-capability idea — those go in `docs/IDEAS.md`).
- Optionally add **`steps:`** — an ordered playbook the agent should follow ("check this, then try that, and if it
  fails, look at X"). The agent treats it as the plan skeleton and **syncs with you at checkpoints** as it goes
  (`docs/IDEAS.md` → "Checkpointed human–agent collaboration").
- Update `status:` as it moves. Still gated: **feasibility on MI250 + novelty + the mandatory human PR gate** all
  still apply — an infeasible directed idea must *surface* that, never silently fail.

## Entry schema
```
### <short title>
- target: <owner/repo>
- status: idea | scoping | in-progress | draft-ready | done | dropped
- why: <one-line rationale>
- evidence: <links>
- steps: <optional — ordered directives the agent should follow, synced at checkpoints>
- notes: <optional — blockers, dependencies>
```

## Backlog

### Make `--model-impl transformers` reach native parity on MI250 (ROCm / gfx90a)
- target: vllm-project/vllm
- status: in-progress
- why: the Transformers backend was made "as fast as native" (#47187) but benchmarked **CUDA / H100 only**; on
  ROCm the Fuser's swapped-in fused kernels may be missing / slower / fall back to unfused on gfx90a → **parity is
  unverified**. Our MI250 is the only way to check — an error or a slowdown vs `--model-impl vllm` is the
  contribution.
- evidence: https://huggingface.co/blog/native-speed-vllm-transformers-backend · https://github.com/vllm-project/vllm/pull/47187
- steps:
  1. stand up / reuse a vLLM ROCm env on a MI250 node (a standing `vllm-forager` container exists on `mi250-052`;
     else the prebuilt `rocm/vllm` docker image, or build `PYTORCH_ROCM_ARCH=gfx90a`, ROCm 6.3+). **[approach
     checkpoint — sync before spending GPU time]** — done, reused the standing container.
  2. smoke-test `vllm serve Qwen/Qwen3-4B --model-impl transformers` — does it start and serve at all? — done,
     starts cleanly.
  3. compare against `--model-impl vllm`: correctness + tokens/s (single, prefill, and **concurrent/batched**).
     — done for Qwen3-4B; **verify the installed commit actually contains the feature under test before
     trusting any number here** (see notes — this bit the agent once already).
  4. if it errors or is slower, check the attention backend on gfx90a and whether the fused kernels fell back to
     unfused. **[mid-experiment checkpoint — report findings before proposing a fix]** — not needed for
     Qwen3-4B (no gap found); still applies if FP8-MoE or a larger model surfaces one.
  5. propose fix direction(s) → sync → implement → verify on MI250 → fork + PR draft (human submits upstream).
     — nothing to fix yet for the dense-model case; revisit once FP8-MoE/larger-model/LoRA legs are run.
- notes: **2026-07-09 smoke-test → CORRECTED 2026-07-10** — full writeup + repro script + raw logs:
  `docs/research/transformers-backend-mi250-repro.md`, `docs/research/logs/2026-07-09-transformers-backend-mi250/`.
  Qwen3-4B, single GCD on `mi250-052`.

  **2026-07-09 first pass was invalid** (caught by the operator, not by the agent): the standing
  `vllm-forager` container's checked-out commit (`290b0d801`, 2026-07-06) **predated #47187** — the "-13.1%
  concurrency gap" it reported was two versions *without* the Fuser being compared to each other, not a real
  ROCm gap. Verified via `gh api .../compare/290b0d801...v0.25.0rc3` — #47187 is one of 83 commits in that
  gap. #47187 is pure Python (0 compiled `.cu`/`.cpp`/`.hip` files), so switching to `v0.25.0rc3` needed only a
  `git checkout` (editable install), no rebuild.

  **2026-07-10 corrected re-test** (`v0.25.0rc3`, Fuser confirmed actually activating via fusion log lines):
  numerically identical output at `temperature=0`; HIP Graphs capture fine (refutes "graphs not wired");
  single-request **+0.9%**, 16-way concurrent **+1.1%**, prefill **-1.0%** — all within noise, transformers
  backend at parity or a hair faster than native `vllm` on MI250. **The "native speed" claim holds on ROCm for
  this case.** Baseline numbers were stable across both passes (confirms the measurement method itself was
  sound; only the *version under test* was wrong the first time).

  Still found (unrelated to the version bug, reproduces on both commits): a non-fatal
  `vllm/compilation/decorators.py` AOT-compile-cache pickle failure in the transformers backend's dynamic
  model-class composition — likely not ROCm-specific (unconfirmed on CUDA, no CUDA host in this infra), costs
  ~26s of repeated `torch.compile` every restart.

  **Process lesson (apply to every future MI250 repro):** verify the installed commit actually contains the
  feature under test *before* benchmarking — a version string (`vllm.__version__` / `importlib.metadata`) can
  be stale after a bare `git checkout` in an editable install; check `git log`/`git describe` or the source file
  directly. Archive full raw server logs for every experiment run going forward (not just excerpts), even ones
  that turn out invalid — label them, don't delete them.

  Remaining scope (unchanged, still open): FP8-MoE path (gfx90a may not support FP8 at all — the blog's actual
  benchmarked scenario, needs a much bigger model); larger models/batches (Qwen3-4B is small, fused-kernel gains
  may show up more at scale); LoRA combination (see entry below).

### Verify + fix LoRA × `--model-impl transformers` on MI250 (ROCm)
- target: vllm-project/vllm
- status: idea
- why: both the feature (#47187) and its LoRA-compat fix (#47832) were **CUDA-only validated**; the
  LoRA × transformers-backend × ROCm intersection is unchecked on gfx90a — a natural, unclaimed opening on the
  hardware we hold.
- evidence: https://github.com/vllm-project/vllm/pull/47832 · https://github.com/vllm-project/vllm/pull/47187
- steps:
  1. reuse the MI250 vLLM env from the entry above.
  2. load a small model + a LoRA adapter under `--model-impl transformers`; exercise it (including streaming).
  3. compare against the native path; capture any error or parity gap. **[checkpoint before fixing]**
- notes: same MI250 vLLM-env dependency as the entry above.
