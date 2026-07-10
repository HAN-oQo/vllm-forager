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
     checkpoint — sync before spending GPU time]**
  2. smoke-test `vllm serve Qwen/Qwen3-4B --model-impl transformers` — does it start and serve at all?
  3. compare against `--model-impl vllm`: correctness + tokens/s (single, prefill, and **concurrent/batched**).
  4. if it errors or is slower, check the attention backend on gfx90a and whether the fused kernels fell back to
     unfused. **[mid-experiment checkpoint — report findings before proposing a fix]**
  5. propose fix direction(s) → sync → implement → verify on MI250 → fork + PR draft (human submits upstream).
- notes: **2026-07-09 smoke-test done** — full writeup + repro script:
  `docs/research/transformers-backend-mi250-repro.md`. Env existed already (`mi250-052`'s standing `vllm-forager`
  container, vLLM `0.23.1rc1.dev788`, ROCm/HIP `7.2`). Qwen3-4B, single GCD. Findings: no crash, numerically
  identical output at `temperature=0`, HIP Graphs capture fine (refutes the "graphs not wired" hypothesis) —
  but **-13.1% aggregate throughput under 16-way concurrency** (1170 → 1017 tok/s) vs `--model-impl vllm`, while
  single-request and prefill-only cases are within ~3% noise. That concurrency gap is the real, measurable
  contribution point (concurrent/batched throughput is exactly what the blog's "native speed" claim is about).
  Also found (likely NOT ROCm-specific, unconfirmed on CUDA in this infra): a non-fatal
  `vllm/compilation/decorators.py` AOT-compile-cache pickle failure specific to the transformers backend's
  dynamic model-class composition — repeats every restart, costs ~26s of repeated `torch.compile` each time.
  Next: profile *why* the 13% concurrency gap exists (kernel-level: is the Fuser's fused-kernel path actually
  activating on ROCm, or silently falling back to unfused?) before proposing a fix.

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
