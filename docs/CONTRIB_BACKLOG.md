# Contribution backlog — human-seeded, top-priority

> Operator-authored contribution ideas for the **vLLM-PR contribution agent** (the Engineer that works the
> candidate queue) — **not** the project dev-loop. Everything here **outranks every auto-discovered candidate**:
> the agent works these **first, top-to-bottom** (order = priority), then fills spare capacity with the scout's
> auto-discovered queue. Mechanism: `docs/IDEAS.md` → "Human-seeded candidates rank first".

## How to use
- Append an entry under **## Backlog**. **Order is priority** — highest at the top; reorder freely.
- Keep each entry **concrete**: a specific PR-worthy target on a specific repo, with a rationale and evidence links
  (not a system-capability idea — those go in `docs/IDEAS.md`).
- Update `status:` as it moves. Still gated: **feasibility on MI250 + novelty + the mandatory human PR gate** all
  still apply — an infeasible directed idea must *surface* that, never silently fail.

## Entry schema
```
### <short title>
- target: <owner/repo>
- status: idea | scoping | in-progress | draft-ready | done | dropped
- why: <one-line rationale>
- evidence: <links>
- notes: <optional — repro steps, blockers, dependencies>
```

## Backlog

### Make `--model-impl transformers` reach native parity on MI250 (ROCm / gfx90a)
- target: vllm-project/vllm
- status: scoping
- why: the Transformers backend was made "as fast as native" (#47187) but benchmarked **CUDA / H100 only**; on
  ROCm the Fuser's swapped-in fused kernels may be missing / slower / fall back to unfused on gfx90a → **parity is
  unverified**. Our MI250 is the only way to check — an error or a slowdown vs `--model-impl vllm` is the
  contribution.
- evidence: https://huggingface.co/blog/native-speed-vllm-transformers-backend · https://github.com/vllm-project/vllm/pull/47187
- notes: needs a vLLM ROCm env on a MI250 node first (none today — the M3 build step; `rocm/vllm` docker is the
  fast path). Smoke-test `vllm serve <small model> --model-impl transformers` vs `--model-impl vllm`; capture
  errors + a perf delta.

### Verify + fix LoRA × `--model-impl transformers` on MI250 (ROCm)
- target: vllm-project/vllm
- status: idea
- why: both the feature (#47187) and its LoRA-compat fix (#47832) were **CUDA-only validated**; the
  LoRA × transformers-backend × ROCm intersection is unchecked on gfx90a — a natural, unclaimed opening on the
  hardware we hold.
- evidence: https://github.com/vllm-project/vllm/pull/47832 · https://github.com/vllm-project/vllm/pull/47187
- notes: same MI250 vLLM-env dependency as the entry above; run a small model + a LoRA adapter under
  `--model-impl transformers` and compare against the native path.
