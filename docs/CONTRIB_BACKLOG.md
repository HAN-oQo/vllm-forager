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

  **2026-07-10 multi-model expansion** (operator-directed: test several models beyond the blog's own, using
  only locally-cached checkpoints under `/remote/vast0/share-mv`, no downloads) — full writeup:
  `docs/research/multi-model-transformers-backend-mi250-repro.md`,
  `docs/research/logs/2026-07-10-multi-model-mi250/`. Ran on isolated containers
  (`vllm-forager-bench` on `mi250-052`, `vllm-forager-aotbug` on `mi250-053`) so as not to disturb whatever the
  standing `vllm-forager` container might be mid-use for elsewhere.

  - **Llama-3.1-8B-Instruct (dense, different family from Qwen)**: parity confirmed again (+0.08% single,
    -0.2% concurrent, +0.48% prefill — all noise). Second independent confirmation the native-speed claim
    holds on ROCm for dense models.
  - **DeepSeek-V2-Lite (MoE)**: `--model-impl transformers` crashes at load — its `trust_remote_code` file
    imports `is_torch_fx_available`, removed from this env's `transformers==5.13.0`. Not our bug (or ROCm's) —
    the model repo's remote code is stale relative to current transformers; `--model-impl vllm` never hits it
    since it uses vLLM's own native implementation.
  - **DeepSeek-V3-Lite (MoE + MLA, natively supported, no remote-code needed)**: `--model-impl vllm` works
    (though returns empty output for the deterministic prompt — likely this specific "lite" checkpoint being a
    degenerate test-scale one, reproduces identically under FP8 too, so not a backend/precision artifact).
    `--model-impl transformers` **crashes during torch.compile fake-tensor tracing** — a genuine, well-scoped
    Fuser bug: `kv_lora_rank(512)+qk_rope_head_dim(64)=576` (MLA's compressed-latent per-head dim) gets used to
    reshape a tensor already sized for `qk_nope_head_dim(128)+qk_rope_head_dim(64)=192` per head (the
    decompressed Q/K view) — the Fuser conflates MLA's two different head-dim representations. Promoted to its
    own entry below — **the most promising contribution candidate found so far**.
  - **FP8 dynamic quant on DeepSeek-V3-Lite** (native path only — transformers-impl blocked by the crash above):
    runs without crashing (refutes "FP8 doesn't work at all on gfx90a" as a blanket claim) but showed **no
    throughput improvement** over bf16 (22.08 vs 22.15 tok/s single, 316 vs 318 tok/s concurrent) — cause
    unexplored; still doesn't test the blog's actual large-scale (8-GPU) FP8-MoE scenario.
  - **AOT-cache pickle bug — root cause now confirmed** (traced on an isolated vLLM source copy on
    `mi250-053`, not shared with the active benchmark work on `mi250-052`): `_get_decoder_cls()`
    (`transformers/base.py`) returns the **real** transformers class (e.g. `Qwen3Model`), not a synthesized
    subclass; `support_torch_compile()` then monkey-patches `cls.__call__` **in place** on that real, shared
    class (`decorators.py:719`). Something in `torch._dynamo.aot_compile`'s serialization embeds a reference to
    the pre-patch `forward`; Python's own `pickle.save_global` (`pickle.py:1088-1093`) saves functions **by
    reference** and requires `getattr(module, name) is obj` to hold — which fails because the class attribute
    has already been reassigned by save-time. Confirmed non-ROCm: this is a pure Python/pickle identity
    mechanism, hardware-independent — doesn't need a CUDA host to verify. Promoted to its own entry below.

  Remaining scope (unchanged, still open): the blog's actual large-scale FP8-MoE scenario (8-GPU, no complete
  local checkpoint available); 70B-class dense FP8; LoRA combination (see entry below); actually attempting a
  fix for the MLA shape bug or the AOT-cache identity mismatch (see the two new entries below).

### Fix DeepSeek-V3 (MLA) shape crash in the Transformers-backend Fuser
- target: vllm-project/vllm
- status: scoping
- why: `--model-impl transformers` crashes on any MLA-architecture model (confirmed: DeepSeek-V3-Lite) during
  torch.compile fake-tensor tracing, before any hardware kernel runs — likely reproduces on CUDA too, but it's a
  real, currently-uncrashable path for an entire architecture family on our own MI250. A concrete, well-scoped
  bug with a root cause already identified (see notes) — the strongest current candidate for an actual PR.
- evidence: `docs/research/multi-model-transformers-backend-mi250-repro.md` §4 (§ "3" in that file) + raw log
  `docs/research/logs/2026-07-10-multi-model-mi250/06-deepseek-v3-lite_transformers_impl_FAILED_shape_bug.log`
- steps:
  1. read `vllm/model_executor/models/transformers/moe.py`/`fuser.py`/`base.py`'s MLA-specific reshape logic and
     find exactly where a `576`-per-head (compressed latent: `kv_lora_rank`+`qk_rope_head_dim`) view gets applied
     to a tensor already shaped for `192`-per-head (decompressed: `qk_nope_head_dim`+`qk_rope_head_dim`).
     **[checkpoint — confirm the exact faulty line before touching anything]**
  2. write a minimal, MI250-independent repro (a unit test constructing the two tensor shapes directly, no full
     model load) to isolate the bug from the rest of the serving stack.
  3. propose + implement a fix on a fork branch; re-verify DeepSeek-V3-Lite serves correctly end-to-end on
     MI250 under `--model-impl transformers` (correctness + no crash). **[checkpoint before fork-push]**
  4. fork + PR draft (human submits upstream) — this one isn't ROCm-specific, so frame the PR as a general
     Transformers-backend/MLA fix, not a ROCm-only patch.
- notes: not yet attempted — this entry captures the root-cause analysis from the 2026-07-10 multi-model repro;
  next session should start at step 1/2.

### Fix (or upstream) the AOT-compile-cache pickle failure in the Transformers backend
- target: vllm-project/vllm (possibly pytorch/pytorch, depending on where the real fix belongs)
- status: scoping
- why: every `--model-impl transformers` restart repeats ~26s of `torch.compile` because the AOT cache never
  successfully saves — root cause fully traced (see notes), not ROCm-specific, small in scope but a real,
  reproducible dev-experience cost for anyone using this backend repeatedly (e.g. our own dashboard-restart-style
  workflows).
- evidence: `docs/research/multi-model-transformers-backend-mi250-repro.md` §5 (§ "6" in that file);
  `docs/research/transformers-backend-mi250-repro.md`'s original discovery
- steps:
  1. decide where the fix belongs: (a) vLLM should build a real subclass instead of monkey-patching the live
     transformers class in `_get_decoder_cls`/`support_torch_compile` (bigger, touches HF `isinstance`
     compatibility assumptions), or (b) torch's `aot_compile` serialization should handle a monkey-patched
     method without requiring pickle-by-reference identity to hold (arguably the more correct fix, but touches
     PyTorch internals, not vLLM). **[checkpoint — this decision changes which repo the PR targets]**
  2. build a minimal repro outside the full vLLM stack (monkey-patch a class method, `torch.compile` it,
     attempt to save/reload) to confirm the mechanism in isolation before proposing anything upstream.
  3. propose + implement whichever fix direction was chosen; verify the AOT cache actually round-trips (saves
     once, loads on next restart, no repeated 26s compile) on MI250.
  4. fork + PR draft (human submits upstream) — again not ROCm-specific in framing.
- notes: not yet attempted — root cause fully understood (see the linked report's §5/§6), no fix code written
  yet. Lower priority than the MLA shape bug above since this one is a performance/dev-experience cost, not an
  outright crash. An earlier pass at this same bug guessed the root cause was a *dynamically composed* decoder
  class — that guess was imprecise; `_get_decoder_cls` actually returns the real transformers class,
  monkey-patched in place, not a synthesized subclass (confirmed by source-level tracing above).

### Verify + fix LoRA × `--model-impl transformers` on MI250 (ROCm)
- target: vllm-project/vllm
- status: scoping
- why: both the feature (#47187) and its LoRA-compat fix (#47832) were **CUDA-only validated**; the
  LoRA × transformers-backend × ROCm intersection is unchecked on gfx90a — a natural, unclaimed opening on the
  hardware we hold.
- evidence: https://github.com/vllm-project/vllm/pull/47832 · https://github.com/vllm-project/vllm/pull/47187
- steps:
  1. reuse the MI250 vLLM env from the entry above.
  2. load a small model + a LoRA adapter under `--model-impl transformers`; exercise it (including streaming).
  3. compare against the native path; capture any error or parity gap. **[checkpoint before fixing]**
- notes: same MI250 vLLM-env dependency as the entry above.
