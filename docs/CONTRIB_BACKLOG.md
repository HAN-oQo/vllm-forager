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

### Fix DeepSeek-V3 (MLA) shape crash in the Transformers-backend
- target: vllm-project/vllm
- status: draft-ready — correctness + speed re-verified on two real (non-degenerate) checkpoints, default
  (compiled) execution; not yet pushed to the fork or opened as a PR (human will run + submit — see step 8)
- why: `--model-impl transformers` crashed on any MLA-architecture model (confirmed: DeepSeek-V3-Lite) during
  torch.compile fake-tensor tracing, before any hardware kernel runs — reproduces independent of ROCm (pure
  Python shape logic). **Now fixed and verified working on MI250** — see notes. Three bugs total, found and
  fixed in sequence: the original shape crash (steps 1-3), a silent head-scrambling bug found only when testing
  a real trained model instead of a degenerate test checkpoint (step 5), and a CUDA-graph-specific correctness
  bug found only under default (non-eager) execution (step 6).
- evidence: `docs/research/mla-transformers-backend-fix-mi250.md` (full writeup: root cause, two failed
  attempts, the working fix, the num_kv_heads bug, the CUDA-graph root cause, multi-model re-verification), raw
  logs `docs/research/logs/2026-07-10-mla-fix-mi250/` (numbered `01`/`02` = failed attempts, `03` = working fix;
  `round2-multimodel/` = final re-verification on DeepSeek-V2-Lite + DeepSeek-V2 full).
  Fork branch (WIP, not yet a PR, not yet pushed with the latest fixes):
  `HAN-oQo/vllm@wip/mla-transformers-backend-head-size-v`.
- steps:
  1. traced the actual mechanism (source-level): every model funnels through ONE function,
     `ALL_ATTENTION_FUNCTIONS["vllm"] = vllm_attention_forward`
     (`vllm/model_executor/models/transformers/__init__.py`), which calls a plain `Attention.forward(query, key,
     value)` — never vLLM's native, compressed-latent `MLAAttention`. Confirmed by grepping the entire
     `transformers/` package: `MLAAttention` is never imported or constructed anywhere in it (double-checked on
     request — no mixin overrides `create_attention_instances` either). The crash: `create_attention_instances`
     sizes that plain `Attention` with `get_head_size()`, which for MLA models returns the *compressed* latent
     dim (576) — meant for the native absorbed path this backend never uses — while the actual `key`/`value` are
     already decompressed by HF's own `kv_b_proj` at 192 (key) / 128 (value) per head. **[done]**
  2. **First attempt (failed): `head_size_v`.** `Attention` already supports asymmetric Q/K vs V sizing via a
     `head_size_v` constructor param — passing `head_size=192`/`head_size_v=128` fixed the *original* crash, but
     surfaced a second one in KV-cache tensor allocation: `AttentionBackend.get_kv_cache_shape(num_blocks,
     block_size, num_kv_heads, head_size)` — the interface *every* backend (ROCm, FlashAttention, FlashInfer,
     CPU, ...) implements — takes one `head_size` with no `head_size_v` equivalent at all; native MLA sidesteps
     this entirely because `MLACommonBackend.get_kv_cache_shape` caches only the compressed latent (no K/V slots
     to be asymmetric between). Confirmed via two separate crashes (with and without `VLLM_MLA_DISABLE=1`) — see
     logs `01`/`02`. **[done, ruled out]**
  3. **Working fix: mirror vLLM's own native "naive" (`use_mla=False`) fallback exactly.**
     `DeepseekV2Attention` (`vllm/model_executor/models/deepseek_v2.py:558-609`) already solves this identical
     problem — no `head_size_v`, just pads V (128) up to Q/K's width (192) with zeros before calling `Attention`,
     then slices the output back down to 128 afterward. Ported that exact pattern into the generic
     `vllm_attention_forward` hook (pads whenever `value.shape[-1] < head_size`, a no-op for every non-MLA
     model) and set `create_attention_instances`'s `head_size` to the decompressed dim directly (no
     `head_size_v`). Two files changed, ~30 lines. **[done — verified on MI250]**
  4. **Verification (MI250, DeepSeek-V3-Lite):** serves successfully under `--model-impl transformers` (no
     crash). Correctness matches `--model-impl vllm` exactly (both return empty output for the deterministic
     prompt — a property of this specific degenerate "lite" test checkpoint, confirmed identical on both paths,
     not a backend difference). **Throughput — corrected after an initial measurement error** (first pass
     compared a cold-start prefill call, including one-time torch.compile/cudagraph-capture cost for that
     prompt length, against what happened to be a warm call on the other backend — reported as "-613%/7.1x
     slower"; re-measured with `bench.py` fixed to call `long_prefill` twice and report both): single-request
     **-11.8%**, 16-way concurrent **-8.0%** (both reproduce consistently across two independent runs) — but
     long-context prefill is **+2.4% (cold, i.e. noise) / -12.9% (warm/steady-state, i.e. transformers is
     slightly *faster*)**. MLA's compression exists to cut attention FLOPs/KV-cache memory for long sequences,
     but that advantage did not show up at this prompt length/model size in this measurement — untested at
     larger scale. **This is a correctness fix (crash → working); the performance picture is a modest,
     consistent regression for single/concurrent decode and roughly a wash for prefill, not the large
     regression first reported.** **[done, corrected]**
  5. **Second real bug, found testing DeepSeek-V2-Lite (a real trained model, not the degenerate 4-layer
     V3-Lite): `num_kv_heads` was wrong.** `create_attention_instances` left `num_kv_heads =
     get_num_kv_heads()`, which for any MLA model returns **1** (native's absorbed path is MQA-shaped once
     latent-compressed). But this backend's `key`/`value` are HF's own already-decompressed tensors — `num_heads`
     distinct heads, not 1. Passing 1 didn't crash; it silently scrambled head boundaries (garbled, not empty,
     output). Fix: also set `num_kv_heads = num_heads` inside the same `is_deepseek_mla` block (native's naive
     fallback already does this — `DeepseekV2Attention` constructs its `Attention` with
     `num_kv_heads=self.num_local_heads`, i.e. equal to `num_heads`). **[done]**
  6. **Third bug, the important one: correct in eager, still garbled under default (compiled) execution —
     even with fixes #3+#5 both applied.** `--enforce-eager` gave the correct answer (`" Paris."`); the exact
     same code under vLLM's default torch.compile+CUDA-graph path gave garbage
     (`"h，\n\n\n\n\n\n\n\n\n\n-\n..."`). Bisected with `--compilation-config '{"cudagraph_mode": "NONE"}'`
     (compile on, CUDA graph off): **correct** — this isolated the bug to CUDA-graph capture/replay specifically,
     not dynamo/torch.compile tracing. Then ruled out "pad/slice is inherently unsafe under CUDA graphs" as the
     cause: started vLLM's **own, unpatched** native naive fallback directly (`--model-impl vllm` +
     `VLLM_MLA_DISABLE=1`, zero code changes of ours) under default CUDA graphs — it worked correctly. Same
     pad/slice operations, same hardware, only difference: native's pad amount
     (`self.qk_head_dim - self.v_head_dim`) is a fixed `__init__`-time Python int, applied **unconditionally**
     every call, while our generic hook wrapped it in `if v_head_dim < head_size:` (needed so the hook is a
     no-op for non-MLA models sharing the same function). Removed the branch — `F.pad` with zero padding and a
     full-range slice are both no-ops when `v_head_dim == head_size`, so this changes nothing for non-MLA models
     — and reran under full default compile+CUDA-graph: correct, reproducible across 3 repeats and under 4
     concurrent requests (different batch-size CUDA-graph buckets). **Root cause: a Python-level data-dependent
     branch around a tensor-allocating op (`F.pad`), even one that always resolves the same way for a given
     layer, produces incorrect output on CUDA-graph replay; an unconditional allocation of the same op does not.**
     **[done — root-caused and fixed]**
  7. **Re-verified with the final code on two real (non-degenerate) checkpoints, correctness + speed, default
     (compiled) execution.** DeepSeek-V2-Lite (16B, TP=1) and DeepSeek-V2 full (236B MoE, TP=8), native
     (`--model-impl vllm`) vs patched (`--model-impl transformers`) side by side. Full writeup:
     `docs/research/mla-transformers-backend-fix-mi250.md` §9; raw results + bench scripts:
     `docs/research/logs/2026-07-10-mla-fix-mi250/round2-multimodel/`.
     - **Correctness:** 5 varied prompts (factual, explanation, code-gen, arithmetic, antonym) on both
       checkpoints — both backends produce coherent, sensible answers on both sizes (not just "no crash" or
       "empty string match" — actual answer quality, matching or exceeding native on some prompts). **[done]**
     - **Speed — lite (TP=1, 4096 ctx):** single-request -14.7%, 16-way concurrent **+15.4% (patched faster)**,
       long-prefill warm **-11.9% (patched faster)**, long-prefill cold +20.4% (noise).
     - **Speed — full 236B (TP=8, matched 768 ctx / 0.95 gpu-mem-util on both sides for fairness):**
       single-request -11.1%, 8-way concurrent -7.9% (patched slower on both, at this scale).
     - **⚠️ New finding, only visible at 236B scale: KV-cache memory.** Native started fine at its defaults
       (`--max-model-len 4096`, `gpu_memory_utilization=0.9`). Patched **could not start at those settings** —
       had to shrink to `--max-model-len 768` + `--gpu-memory-utilization 0.95` just to fit. Root cause: the
       naive fallback stores K and V uniformly at `head_size=192` (`2 × num_heads(16) × 192 = 6144`/token),
       vs native's single compressed latent (`kv_lora_rank+qk_rope_head_dim=576`/token) — **patched uses ~10.7×
       more KV-cache memory per token.** Invisible on the small "lite" checkpoint; on a model where weights
       alone dominate GPU memory, it directly caps the maximum servable context length (4096 vs 768 in this
       test). **This must be stated explicitly in the PR** — the fix is "crash → correct", not "on par with
       native," especially for long-context or memory-constrained deployments.
     - **Attempted, deferred:** Kimi-K2 (`moonshotai/Kimi-K2.5`/`.6`) confirmed via `config.json`
       (`kv_lora_rank`/`qk_nope_head_dim`/`qk_rope_head_dim`/`v_head_dim`) to be the same MLA architecture
       family — but the checkpoint is 555GB, exceeding this node's ~549GB total 8×MI250 VRAM even before
       KV-cache/activations. Needs a bigger node or a quantized checkpoint; not attempted this session.
  8. **Remaining before a real PR:** unit test (ideally no full model load — construct the shape mismatch
     directly, and ideally one that exercises full CUDA-graph capture so a regression on the
     branch-vs-unconditional distinction would be caught); push the final code to the fork branch (currently
     only local); PR draft (human submits upstream) — frame as a general Transformers-backend/MLA fix, not
     ROCm-specific, **explicitly disclosing the KV-cache memory tradeoff from step 7**; consider whether it's
     worth flagging the general pattern ("branching around a CUDA-graph region is unsafe even when the branch is
     call-invariant") as a separate, standalone report — this repo's naive fallback happened to dodge it by luck
     (never branches), but any other model code that *does* branch around an allocation inside a
     `@support_torch_compile` region could hit the same bug silently.
  - **Benchmarking lesson (apply going forward):** always distinguish cold (first call, may include
    compile/cudagraph-capture cost for a new shape) from warm (repeated call, steady-state) when timing a test
    that varies by prompt/sequence length — a single call risks silently comparing cold-vs-warm across two
    different runs.
  - **Correctness-testing lesson (apply going forward):** "no crash" and "matches the other backend" are not
    "correct" — DeepSeek-V3-Lite's empty-output match on both backends looked like a passing correctness check
    but was actually a degenerate 4-of-61-layer test checkpoint producing empty output regardless of backend.
    Only testing DeepSeek-V2-Lite (a real, full-layer checkpoint) surfaced the num_kv_heads and CUDA-graph bugs.
    Also: always test the default (compiled) execution path, not just `--enforce-eager` — eager-mode correctness
    does not imply compiled-mode correctness.
- notes: full investigation — two failed attempts, the pad/slice fix, the num_kv_heads bug, and the
  CUDA-graph/branching root cause — is documented in `docs/research/mla-transformers-backend-fix-mi250.md`,
  written deliberately to include the dead ends so a future session or reviewer doesn't have to re-discover any
  of this.

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
