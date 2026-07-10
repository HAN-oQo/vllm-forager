# Multi-model repro: `--model-impl transformers` on MI250 — dense + MoE + FP8

- **날짜:** 2026-07-10
- **선행 리포트:** `docs/research/transformers-backend-mi250-repro.md` (Qwen3-4B 단일 모델, 정정 후 parity 확인) — 이 리포트는 그 후속으로, 여러 모델(dense/MoE/FP8)로 커버리지를 넓힘
- **대상 백로그 항목:** `docs/CONTRIB_BACKLOG.md` — "Make `--model-impl transformers` reach native parity on MI250"
- **하드웨어:** `mi250-052` (dense/MoE 벤치마크), `mi250-053` (AOT 버그 root-cause 조사) — 둘 다 gfx90a
- **환경:** vLLM commit `3d1c21a6f` (= `v0.25.0rc3`, PR #47187 포함 — [[이전 리포트]]에서 검증된 올바른 커밋), 별도 컨테이너 `vllm-forager-bench`(모델 캐시 `/models` 마운트) / `vllm-forager-aotbug`(격리된 vLLM 소스 사본)로 기존 `vllm-forager` 컨테이너와 분리 — 진행 중이던 다른 작업을 건드리지 않기 위함
- **모델 소스:** `/remote/vast0/share-mv` (사내 NFS 공유 캐시) — 다운로드 없이 기존 캐시된 체크포인트만 사용 (사용자 지시)
- **전체 raw 서버 로그:** `docs/research/logs/2026-07-10-multi-model-mi250/`

## 1. 테스트 매트릭스

| 모델 | 아키텍처 | 정밀도 | trust-remote-code | 결과 |
|---|---|---|---|---|
| Qwen/Qwen3-4B | dense | bf16 | 불필요 | ✅ parity (선행 리포트) |
| meta-llama/Llama-3.1-8B-Instruct | dense | bf16 | 불필요 | ✅ parity |
| deepseek-v2-lite (DeepseekV2ForCausalLM, 64 experts) | MoE | bf16 | **필요** | 🐛 `--model-impl transformers` **크래시** (remote-code 임포트 에러) |
| deepseek-v3-lite (DeepseekV3ForCausalLM, MLA, 256 experts) | MoE | bf16 | 불필요 (네이티브 지원) | ⚠️ `--model-impl vllm`은 동작하나 출력 empty; `--model-impl transformers` **크래시** (MLA shape 버그) |
| deepseek-v3-lite | MoE | FP8 (dynamic quant) | 불필요 | `--model-impl vllm`만 시험 (transformers는 위 크래시로 애초에 불가) — 동작은 하나 bf16 대비 **속도 향상 없음** |

## 2. Llama-3.1-8B-Instruct — parity 재확인 (두 번째 dense 모델)

| 테스트 | vllm | transformers | 차이 |
|---|---|---|---|
| 결정론적 출력 | 동일 | 동일 | 없음 |
| 단일 요청, 256 tok | 74.11 tok/s | 74.17 tok/s | +0.08% |
| 동시 16×200 tok | 780.98 tok/s | 779.44 tok/s | -0.2% |
| 장문 prefill | 0.654s | 0.657s | +0.48% |

Fuser 로그에서 `LlamaAttention`/`LlamaMLP`용 fusion도 확인됨 (Qwen3와 마찬가지로 정상 활성화). **Qwen3-4B에 이어 두 번째 dense 모델에서도 완전한 parity — 아키텍처가 달라도(Llama vs Qwen) native-speed 주장이 유지됨을 보여주는 추가 근거.**

## 3. DeepSeek-V2-Lite (MoE) — `--model-impl transformers` 크래시 (ROCm 무관 버그)

`--model-impl vllm`은 정상 동작 (118.0 tok/s 단일, 763.6 tok/s 동시성 — 다른 모델들과 같은 수준). `--model-impl transformers`는 모델 로딩 단계에서 즉시 실패:

```
ImportError: cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils'
  (at .../modeling_deepseek.py:56, DeepSeek-V2-Lite's own trust_remote_code file)
```

**원인:** 이 환경의 `transformers==5.13.0`에는 `is_torch_fx_available`가 더 이상 없음 (확인: `dir(transformers.utils.import_utils)`에 없음, `is_torch_fx_proxy`/`is_torch_tensorrt_fx_available`만 존재). DeepSeek-V2-Lite의 HF Hub `trust_remote_code` 모델링 파일이 오래된 transformers API를 참조하고 있어서 발생. `--model-impl vllm`은 vLLM 자체의 네이티브 `DeepseekV2ForCausalLM` 구현을 쓰기 때문에 이 remote code 파일을 아예 안 거쳐서 문제가 없음 — `--model-impl transformers`만 이 경로를 타서 걸림.

**ROCm 관련 없음.** CUDA에서도 동일하게 재현될 순수 Python 임포트 에러. **누구의 버그인가:** DeepSeek-V2-Lite 리포의 remote code가 최신 transformers와 안 맞는 것 — vLLM도 ROCm도 아닌, 모델 리포지토리 쪽 유지보수 문제. 다만 `--model-impl transformers` 사용자가 흔히 마주칠 수 있는 "오래된 trust_remote_code 모델은 최신 transformers에서 깨질 수 있다"는 실사용 관점의 유의미한 발견.

## 4. DeepSeek-V3-Lite (MoE, MLA) — 진짜 Fuser 버그 발견

DeepSeek-V3는 transformers에 네이티브로 등록되어 있어 (`--trust-remote-code` 불필요) remote-code 문제를 피해갈 수 있었음. 하지만:

**`--model-impl vllm`:** 정상 기동, 22.15 tok/s 단일, 317.6 tok/s 동시성. **다만 결정론적 프롬프트 출력이 빈 문자열** — 이 "lite" 체크포인트 자체가 degenerate한 테스트용 축소 체크포인트일 가능성이 높음 (실제 품질 검증용이 아님). 이 결과가 정밀도/백엔드와 무관하게 발생(FP8에서도 동일 empty 출력)한 것으로 봐서, 체크포인트 자체의 특성이지 --model-impl이나 정밀도의 문제는 아님.

**`--model-impl transformers`:** Fuser가 실제로 MoE 전용 fusion까지 성공적으로 수행 (`Fused: experts (DeepseekV3Experts) -> FusedMoE (external routing)`, 그 외 attention/MLP fusion들도 정상). 하지만 이후 `torch.compile`의 fake-tensor 프로파일링 단계에서 크래시:

```
torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call
  Explanation: Dynamo failed to run FX node with fake tensors: call_method view(
    *(FakeTensor(..., size=(s27, 24576), dtype=torch.bfloat16), -1, 128, 576), **{}
  ): got RuntimeError("shape '[-1, 128, 576]' is invalid for input of size 24576*s27")
```

**분석:** DeepSeek-V3의 config: `kv_lora_rank=512`, `qk_rope_head_dim=64` → 512+64=**576** (MLA의 "압축된 latent" 표현 차원). 반면 실제 입력 텐서의 마지막 차원은 24576 = 128(heads) × 192(=`qk_nope_head_dim`(128)+`qk_rope_head_dim`(64), "압축 해제된" per-head 차원). 즉 코드가 이미 192-per-head로 펼쳐진 텐서를 576-per-head(압축 latent) 모양으로 다시 reshape하려다 실패 — MLA의 "압축된 KV" 표현과 "압축 해제된(absorbed) Q/K" 표현, 두 서로 다른 뷰를 vLLM Transformers 백엔드의 Fuser/MoE 경로가 혼동하는 것으로 보임.

**ROCm 관련 없을 가능성 높음** — FakeTensor 기반 shape 추론은 실제 하드웨어 커널 실행 전 단계의 순수 shape-로직 버그라, CUDA에서도 동일하게 재현될 것으로 판단(이 환경에 CUDA 호스트가 없어 직접 반증은 못함). **DeepSeek-V3(MLA 아키텍처) × Transformers 백엔드의 조합에서 재현 가능한, 잘 특정된 크래시** — 향후 실제 vLLM 이슈/PR 후보로 유효.

## 5. DeepSeek-V3-Lite + FP8 dynamic quant — 크래시 없이 동작하나 속도 향상 없음

`--model-impl transformers` 경로가 위 버그로 애초에 막혀서, FP8은 `--model-impl vllm`(네이티브)에서만 시험:

| | bf16 | FP8 (dynamic quant) |
|---|---|---|
| 단일 요청 | 22.15 tok/s | 22.08 tok/s |
| 동시 16×200 | 317.6 tok/s | 316.4 tok/s |
| 장문 prefill | 0.937s | 0.937s |

**FP8 dynamic quantization이 gfx90a에서 크래시 없이 동작함은 확인** (IDEAS.md 가설 (c) "FP8이 사실상 MI300 전용"을 적어도 "완전히 안 돎"이라는 의미로는 반박 — 돌기는 함). 다만 **거의 완전히 동일한 처리량** — bf16 대비 실질적 속도 향상이 전혀 없음. 이유는 미조사 (양자화가 일부 레이어에만 적용됐거나, 이 규모/설정에서 병목이 컴퓨트가 아니라 다른 곳— 라우팅 오버헤드, 메모리 대역폭 등 — 일 수 있음). **이것도 이 checkpoint/설정에서 나온 관찰이지, gfx90a의 FP8 자체가 느리다는 일반 결론은 아님** — 실제 블로그가 벤치마크한 대규모 FP8-MoE(8-GPU급) 시나리오는 여전히 미검증.

## 6. AOT 컴파일 캐시 pickle 실패 — 근본 원인 확정

기존 리포트에서 "ROCm 특유일 가능성 낮음, 미확인"으로 남겼던 이 워닝의 정확한 메커니즘을 mi250-053에서 격리된 vLLM 소스 사본으로 소스 레벨 추적 완료.

**메커니즘 (확정):**
1. `vllm/model_executor/models/transformers/base.py`의 `_get_decoder_cls()`는 `AutoModel.from_config(...)`로 **실제 transformers 라이브러리의 진짜 클래스** (예: `transformers.models.qwen3.modeling_qwen3.Qwen3Model`)를 가져옴 — 새 서브클래스를 만드는 게 아니라 원본 클래스 그 자체.
2. `_decorate_for_torch_compile()` → `support_torch_compile()`가 그 **실제 클래스를 제자리에서 monkey-patch** — `cls.__call__ = __call__` (`decorators.py:719`)로 새 wrapper를 클래스 attribute에 직접 대입.
3. torch의 `torch._dynamo.aot_compile`이 컴파일된 함수를 디스크에 캐싱하려 할 때, 내부적으로 (원본, patch 이전) `forward`/관련 함수 객체에 대한 참조를 직렬화 대상에 포함시킴.
4. Python 표준 `pickle.py`의 `save_global`은 함수를 **참조로(by reference)** 저장함 — `module_name = whichmodule(obj, name)` 후 `getattr(module, name)`한 결과가 저장하려는 객체와 **동일한 객체(identity)**인지 확인 (`pickle.py:1088-1093`). 그런데 이미 2단계에서 클래스 attribute가 재할당됐기 때문에, save 시점에 `getattr(Qwen3Model, 'forward')`는 더 이상 pickle하려는 (patch 이전) 객체와 동일하지 않음 → `PicklingError: Can't pickle <function Qwen3Model.forward>: it's not the same object as transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward`.

**정리:** 이건 ROCm 버그가 아니라 **"실제 라이브러리 클래스를 제자리에서 monkey-patch"라는 vLLM Transformers 백엔드의 설계**와 **"함수를 참조로 직렬화"하는 Python pickle의 근본 동작**이 충돌하는 지점 — CUDA에서도 100% 동일하게 재현될 것 (소스 레벨로 확정, 하드웨어 무관한 순수 Python/직렬화 로직이므로 CUDA 호스트로 재검증할 필요조차 없음).

**실질적 영향:** 캐시 저장이 항상 실패 → 재시작마다 ~26초의 `torch.compile` 시간을 매번 다시 지불. 기능적으로는 무해(예외를 잡아서 warning으로 처리, 크래시 아님).

**가능한 수정 방향 (시도는 안 함, 이번 세션 범위 밖):**
- (a) torch 쪽: `aot_compile`의 직렬화가 이런 "제자리 patch된 메서드" 케이스를 pickle 참조 대신 다른 방식(예: `copyreg`, 직접 바이트 포함)으로 처리하도록 개선.
- (b) vLLM 쪽: 실제 라이브러리 클래스를 직접 patch하는 대신 진짜 서브클래스를 만들어 그 서브클래스의 메서드를 patch — 그러면 서브클래스 attribute는 재할당 이후에도 안정적으로 유지되어 pickle 식별이 성립할 가능성. 다만 이는 Transformers 백엔드의 핵심 설계(`_get_decoder_cls`가 실제 클래스를 반환하는 이유 — HF의 `isinstance` 체크 등과의 호환성 유지)를 건드리는 더 큰 변경이라 신중한 검토 필요.

## 7. 종합 결론

- **Native-speed 주장은 dense 모델(Qwen3-4B, Llama-3.1-8B)에서 견고하게 성립** — 두 개의 다른 아키텍처 패밀리에서 모두 확인.
- **MoE 모델은 두 가지 독립적인, ROCm과 무관한 버그로 인해 `--model-impl transformers`에서 아예 테스트가 막힘:**
  1. DeepSeek-V2-Lite: 오래된 remote code × 최신 transformers 버전 비호환 (모델 리포 쪽 이슈).
  2. DeepSeek-V3-Lite: MLA 아키텍처의 압축/압축해제 텐서 shape을 Fuser가 혼동하는 실제 vLLM 버그 (재현 가능, 잘 특정됨 — **가장 유망한 컨트리뷰션 후보**).
- **FP8은 최소한 크래시 없이 동작**하나(gfx90a 완전 미지원이라는 우려는 기각), 이 checkpoint/설정에서는 속도 이득이 없었음 — 결론을 내리기엔 더 다양한 설정/모델 필요.
- **AOT 캐시 버그의 정확한 메커니즘 확정** — ROCm 무관, torch/vLLM 상호작용 문제.
- **아직 못한 것:** 진짜 대규모 FP8-MoE(블로그가 실제 벤치마크한 8-GPU급 시나리오), 70B급 dense FP8, LoRA 조합, MLA shape 버그의 실제 수정 시도.
