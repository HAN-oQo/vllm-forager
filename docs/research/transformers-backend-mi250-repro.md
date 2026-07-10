# Repro: `--model-impl transformers` on MI250 (ROCm/gfx90a) — Qwen3-4B

- **날짜:** 2026-07-09
- **대상 백로그 항목:** `docs/CONTRIB_BACKLOG.md` — "Make `--model-impl transformers` reach native parity on MI250"
- **참고 근거:** [native-speed vLLM transformers backend (HF blog)](https://huggingface.co/blog/native-speed-vllm-transformers-backend), [vLLM #47187](https://github.com/vllm-project/vllm/pull/47187)
- **하드웨어:** `mi250-052` (AMD Instinct MI250, gfx90a) — vLLM ROCm 개발 컨테이너 `vllm-forager` (`rocm/vllm-dev:base`, vLLM `0.23.1rc1.dev788+gfa4321de3`, PyTorch `2.11.0` / HIP `7.2.53211`), 단일 GPU (`HIP_VISIBLE_DEVICES=0`, MI250 8-GCD 카드의 1개 GCD)

## 1. Issue overview

블로그 글은 vLLM의 Transformers 백엔드(`--model-impl transformers`)가 "네이티브 속도"에 도달했다고 주장하지만, 벤치마크는 **8×H100 (CUDA)** 에서만 수행됨. CUDA Graphs, `torch.compile`, 런타임에 꽂히는 fused 커널(#47187의 fx-그래프 기반 "Fuser")이 핵심 메커니즘인데, 이 중 어느 것도 ROCm에서 검증된 적이 없음. `docs/IDEAS.md`가 제기한 리스크 가설:

- (a) 런타임에 꽂히는 attention 백엔드가 ROCm에서 다르게 동작 (실패 또는 unfused로 폴백)
- (b) CUDA Graphs → HIP Graphs가 이 코드패스에 안 붙어있을 수 있음
- (c) FP8은 사실상 MI300 전용 기능 — gfx90a에서 FP8-MoE 경로 자체가 안 돌 가능성
- (d) fused 커널(`MergedColumnParallelLinear`/`QKVParallelLinear`)이 ROCm 빌드를 필요로 함

## 2. Approach

1. `mi250-051/052/053` 중 유휴 노드 확인 (051은 타 워크로드가 8-GPU 전부 점유 중이어서 052 사용).
2. 052의 기존 `vllm-forager` 컨테이너(이미 vLLM ROCm dev 빌드 설치됨) 안에서 `Qwen/Qwen3-4B`를 두 가지 모드로 각각 기동:
   - `--model-impl vllm` (baseline)
   - `--model-impl transformers`
3. 두 서버에 대해 동일한 세 가지 테스트를 순서대로 실행 (동시에 두 서버를 띄우지 않고, 하나씩 종료 후 재기동 — 공유 노드에 대한 VRAM 점유를 최소화):
   - 결정론적(`temperature=0`) 짧은 프롬프트 → **정확도** 비교
   - 단일 요청, 256 토큰 생성 → **단일-요청 처리량**
   - **동시 16개 요청**, 각 200 토큰 생성 → **동시성 처리량**
   - ~1560 토큰 프롬프트, 32 토큰만 생성 → **prefill 위주 지연시간**
4. 각 실행 로그에서 크래시/폴백/워닝 여부 확인.

## 3. Reproduce (그대로 재현 가능한 절차)

```bash
# MI250 노드 (gfx90a) — 컨테이너 안에서
cd /tmp/forager-repro   # 임의 작업 디렉터리

# --- baseline ---
HIP_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B --model-impl vllm \
    --port 8001 --gpu-memory-utilization 0.5 &

# 서버 기동 완료("Application startup complete") 대기 후:
curl -s http://localhost:8001/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-4B","prompt":"The capital of France is","max_tokens":32,"temperature":0}'

# 처리량/지연 측정은 아래 bench.py 사용 (requests 필요)
python3 bench.py 8001 baseline_vllm_impl

# --- 서버 종료 후 transformers 백엔드로 교체 ---
HIP_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B --model-impl transformers \
    --port 8002 --gpu-memory-utilization 0.5 &
python3 bench.py 8002 transformers_impl
```

`bench.py`는 단일-요청 256토큰 / 동시 16×200토큰 / 장문 prefill(1561 prompt tokens, 32 decode) 세 가지를 측정하는 ~50줄짜리 스크립트 (`requests` + `ThreadPoolExecutor`); 이 리포트와 함께 전달.

## 4. 결과

| 테스트 | `--model-impl vllm` | `--model-impl transformers` | 차이 |
|---|---|---|---|
| 결정론적 출력 (`"The capital of France is"`, 32 tok) | `" Paris. The capital of Germany is Berlin..."` | **byte-for-byte 동일** | 없음 — 수치적으로 동등 |
| 단일 요청, 256 tok | 2.140s (119.6 tok/s) | 2.215s (115.6 tok/s) | **-3.3%** (오차범위 근접) |
| 동시 16개 요청, 각 200 tok (합산 처리량) | 2.735s wall, **1170.2 tok/s** | 3.146s wall, **1017.0 tok/s** | **-13.1%** (측정 가능한 실제 격차) |
| 장문 prefill (1561 prompt tok, 32 decode) | 0.412s | 0.421s | -2.3% (오차범위 내) |
| CUDA/HIP Graphs 캡처 | — | PIECEWISE 51/51, FULL 35/35 모두 성공 | 가설 (b) **반박** — HIP Graphs 정상 동작 |
| 서버 크래시/에러 | 없음 | 없음 | 가설 (a)/(d) 이 조합에서는 **미관측** |

### 발견된 부수 이슈 (non-fatal warning)

`--model-impl transformers`로 기동할 때마다 다음 워닝이 뜸:

```
unable to save AOT compiled function to ...: Can't pickle <function Qwen3Model.forward at 0x...>:
it's not the same object as transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward
```

- **원인 추적:** `vllm/compilation/decorators.py:707-715`의 `save_aot_compiled_function`이 torch.compile AOT 캐시를 디스크에 저장하려다 실패. Transformers 백엔드는 `decoder_cls = type(model.get_decoder())` (`vllm/model_executor/models/transformers/base.py:232`)로 HF 디코더 클래스를 동적으로 감싸는데, 이 과정에서 컴파일된 `forward`의 객체 아이덴티티가 `transformers` 모듈이 원래 갖고 있는 `Qwen3Model.forward`와 달라져서 pickle의 "동일 객체인지" 체크에 걸림.
- **ROCm 전용 여부:** 소스 분석상 이 메커니즘은 torch.compile/dynamo의 범용 동작이라 **ROCm 특이 버그로 보이진 않음** — 다만 이 환경에 CUDA 호스트가 없어 직접 재현/반증은 못 했음 (정직하게 미확인 상태로 남김).
- **영향:** 캐시가 매번 저장 실패 → 재기동마다 ~26초의 torch.compile 시간을 매번 다시 지불 (이번 실행에서 실측). 기능적으로는 무해하지만, 반복 배포/재기동이 잦은 환경에서는 체감되는 비용.

### 아직 테스트 못한 것

- 블로그가 실제로 벤치마크한 **FP8-MoE 경로** (가설 c) — 훨씬 큰 모델과 긴 다운로드/실행 시간이 필요해 이번 세션에는 미포함.
- LoRA × `--model-impl transformers` 조합 (백로그의 두 번째 항목).
- 다른 MI250 GCD/노드에서의 재현성 (이번엔 052의 GCD 0 하나만 사용).

## 5. Outcome

**부분 검증(partial positive) — dense 소형 모델(Qwen3-4B)에서는 ROCm에서도 정확히 동작하지만, 동시성 처리량에서 ~13% 격차가 실측됨.**

- 크래시/수치 오류 없음 → 가설 (a)/(d)는 이 조합에서는 근거 없음 (다만 fused-kernel 경로가 실제로 활성화됐는지는 미확인 — 커널 프로파일링 없이는 "unfused로 조용히 폴백"과 "동작은 하지만 살짝 느림"을 구분 못함, 13% 격차가 바로 그 폴백의 흔적일 수 있음).
- 가설 (b) HIP Graphs 미지원 → **반박됨**.
- **새로운, 측정 가능한 발견:** 동시 요청 처리량에서 native `vllm` 대비 13.1% 낮음. 이게 바로 블로그가 주장하는 "네이티브 속도"의 핵심 시나리오(배치/동시성 하에서의 처리량)이므로, 가장 유의미한 격차.
- **별도 발견:** AOT 컴파일 캐시 저장 실패 (비ROCm 특이적일 가능성 높음, 별도의 작은 vLLM 버그로 독립 제출 가능).
