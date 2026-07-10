# Repro: `--model-impl transformers` on MI250 (ROCm/gfx90a) — Qwen3-4B

- **날짜:** 2026-07-09 (1차, invalid) / 2026-07-10 (정정 재시험)
- **대상 백로그 항목:** `docs/CONTRIB_BACKLOG.md` — "Make `--model-impl transformers` reach native parity on MI250"
- **참고 근거:** [native-speed vLLM transformers backend (HF blog)](https://huggingface.co/blog/native-speed-vllm-transformers-backend), [vLLM #47187](https://github.com/vllm-project/vllm/pull/47187)
- **하드웨어:** `mi250-052` (AMD Instinct MI250, gfx90a) — vLLM ROCm 개발 컨테이너 `vllm-forager` (`rocm/vllm-dev:base`, PyTorch `2.11.0` / HIP `7.2.53211`), 단일 GPU (`HIP_VISIBLE_DEVICES=0`, MI250 8-GCD 카드의 1개 GCD)
- **전체 raw 서버 로그:** `docs/research/logs/2026-07-09-transformers-backend-mi250/` (01~06, `_INVALID_pre47187` 접미사가 붙은 게 정정 전 잘못된 실행)

## 0. ⚠️ 정정 (Correction) — 1차 결과는 무효

**1차 시험(2026-07-09)은 vLLM `0.23.1rc1.dev788+gfa4321de3` (commit `290b0d801`, 2026-07-06)에서 돌렸는데, 이 커밋에는 애초에 이 리포트가 테스트하려던 기능인 [#47187 "Make the Transformers modeling backend as fast as native vLLM"]이 아직 안 들어가 있었음.** 즉 1차의 "-13.1% 동시성 격차"는 ROCm의 문제가 아니라 — Fuser 코드 자체가 없는 구버전끼리 비교한 것이었고, 이 실수는 사용자가 직접 잡아냄 (`vllm-project/vllm` 릴리스 태그 `v0.25.0rc3`를 근거로 지적).

**검증 방법:**
```bash
gh api repos/vllm-project/vllm/compare/290b0d801...v0.25.0rc3 --jq '.commits[].commit.message' \
  | grep -i transformers
# → "Make the Transformers modeling backend as fast as native vLLM (#47187)" 가
#   정확히 이 83-커밋 gap 안에 있었음 — 1차 테스트 환경엔 없었다는 확증.
```

`#47187`은 순수 Python 변경(컴파일된 `.cu`/`.cpp`/`.hip` 파일 0개)이라, 컨테이너 안에서 `git checkout v0.25.0rc3` 만으로 (재빌드 없이) 재시험 가능했음 — editable install (`pip install -e .`)이라 git 체크아웃만 바꿔도 바로 반영됨.

**정정 후 결론(§5 참고): 13% 격차는 사라짐 — Fuser가 실제로 활성화된 상태에서는 MI250에서도 native와 거의 동일하거나 오히려 근소하게 더 빠름.** 1차 데이터는 "잘못된 버전을 테스트하면 이렇게 된다"는 사례로 아래에 그대로 남겨둠(삭제하지 않음), 각 표에 `INVALID`로 표시.

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

**정정 재시험 전 반드시 먼저 할 것 (§0 교훈):**
```bash
# 테스트하려는 vLLM PR/기능이 설치된 커밋에 실제로 있는지부터 확인
cd /workspace/vllm && git log -1 --oneline
gh api repos/vllm-project/vllm/compare/<현재커밋>...<원하는 태그/커밋> \
  --jq '.commits[].commit.message' | grep -i <키워드>
# 없으면: 순수 Python 변경인지 확인(컴파일 파일 0개) 후 git checkout으로 전환
#   (editable install이면 재빌드 불필요; 아니면 재빌드 필요 — 시간 소요 큼)
```

**모든 서버 raw 로그는 앞으로도 매 실험마다 통째로 보관** — `docs/research/logs/<날짜>-<주제>/`에 번호를 매겨 저장 (예: `01-baseline_vllm_impl.log`), 무효로 판명난 실행도 삭제하지 않고 `_INVALID_<이유>` 접미사만 붙여 남김 (이번 사례처럼 "잘못된 버전 테스트"도 나중에 재현/검증에 쓸모 있음).

`bench.py`는 단일-요청 256토큰 / 동시 16×200토큰 / 장문 prefill(1561 prompt tokens, 32 decode) 세 가지를 측정하는 ~50줄짜리 스크립트 (`requests` + `ThreadPoolExecutor`); 이 리포트와 함께 전달.

## 4. 결과

### 4a. 1차 — `INVALID` (commit `290b0d801`, #47187 없음)

| 테스트 | `--model-impl vllm` | `--model-impl transformers` | 차이 |
|---|---|---|---|
| 결정론적 출력 (`"The capital of France is"`, 32 tok) | `" Paris. The capital of Germany is Berlin..."` | byte-for-byte 동일 | 없음 |
| 단일 요청, 256 tok | 2.140s (119.6 tok/s) | 2.215s (115.6 tok/s) | -3.3% |
| 동시 16개 요청, 각 200 tok (합산 처리량) | 2.735s wall, 1170.2 tok/s | 3.146s wall, 1017.0 tok/s | **-13.1%** |
| 장문 prefill (1561 prompt tok, 32 decode) | 0.412s | 0.421s | -2.3% |

→ **위 표는 무효.** Fuser 코드 자체가 없는 커밋이라 "13% 격차"는 애초에 비교 대상이 아니었음 (§0 참고).

### 4b. 정정 재시험 — `v0.25.0rc3` (commit `3d1c21a6f`, #47187 포함, Fuser 실제 활성화 확인됨)

기동 로그에서 Fuser가 실제로 동작하는 것도 직접 확인:
```
Fused: q_proj + k_proj + v_proj (self_attn: Qwen3Attention) -> qkv_proj (QKVParallelLinear)
Fused: q_norm (Qwen3RMSNorm) -> RMSNorm (CustomOp)
Fused: gate_proj + up_proj (mlp: Qwen3MLP) -> gate_up_proj (MergedColumnParallelLinear)
... (전체: docs/research/logs/.../06-transformers_impl_v0.25.0rc3_CORRECTED.log)
```

| 테스트 | `--model-impl vllm` | `--model-impl transformers` | 차이 |
|---|---|---|---|
| 결정론적 출력 (`"The capital of France is"`, 32 tok) | `" Paris. The capital of Germany is Berlin..."` | byte-for-byte 동일 | 없음 — 수치적으로 동등 |
| 단일 요청, 256 tok | 2.136s (119.9 tok/s) | 2.116s (121.0 tok/s) | **+0.9%** (transformers가 근소하게 더 빠름) |
| 동시 16개 요청, 각 200 tok (합산 처리량) | 2.737s wall, 1169.1 tok/s | 2.708s wall, **1181.9 tok/s** | **+1.1%** (transformers가 근소하게 더 빠름) |
| 장문 prefill (1561 prompt tok, 32 decode) | 0.408s | 0.412s | -1.0% (오차범위 내) |
| CUDA/HIP Graphs 캡처 | — | PIECEWISE 51/51, FULL 35/35 모두 성공 | 가설 (b) **반박** — HIP Graphs 정상 동작 |
| Fuser 실제 활성화 | — | 로그에 fusion 이벤트 확인됨 (위 발췌) | 가설 (a)/(d) **반박** — fused 커널이 실제로 ROCm에서 돎 |
| 서버 크래시/에러 | 없음 | 없음 | — |

**세 지표 모두 오차범위(±1~2%) 안에서 사실상 동률 — 오히려 transformers 쪽이 근소 우위.** baseline 자체도 1차/2차에서 거의 동일한 수치(119.6→119.9 tok/s, 1170.2→1169.1 tok/s)를 냈으므로, 측정 환경/방법 자체는 안정적이었고 차이는 순전히 vLLM 커밋 차이(Fuser 유무)에서 왔음이 확인됨.

### 발견된 부수 이슈 (non-fatal warning)

`--model-impl transformers`로 기동할 때마다 다음 워닝이 뜸:

```
unable to save AOT compiled function to ...: Can't pickle <function Qwen3Model.forward at 0x...>:
it's not the same object as transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward
```

- **원인 추적:** `vllm/compilation/decorators.py:707-715`의 `save_aot_compiled_function`이 torch.compile AOT 캐시를 디스크에 저장하려다 실패. Transformers 백엔드는 `decoder_cls = type(model.get_decoder())` (`vllm/model_executor/models/transformers/base.py:232`)로 HF 디코더 클래스를 동적으로 감싸는데, 이 과정에서 컴파일된 `forward`의 객체 아이덴티티가 `transformers` 모듈이 원래 갖고 있는 `Qwen3Model.forward`와 달라져서 pickle의 "동일 객체인지" 체크에 걸림.
- **ROCm 전용 여부:** 소스 분석상 이 메커니즘은 torch.compile/dynamo의 범용 동작이라 **ROCm 특이 버그로 보이진 않음** — 다만 이 환경에 CUDA 호스트가 없어 직접 재현/반증은 못 했음 (정직하게 미확인 상태로 남김).
- **영향:** 캐시가 매번 저장 실패 → 재기동마다 ~26초의 torch.compile 시간을 매번 다시 지불 (이번 실행에서 실측). 기능적으로는 무해하지만, 반복 배포/재기동이 잦은 환경에서는 체감되는 비용.

## 5. Outcome

**긍정 검증(validated) — 실제로 Fuser가 포함된 커밋(`v0.25.0rc3`, #47187 이후)에서는 dense 소형 모델(Qwen3-4B) 기준 MI250이 native `vllm` 백엔드와 사실상 동률, 오히려 근소하게 더 빠름.** "네이티브 속도" 주장은 이 조합에서 ROCm에서도 **성립**하는 것으로 확인됨.

- 가설 (a) attention 백엔드 실패/폴백 → **관측 안 됨**, 정상 동작.
- 가설 (b) HIP Graphs 미지원 → **반박됨**.
- 가설 (d) fused 커널이 ROCm 빌드 필요 → **반박됨** — 로그로 fusion이 실제 활성화된 것 직접 확인.
- 가설 (c) FP8-MoE 경로 → **여전히 미검증** (아래 참고).
- **1차 실패의 교훈 (중요, 앞으로도 지킬 것):** 벤치마크를 시작하기 전에 **설치된 커밋이 테스트하려는 기능을 실제로 포함하는지부터 확인**해야 함 — 버전 문자열(`vllm.__version__`)만 보고 판단하면 안 됨 (이번에도 `git checkout` 후에도 캐시된 버전 문자열이 그대로 남아 헷갈릴 뻔함; `importlib.metadata.version()`도 마찬가지로 stale일 수 있음 — 실제 소스 파일 존재 여부나 `git log`/`git describe`로 직접 확인하는 게 확실함).
- **별도 발견 (여전히 유효):** AOT 컴파일 캐시 저장 실패 워닝 — 이건 Fuser 유무와 무관하게 재현됨 (위 "발견된 부수 이슈" 참고). 비ROCm 특이적일 가능성 높음, 별도의 작은 vLLM 버그로 독립 제출 가능.

### 아직 테스트 못한 것 (변동 없음)

- 블로그가 실제로 벤치마크한 **FP8-MoE 경로** (가설 c) — 훨씬 큰 모델과 긴 다운로드/실행 시간이 필요해 이번 세션에는 미포함.
- LoRA × `--model-impl transformers` 조합 (백로그의 두 번째 항목).
- 다른 MI250 GCD/노드에서의 재현성 (이번엔 052의 GCD 0 하나만 사용).
- 더 큰 모델/배치에서도 이 동률이 유지되는지 (Qwen3-4B는 작은 모델이라 fused-kernel 이득이 상대적으로 작을 수 있음).
