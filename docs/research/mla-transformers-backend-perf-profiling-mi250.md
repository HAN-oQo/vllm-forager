# Profiling the Transformers-backend MLA fix's speed regression on MI250

- **날짜:** 2026-07-10
- **선행 리포트:** `docs/research/mla-transformers-backend-fix-mi250.md` — MLA 크래시 수정 + 정합성 검증. §9에서
  single-request decode가 두 모델 크기 모두에서 일관되게 -11~15% 느리다고 측정했음.
- **이 리포트의 질문:** 그 -11~15%는 정확히 어디서 나오는가? 내가 고친 pad/slice(attention) 로직 자체가
  원인인가, 아니면 다른 이유인가? **답: 대부분 다른 이유다 — attention 커널 비용은 거의 동일하고, 진짜
  원인은 MoE 라우터(softmax+topk)가 torch.compile에서 훨씬 덜 효율적으로 fuse된다.**
- **하드웨어:** `mi250-053`, 격리된 컨테이너 (`vllm-forager-aotbug`), DeepSeek-V2-Lite, 같은 GPU(GPU0)·같은
  포트에서 순차 측정 (GPU 간 열/대역폭 편차를 배제하기 위해 — 사용자 지적으로 방법론 수정).
- **원본 프로파일 트레이스 + 파싱 스크립트:** 이 세션의 스크래치패드에 있음 (raw `.pt.trace.json.gz` +
  `profiler_out_0.txt` — vLLM의 `/start_profile`·`/stop_profile` 엔드포인트로 수집. 필요시 재수집 가능,
  방법은 §1 참고).

## 0. 먼저 정리: "같은 GPU 순차 측정"이 왜 필요했나

이전 라운드(§9, `mla-transformers-backend-fix-mi250.md`)의 속도 비교는 native와 patched를 **서로 다른
GPU**(GPU0/GPU1)에 동시에 띄워서 쟀다. 사용자가 "왜 동일하게 안 했어?"라고 지적 — MI250은 물리적으로 다른
GCD라 열/대역폭 상태가 다를 수 있어 진짜 통제된 비교가 아니었다. 같은 GPU·같은 포트에서 **순차로** 3가지를
다시 쟀다 (각각 3회 반복, 안정적으로 재현):

| | tok/s (단일 요청, 256 decode) |
|---|---|
| native MLA (압축, 기본값) | ~120.5 |
| **native 자체의 naive fallback** (`VLLM_MLA_DISABLE=1`, 내 패치 전혀 없음) | ~124.4 |
| patched (Transformers backend + 내 pad/slice fix) | ~103.4 |

핵심: **native의 naive fallback(비압축, 내 패치와 동일한 알고리즘)이 오히려 native MLA(압축)보다 3.2% 더
빠르다.** 즉 "압축 latent가 비압축보다 빠르다"는 알고리즘 차이는 이 규모에서는 성립하지 않는다 — 그런데
patched는 **똑같은 알고리즘을 쓰는 native naive fallback보다도 -16.9% 느리다.** 이건 MLA 압축 여부의
문제가 아니라, Transformers 백엔드 wrapper 자체에서 나오는 오버헤드라는 뜻이다.

## 1. 프로파일링 방법

vLLM은 OpenAI 서버에 내장 프로파일러 엔드포인트가 있다 (`vllm/entrypoints/serve/profile/api_router.py`):
`--profiler-config '{"profiler": "torch", "torch_profiler_dir": "<dir>"}'`로 서버를 띄우면
`/start_profile`, `/stop_profile` POST로 torch profiler를 켜고 끌 수 있다. 방법:

```bash
# 서버를 프로파일러 설정과 함께 띄움 (예: native naive fallback)
VLLM_MLA_DISABLE=1 python3 -m vllm.entrypoints.openai.api_server \
  --model /models/deepseek-v2-lite --model-impl vllm --max-model-len 4096 --port 8005 \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/tmp/profile_naive", "torch_profiler_record_shapes": true}'

# 워밍업(컴파일/cudagraph 캡처 비용을 트레이스 밖으로) 한 번, 그 다음 프로파일 구간 시작
curl -s .../v1/completions -d '{"prompt": "...", "max_tokens": 32, ...}' > /dev/null
curl -s -X POST http://localhost:8005/start_profile
for i in 1 2 3; do curl -s .../v1/completions -d '{"prompt": "...", "max_tokens": 128, ...}' > /dev/null; done
curl -s -X POST http://localhost:8005/stop_profile
```

`torch_profiler_dir`에 `profiler_out_0.txt`(`prof.key_averages().table()` 형식, 커널별 집계 — 이번 분석에
바로 사용) + `.pt.trace.json.gz`(chrome-trace 형식 raw 데이터, 필요시 `chrome://tracing`이나
`torch.profiler`로 더 깊게 볼 수 있음)가 생긴다. **native naive fallback**과 **patched**에 대해 각각 같은
절차(워밍업 1회 + 128-토큰 요청 3회)로 두 트레이스를 수집해 커널 단위로 diff했다.

## 2. 핵심 발견: attention 커널 비용은 거의 동일 — MLA fix는 무죄

| 커널 | native-naive | patched | 차이 |
|---|---|---|---|
| `kernel_paged_attention_2d` (실제 attention 연산, 10368회) | 226.28ms | 223.62ms | **-1.2% (사실상 동일)** |
| `reshape_and_cache_kernel` (KV 캐시 쓰기) | 172.92ms | 173.47ms | +0.3% (노이즈) |

**내가 고친 pad/slice 로직이 들어간 attention 경로 자체는 native와 사실상 같은 비용이다.** 즉 §9에서 관측한
-11~15% 슬로우다운의 원인은 attention/MLA fix가 아니다.

## 3. 진짜 원인: MoE 라우터(softmax+topk)의 torch.compile fusion 격차

전체 decode-loop CUDA 시간(`execute_context_0(0)_generation_1(1)`, 프로파일 구간 전체를 감싸는 최상위
버킷): native-naive **2906ms** → patched **3547ms**, **차이 +641ms.** 이 641ms 중 가장 크고 명확한
단일 원인:

| 커널 (같은 논리 연산 — 라우터 softmax 준비) | native-naive | patched | 차이 |
|---|---|---|---|
| `triton_red_fused__softmax_exp_max_prepare_softmax_on...` (reduction, fused) | 50.16ms | — | |
| `triton_poi_fused__softmax__to_copy_arange_bitwise_no...` (pointwise, fused) | 42.79ms | — | |
| **native 소계** | **92.95ms** | | |
| `triton_per_fused__softmax_exp_prepare_softmax_online...` (persistent, 별도 커널 하나) | — | 466.22ms | |
| **patched 소계** | | **466.22ms** | **+373.27ms** |

**같은 논리 연산(MoE 라우터의 online-softmax 기반 top-k 게이팅 준비)이 native에서는 작은 커널 2개로 fuse
되어 93ms인데, patched에서는 fuse되지 않은 커널 하나로 466ms — 약 5배 비싸다.** `torch.topk` 자체
(`warpMergeSortTopK`)는 오히려 patched가 더 쌈 (93.5ms vs 143.6ms, -50ms) — 그러니 이건 "MoE 라우팅 전체가
비효율적"이 아니라 **딱 이 fusion 경계 하나**의 문제다.

**373ms는 전체 641ms 격차의 58%.** 나머지 약 42%(~268ms)는 inductor가 붙이는 `triton_poi_fused_N` /
`triton_red_fused_N` 같은 익명 fusion-group 번호가 컴파일마다 달라져서(이 커널 앞에 새 op 하나가 끼어들면
뒤따르는 그룹 번호가 전부 밀림) 이름만으로는 안전하게 대응 짝을 찾을 수 없었다 — 십중팔구 같은 근본 원인
(라우터 코드가 별도 fusion 경계로 쪼개지면서 그 앞뒤 RMSNorm/dispatch 커널들의 fusion 경계도 같이
바뀜)의 연쇄 효과로 보이지만, 이 레벨의 분석으로는 깔끔하게 분리해서 확증하지 못했다.

### 왜 이런 차이가 나는가 — 코드 레벨 원인

라우터의 softmax+top-k 계산은 **vLLM 코드가 아니라 `transformers` 라이브러리 자신의 일반 코드**다:

```python
# transformers/models/deepseek_v2/modeling_deepseek_v2.py, DeepseekV2TopkRouter.forward
scores = router_logits.softmax(dim=-1, dtype=torch.float32)
...
topk_weights, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
```

이건 평범한 PyTorch eager 코드 — Transformers 백엔드가 HF 모델을 통째로 감싸는 구조라, 이 라우터 forward도
그대로 dynamo에 의해 컴파일된다. 반면 vLLM의 네이티브 MoE 경로는 라우터 계산을 자기 자신의
`FusedMoE`/게이트 추상화(`scoring_func="softmax"`를 파라미터로 받는, `vllm/model_executor/models/deepseek_v2.py:375`
근처)를 통해 하는데, 손으로 튜닝된 이 경로가 inductor에서 훨씬 잘 fuse된다.

대조적으로, **experts(FFN) 계산 자체는 이미 커스텀 op로 감싸져 있다** —
`vllm/model_executor/models/transformers/moe.py`의 `TransformersMoERunner.forward()`가
`torch.ops.vllm.transformers_moe_forward`를 통해 experts forward를 호출 (주석: "we need to forward through
a custom op so the topk_ids can be transferred without interfering with cudagraphs"). **즉 experts 쪽은
이미 dynamo/cudagraph에 대해 불투명한 경계로 처리했는데, 라우터(softmax+topk) 쪽은 그렇게 하지 않았다** —
바로 이 비대칭이 이번에 발견된 격차의 구조적 원인으로 보인다.

## 4. 이건 내 MLA fix와 무관하다 — 별도의, 더 일반적인 문제

이 라우터-softmax fusion 격차는 **MLA와 아무 상관이 없다** — DeepSeek-V2-Lite가 MoE 모델이라서 우연히
같이 측정에 잡혔을 뿐, `--model-impl transformers`로 **어떤 MoE 모델이든**(Mixtral, Qwen-MoE 등, MLA 여부
무관) 실행하면 똑같이 겪을 가능성이 높다 (다른 MoE 모델로 아직 검증은 안 했음 — 다음 단계로 좋은 후보).

**그러니 이건 MLA PR에 묶지 말고 별도의 컨트리뷰션 후보로 다뤄야 한다.**

## 5. 성능을 고칠 수 있는 방법 — 우선순위대로

1. **(가장 유력) 라우터(softmax+topk)도 커스텀 op으로 감싸기.** `moe.py`의 `TransformersMoERunner`가
   이미 experts forward에 대해 이 패턴(`transformers_moe_forward` 커스텀 op)을 쓰고 있다 — 라우터의
   `gate(hidden_states)` 호출도 똑같이 커스텀 op 경계로 감싸면(예:
   `torch.ops.vllm.transformers_moe_gate`), dynamo/inductor가 이 부분을 불투명한 블랙박스로 취급하게 되어
   지금처럼 어설프게 fuse하다 마는 대신, vLLM이 직접 손으로 튜닝한 (또는 최소한 격리되어 예측 가능한) 커널
   하나로 대체할 여지가 생긴다. **가장 작고, 기존 코드 패턴을 그대로 재사용하는 수정.** 다만 커스텀 op으로
   감싸는 것 자체가 내부 연산까지 자동으로 더 빠르게 fuse해주는 건 아니므로 — 그 안에서 실제로 vLLM의
   기존 최적화된 게이트 커널을 호출하도록 구현해야 진짜 효과가 있다.
2. **정밀 진단이 더 필요함:** 위 1번을 시도하기 전에, 정확히 어떤 요인이 이 fusion을 막는지 (fp32 업캐스트?
   `sorted=False` 옵션? 그룹 게이팅 분기? 아니면 그냥 이 특정 shape/reduction 패턴에 대한 inductor의 일반
   휴리스틱 한계?) 좀 더 좁혀볼 필요가 있다 — `TORCH_LOGS=inductor` 나 `torch._inductor.config.trace`로
   두 컴파일의 실제 생성된 Triton 소스를 나란히 비교하면 답이 나올 것.
3. **`transformers` 라이브러리 쪽 이슈일 수도 있다.** 라우터 코드 자체가 `transformers`의 것이므로, inductor의
   일반적인 fusion 휴리스틱이 이 특정 코드 패턴(softmax + `dtype=torch.float32` 업캐스트 + 조건부
   `topk_method` 분기)에 약한 것이라면, vLLM이 아니라 PyTorch/inductor 쪽에 리포트하는 게 맞을 수도 있다 —
   1번 시도 후에도 남는 격차가 크다면 이 방향도 병행 검토.
4. **당장은 하지 않는 게 나은 것: MLA fix 자체를 건드려서 "고치려는" 시도.** §2에서 확인했듯 attention
   커널 비용은 이미 native와 동일 — 여기서 더 손댈 게 없다. 시간을 라우터 쪽에 쓰는 게 맞다.

## 6. 결론

- §9에서 보고한 -11~15% 단일 요청 슬로우다운의 **대부분(최소 58%, 아마 그 이상)은 내 MLA pad/slice fix와
  무관하다** — MoE 라우터의 torch.compile fusion 격차 때문이며, MLA가 아닌 MoE 모델 전반에 해당할 가능성이
  높은, 별도의 Transformers-backend 성능 이슈다.
- MLA fix가 실제로 담당하는 몫은 attention 커널 비용이 native와 사실상 동일하다는 점에서 **거의 0에
  가깝다** — 즉 MLA fix 자체는 "정확하지만 좀 느림"이 아니라 **"정확하고, 속도상 거의 공짜"**에 더 가깝다.
  (§9의 KV-cache 메모리 트레이드오프는 여전히 유효 — 그건 attention/캐시 레이아웃 문제라 이번 발견과는
  다른 축이다.)
- MoE 라우터 fusion 격차는 새로운, 별도의 컨트리뷰션 후보로 `docs/CONTRIB_BACKLOG.md`에 등록함
  (`Fix MoE router softmax/topk torch.compile fusion gap in the Transformers backend`).
- 아직 안 한 것: 다른 MoE 모델(Mixtral 등, MLA 아닌 것)로 같은 커널 격차가 재현되는지 확인, inductor가
  생성한 실제 Triton 소스 비교, 커스텀 op 래핑 시도 + 재측정.
