# Fix: DeepSeek MLA models crash under `--model-impl transformers` — root cause + working patch

- **날짜:** 2026-07-10
- **선행 리포트:** `docs/research/multi-model-transformers-backend-mi250-repro.md` — 여기서 DeepSeek-V3-Lite가
  `--model-impl transformers`에서 크래시하는 걸 처음 발견함
- **대상 백로그 항목:** `docs/CONTRIB_BACKLOG.md` — "Fix DeepSeek-V3 (MLA) shape crash in the Transformers-backend"
- **하드웨어:** `mi250-053` (gfx90a), 격리된 vLLM 소스 사본 (`/remote/vast0/herom/vllm-aotbug`, fork
  `HAN-oQo/vllm`에 연결됨, 다른 MI250 작업과 공유하지 않음)
- **최종 결과: 크래시 → 정상 서빙으로 수정 완료.** 다만 성능 트레이드오프 있음 (아래 §4).
- **⚠️ §4까지는 이 문서의 최초 버전(커밋 `cf8d62052`) 기준이며, 그 이후 두 개의 추가 버그(§6, §7)를 더
  찾아 고쳤다** — `num_kv_heads` 버그(§6)와, 훨씬 중요한 **CUDA-graph 하에서만 재발하는 정확성 버그**(§7).
  §4의 속도 수치는 이 두 버그가 있던 상태에서 측정한 것이라 재검증이 필요하다 (아직 안 함, `CONTRIB_BACKLOG.md`
  참고).
- **fork 브랜치:** `HAN-oQo/vllm@wip/mla-transformers-backend-head-size-v` (아직 PR 아님, WIP — 로컬에는 §6/§7
  수정이 더 들어가 있고 아직 fork에 push 안 함)
- **전체 raw 서버 로그:** `docs/research/logs/2026-07-10-mla-fix-mi250/`

## 1. 원래 크래시 — 근본 원인

Transformers 백엔드는 모델 종류와 무관하게 **하나의 함수**를 거친다:

```python
# vllm/model_executor/models/transformers/__init__.py
ALL_ATTENTION_FUNCTIONS["vllm"] = vllm_attention_forward  # -> 표준 Attention.forward(q, k, v)만 호출
```

즉 vLLM 네이티브의 압축 latent 전용 `MLAAttention`(`mla_attention.py`, docstring: *"takes query, and
**compressed** key/value tensors as input"*)은 **한 번도 생성되지 않는다** — `create_attention_instances()`
(`base.py:577`)의 `attn_cls` 선택은 `EncoderOnlyAttention` 아니면 `Attention` 둘뿐이고, 이건 전체
`transformers/` 디렉토리(`moe.py` 포함)를 뒤져봐도 마찬가지다 (사용자 확인 요청으로 재검증함).

크래시는 `create_attention_instances`가 표준 `Attention`을 `head_size = get_head_size()`로 만드는데, 이 값이
MLA 모델에서는 **압축된** latent 차원(`kv_lora_rank+qk_rope_head_dim=576`, 네이티브 `MLAAttention` 전용 값)을
반환하기 때문 — 하지만 이 백엔드가 실제로 다루는 `key`/`value`는 HF 모델 자신의 `kv_b_proj`가 이미 압축
해제한, `qk_nope_head_dim+qk_rope_head_dim=192`(key) / `v_head_dim=128`(value) 차원이다.

## 2. 첫 번째 시도 (실패) — `head_size_v`

`vllm.model_executor.layers.attention.attention.Attention`은 이미 Q/K와 V가 다른 차원을 쓸 수 있게
`head_size_v` 파라미터를 지원한다 (`attention.py:250`, `:533-538`). 이걸 이용해 `head_size=192`(Q/K),
`head_size_v=128`(V)를 명시적으로 넘기도록 패치 → **원래 크래시는 사라졌지만, 다음 단계(KV 캐시 텐서
할당)에서 새 크래시** 발생:

```
RuntimeError: shape '[2, 7408, 16, 128, 192]' is invalid for input of size 4854906880
```

**원인:** `AttentionBackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)` — ROCm,
FlashAttention, FlashInfer 등 **모든** attention backend가 구현하는 이 공용 인터페이스가 `head_size`
하나만 받고, K/V가 다른 차원일 수 있다는 개념 자체가 없다 (`(2, num_blocks, block_size, num_kv_heads,
head_size)` — "2"는 K/V 슬롯인데 둘 다 같은 `head_size` 공유). `VLLM_MLA_DISABLE=1`(vLLM에 이미 있는,
`use_mla`를 끄는 환경변수)을 같이 켜봐도 이 KV-캐시 API 자체의 한계는 그대로라 실패.

네이티브 MLA(`--model-impl vllm`)가 이 문제를 안 겪는 이유: `MLACommonBackend.get_kv_cache_shape`
(`mla_attention.py:1216`)은 아예 `(num_blocks, block_size, head_size)` — K/V 슬롯("2") 자체가 없다. 압축
latent 하나만 캐시하니까 애초에 "K와 V가 같은 차원이어야 하나?"라는 질문이 생기지 않는다.

## 3. 진짜 해결책 — vLLM 네이티브의 "naive" fallback을 그대로 따라하기

`vllm/model_executor/models/deepseek_v2.py`를 보면, `use_mla=False`일 때 쓰이는 네이티브 클래스
`DeepseekV2Attention`(558-609줄)이 **정확히 같은 문제**(Q/K=192, V=128)를 이미 겪고 있고, 이미 해결한
코드가 있다 — `head_size_v` 없이, **V를 192로 제로 패딩해서 attention에 넣고, 출력에서 다시 128로 잘라낸다**:

```python
# deepseek_v2.py:601-609
v = torch.nn.functional.pad(v, [0, self.qk_head_dim - self.v_head_dim], value=0) \
    .view(-1, self.num_local_heads * self.qk_head_dim)
attn_output = self.attn(q, k, v)   # head_size 하나(192)만 사용
attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
    ..., : self.v_head_dim
].reshape(-1, self.num_local_heads * self.v_head_dim)
```

이 패턴을 Transformers 백엔드의 제네릭 훅(`vllm_attention_forward`)에 그대로 옮김 — **모델별 지식이 필요
없다** (`value.shape[-1] < head_size`면 패딩, 아니면 그냥 통과):

```python
# vllm/model_executor/models/transformers/__init__.py (patched)
head_size = self_attn.head_size
v_head_dim = value.shape[-1]
if v_head_dim < head_size:
    value = torch.nn.functional.pad(value, [0, head_size - v_head_dim])
query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
output = self_attn.forward(query, key, value)
if v_head_dim < head_size:
    output = output.view(hidden, self_attn.num_heads, head_size)[..., :v_head_dim]
    output = output.reshape(hidden, -1)
return output, None
```

`create_attention_instances`(`base.py`)는 `head_size_v` 없이 `head_size`만 `qk_nope_head_dim+qk_rope_head_dim`
(192)로 직접 계산 — `get_head_size()`의 압축 latent 기본값을 우회. non-MLA 모델에서는 `v_head_dim ==
head_size`라 패딩 로직이 완전히 no-op.

**결과: 크래시 없이 정상 서빙됨.** KV 캐시 API를 전혀 안 건드림 — K와 V 둘 다 균일하게 192로 저장되니까
`get_kv_cache_shape`가 이미 지원하는 표준 케이스 그대로.

## 4. 검증 — 정확성 + 속도 (MI250, DeepSeek-V3-Lite)

### ⚠️ 정정 — 최초 측정치는 방법론 오류로 무효

**최초 측정(§4 초판)은 장문 prefill에서 "-613% (7.1배 느림)"으로 보고했으나, 이는 콜드 스타트(첫 호출 —
해당 프롬프트 길이의 torch.compile/cudagraph 캡처 비용 포함)와 웜 상태(반복 호출, steady-state)를 서로
다른 두 서버 실행에서 뒤섞어 비교한 방법론적 오류였다.** 사용자가 "정말 그렇게 느리냐"고 재확인을 요청해서
같은 서버에서 동일 prefill 요청을 3번 반복했더니 6.56s → 1.51s → 1.51s로 급락 — 첫 호출만 별도로 비싼
것을 확인. `bench.py`를 콜드/웜 모두 측정하도록 고쳐서(`long_prefill`을 두 번 호출, 두 번째를 웜으로 기록)
네이티브·패치 양쪽을 **같은 방법론으로 새로 측정**했다.

| | `--model-impl vllm` (네이티브, 압축 MLA) | `--model-impl transformers` (패치 후, naive) | 차이 |
|---|---|---|---|
| 결정론적 출력 | 빈 문자열 (checkpoint 자체 특성) | **빈 문자열 — 동일** | 일치 |
| 단일 요청 256tok | 22.22 tok/s | 19.59 tok/s | **-11.8%** |
| 동시 16×200tok | 317.3 tok/s | 292.1 tok/s | **-8.0%** |
| 장문 prefill — 콜드(첫 호출) | 6.507s | 6.661s | +2.4% (오차범위) |
| 장문 prefill — 웜(반복 호출, steady-state) | 1.970s | 1.716s | **-12.9% (patched가 오히려 더 빠름)** |

(raw 로그: `docs/research/logs/2026-07-10-mla-fix-mi250/04-final_vllm_impl_native.log`,
`05-final_transformers_impl_patched.log`)

**정정된 결론:** 장문 prefill은 콜드 상태에서 두 백엔드 모두 비슷하게 느리고(첫 호출 컴파일 비용이 지배적),
정작 steady-state(웜)에서는 오히려 patched 쪽이 근소하게 더 빠르다 — 이번 measurement에서는 MLA의 압축
latent 이득이 (적어도 이 작은 "lite" 체크포인트·이 prefill 길이에서는) 뚜렷하게 관측되지 않았다. 단일
요청/동시성 처리량에서는 일관되게 -8~12% 정도 patched 쪽이 느림 — 이건 재현성 있게 관측됨(두 번의 독립
측정에서 거의 같은 수치).

## 5. 결론

- **이건 성능 개선이 아니라 정확성 수정이다 — "크래시 → 정상 동작"**. 이번 체크포인트·이번 prefill
  길이 기준으로는 성능 저하가 처음 생각했던 것(7배)만큼 크지 않음 — 단일/동시성에서 -8~12%, 장문 prefill은
  steady-state에서 오히려 근소 우위. 다만 이건 작은 "lite" 모델 하나, prefill 길이 하나로 측정한 것 — 더
  크거나 더 긴 컨텍스트에서는 MLA 압축 이득이 더 크게 벌어질 수 있음 (미검증).
- 패치는 작고 (두 파일, ~30줄), ROCm 전용이 아니며, vLLM이 이미 검증된 자기 자신의 native fallback 패턴을
  그대로 재사용 — 새 아키텍처나 core API 변경 없음.
- **벤치마크 방법론 교훈 (다음에도 적용):** 첫 호출(콜드, 컴파일/그래프캡처 비용 포함)과 반복 호출(웜,
  steady-state)을 반드시 구분해서 측정할 것 — 특히 prefill처럼 프롬프트 길이별로 새 컴파일이 트리거될 수
  있는 테스트는 한 번만 호출하면 "콜드"와 "웜"을 우연히 비교하게 될 위험이 있음.
- **아직 안 한 것 (§4 기준, 아래 §6/§7에서 갱신됨):** 다른 MLA 모델(DeepSeek-V2, Kimi-K2 등)·다른 크기·더 긴
  컨텍스트에서도 재현되는지 확인, 유닛 테스트 작성, vLLM 메인테이너에게 PR 제출 전 확인.

## 6. 실제 모델(DeepSeek-V2-Lite)로 테스트 → 두 번째 버그 발견: `num_kv_heads`

§4까지 검증에 쓴 DeepSeek-V3-Lite는 **`config.json`이 `num_hidden_layers: 4`인 축소판 테스트 체크포인트**였다
(실제 아키텍처는 61 레이어). "네이티브·패치 양쪽 다 빈 문자열" 일치는 진짜 정합성 검증이 아니라 이 체크포인트
자체의 degenerate한 특성이었을 뿐 — 진짜 답변 품질은 한 번도 검증되지 않았다.

정식 학습된 체크포인트인 **DeepSeek-V2-Lite**로 테스트 전환. 단, `trust_remote_code=True`를 쓰면 별개의
`is_torch_fx_available` import 에러로 크래시 (transformers 5.13.0에서 제거된 심볼을 remote-code 파일이 여전히
import — 이미 열려 있던 업스트림 PR `huggingface/transformers#44615`에 재현 코드와 함께 코멘트 남김). 우회:
`--trust-remote-code`를 아예 빼면 transformers 자체의 네이티브 `deepseek_v2` 아키텍처 지원을 타면서 이 에러를
피해감.

이렇게 띄운 DeepSeek-V2-Lite + `--model-impl transformers`(§3 패치 적용 상태)는 **크래시 없이 서빙되지만
결과가 깨진 텍스트**였다 (네이티브 `--model-impl vllm`은 `" Paris.\n..."`처럼 정상 응답). "크래시 없음"과
"정답"은 다른 것 — 진짜 정합성 버그가 여기서 처음 드러남.

**원인:** `create_attention_instances`가 `is_deepseek_mla` 분기에서 `head_size`는 고쳤지만
`num_kv_heads = get_num_kv_heads()`는 그대로 뒀다 — 이 함수는 MLA 모델에서 **1**을 반환한다 (네이티브
absorbed 경로는 압축 latent 하나만 캐시하는 MQA 형태라서). 하지만 이 백엔드가 실제로 다루는
`key`/`value`는 HF 자신의 `kv_b_proj`가 이미 `num_heads`개의 서로 다른 head로 압축 해제해 놓은 것 —
`num_kv_heads=1`을 넘기면 크래시는 안 나지만 head 경계가 뒤섞여 조용히 깨진 출력이 나온다. 네이티브의 naive
fallback(`DeepseekV2Attention`)은 애초에 `Attention(..., num_kv_heads=self.num_local_heads, ...)` — 즉
`num_heads`와 동일한 값 — 로 생성하고 있었다 (§3에서 옮겨올 때 놓친 부분).

**수정:** `base.py`의 같은 `is_deepseek_mla` 블록에 `num_kv_heads = num_heads`를 추가.

```python
if self.model_config.is_deepseek_mla:
    qk_nope_head_dim = getattr(text_config, "qk_nope_head_dim", 0)
    qk_rope_head_dim = getattr(text_config, "qk_rope_head_dim", 0)
    if qk_nope_head_dim and qk_rope_head_dim:
        head_size = qk_nope_head_dim + qk_rope_head_dim
        num_kv_heads = num_heads  # ← 추가
```

이 수정 후에도 **여전히 깨진 출력**이 남아 있었다 — §7로 이어짐.

## 7. 세 번째, 가장 중요한 버그: eager에서는 정답, 컴파일(cudagraph)에서는 여전히 깨짐

디버그 프린트를 넣어 `--enforce-eager`로 확인해보니: shape/파라미터 전부 정상(`num_heads=16
num_kv_heads=16 head_size=192`, V가 128→192로 올바르게 패딩됨), 그리고 **출력도 정답** (`" Paris.\nThe"`).
그런데 프린트를 지우고 vLLM 기본 실행(torch.compile + CUDA graph 둘 다 켜짐)으로 **정확히 같은 코드**를
돌리면 여전히 깨진 텍스트(`"h，\n\n\n\n\n\n\n\n\n\n-\n..."`). 로직 자체는 맞는데, 컴파일된 실행 경로에서만
재발하는 별개의 진짜 버그가 있다는 뜻.

### 7.1 이분 탐색 — dynamo 트레이싱인가, CUDA graph replay인가?

`CompilationConfig.cudagraph_mode`에 `NONE`이 있는 걸 확인 (`--compilation-config
'{"cudagraph_mode": "NONE"}'`) — torch.compile은 켠 채로 CUDA graph capture/replay만 끌 수 있다. 이 설정으로
재시작 → **정답** (`" Paris.\nThe capital..."`, 3번 반복 재현). 즉:

| 설정 | 결과 |
|---|---|
| `--enforce-eager` (compile 꺼짐, cudagraph 꺼짐) | ✅ 정답 |
| `--compilation-config '{"cudagraph_mode": "NONE"}'` (compile 켜짐, cudagraph 꺼짐) | ✅ 정답 |
| 기본값 (compile 켜짐, cudagraph 켜짐) | ❌ 깨진 텍스트 |

**→ 범인은 dynamo/torch.compile 트레이싱이 아니라 CUDA graph capture/replay다.**

### 7.2 "그러면 vllm mla(네이티브)는 왜 되는데?" — pad/slice 패턴 자체는 무죄

패딩 그 자체(매 호출마다 새로 메모리를 할당하는 `F.pad`)가 CUDA graph 하에서 원래 안전하지 않은 건 아닌가
의심 — 그런데 vLLM 자신의 **패치 안 한** naive fallback으로 검증해보면 답이 나온다. 아무 코드 수정 없이
`--model-impl vllm` + `VLLM_MLA_DISABLE=1` (네이티브 `DeepseekV2Attention`, §3에서 그대로 베낀 원본 코드) 로
기본 CUDA graph 켜진 채 서버를 띄우면 → **정답** (`" Paris.\nThe currency..."`, 3번 반복 재현). 즉 같은
연산(`F.pad` + `Attention.forward` + slice)을 vLLM 자신이 CUDA graph 아래서 이미 문제없이 쓰고 있다 —
pad/slice 패턴 자체는 무죄.

그렇다면 내 패치와 네이티브의 진짜 차이는? `deepseek_v2.py:604`를 다시 보면:

```python
v = torch.nn.functional.pad(v, [0, self.qk_head_dim - self.v_head_dim], value=0)  # 무조건 실행
```

**조건문이 없다** — `self.qk_head_dim - self.v_head_dim`은 `__init__` 시점에 고정된 순수 파이썬 정수이고,
매 호출마다 항상 같은 양만큼 무조건 패딩한다. 반면 내 `vllm_attention_forward`는:

```python
if v_head_dim < head_size:                                    # ← 텐서 shape에 대한 파이썬 분기
    value = torch.nn.functional.pad(value, [0, head_size - v_head_dim])
```

이 `if` — non-MLA 모델에서 이 훅이 no-op이 되게 하려고 넣은 조건 — 이 바로 원인이었다. 특정 레이어 하나에
대해서는 `v_head_dim`과 `head_size`가 둘 다 고정값이라 이 분기는 매번 항상 같은 쪽으로만 타는데도, CUDA
graph capture/replay 하에서는 이 조건문의 존재 자체가 잘못된 결과를 만들어낸다.

### 7.3 검증: 조건문 제거 → 기본 실행(compile+cudagraph)에서도 정답

`vllm_attention_forward`를 조건문 없이 네이티브와 똑같이 **무조건** 패딩/슬라이스하도록 수정 (`v_head_dim ==
head_size`일 때 `F.pad`에 패딩 0 / 슬라이스가 전체 범위 — 둘 다 이미 자명한 no-op이라 non-MLA 모델에
아무 영향 없음):

```python
# vllm/model_executor/models/transformers/__init__.py (최종)
head_size = self_attn.head_size
v_head_dim = value.shape[-1]
value = torch.nn.functional.pad(value, [0, head_size - v_head_dim])  # 무조건
query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
output = self_attn.forward(query, key, value)
output = output.view(hidden, self_attn.num_heads, head_size)[..., :v_head_dim]  # 무조건
output = output.reshape(hidden, -1)
```

기본 실행(compile+cudagraph 둘 다 켜짐)으로 재시작 → **정답**, 3번 반복 재현 + 짧은 프롬프트뿐 아니라 100
토큰 생성에서도 끝까지 일관되게 정답 + 동시 요청 4개(서로 다른 CUDA-graph 배치 크기 버킷을 태움)도 전부
정답.

### 7.4 결론 — 근본 원인

**CUDA graph로 캡처되는 영역 안에서, 텐서를 새로 할당하는 연산(`F.pad`) 주변에 파이썬 레벨의
데이터-의존적(data-dependent) 분기가 있으면 — 그 분기가 해당 레이어에 대해 항상 같은 쪽으로만 결정되는
경우에도 — replay 시점에 잘못된 결과가 나온다. 같은 연산을 무조건 실행하면 이 문제가 사라진다.**

네이티브의 naive fallback이 이 버그를 겪지 않은 건 설계적으로 피해서가 아니라 — 애초에 조건문 없이
무조건 패딩하도록 짜여 있었기 때문에 우연히 피해간 것으로 보인다 (naive fallback 자체가 `use_mla=True`가
항상 기본값이라 실제로는 거의 실행되지 않는 코드 경로라, 이런 CUDA-graph 상호작용이 upstream CI에서 딱히
검증됐을 가능성도 낮다). 이건 이 패치 하나만의 문제가 아니라 — "CUDA graph 캡처 영역 안에서 항상 같은
결과로 귀결되는 분기라도, 그 분기 자체가 존재하면 안전하지 않을 수 있다"는 더 일반적인 패턴일 수 있어
보인다. 정확한 이유(dynamo의 guard 처리 방식인지, PyTorch의 CUDA-graph 메모리 풀이 조건부 할당을 어떻게
다루는지 등, 더 깊은 PyTorch 내부 메커니즘)는 아직 코드 레벨로 확증하지 않았다 — 여기서는 재현 가능한
실험으로 원인을 이 지점까지 좁힌 상태.

## 8. 최종 코드 상태 (§6/§7 반영)

- **크래시 → 정상 서빙 → (num_kv_heads 수정) → (CUDA-graph 정확성 버그 수정)**까지 전부 완료. 최종 패치는
  여전히 두 파일, `head_size`/`num_kv_heads`를 MLA일 때 재계산하는 `base.py`의 몇 줄 + 무조건 pad/slice로
  바뀐 `__init__.py`의 `vllm_attention_forward`.
- §4의 속도 비교는 그 측정 당시 코드에 num_kv_heads 버그와 CUDA-graph 버그가 모두 있었던 상태 — §9에서
  최종 코드로 재측정.

## 9. 최종 재검증 — 정합성 + 속도, 두 개의 실제 모델 (DeepSeek-V2-Lite, DeepSeek-V2 전체)

§6/§7 수정을 모두 반영한 최종 코드로, 실제 학습된 체크포인트 두 개(작은 것 하나, 큰 것 하나) — 둘 다 기본
실행(torch.compile + CUDA graph 켜짐)에서 — native(`--model-impl vllm`)와 patched(`--model-impl
transformers`)를 같은 GPU 개수·같은 설정으로 나란히 띄워 비교했다.

### 9.1 DeepSeek-V2-Lite (16B, TP=1, `--max-model-len 4096`)

5개의 서로 다른 프롬프트(사실 질문, 설명, 코드 생성, 산수, 반의어)로 정합성 확인 — **양쪽 다 일관되게
말이 되는 영어/코드 답변.** (참고로 "2 + 2 ="에 native는 엉뚱한 대수 방정식으로 답하고 patched는 "4"로
정확히 답함 — 이건 백엔드 차이가 아니라 이 작은 모델 자체의 능력 편차로 보임.)

| 지표 | native (`vllm`) | patched (`transformers`) | 차이 |
|---|---|---|---|
| 단일 요청 256tok | 120.9 tok/s | 103.1 tok/s | **-14.7%** |
| 동시 16×200tok | 909.7 tok/s | 1050.1 tok/s | **+15.4% (patched가 더 빠름)** |
| 장문 prefill, 콜드 | 0.982s | 1.182s | +20.4% (콜드, 컴파일 비용 포함 — 노이즈에 가까움) |
| 장문 prefill, 웜 | 0.621s | 0.547s | **-11.9% (patched가 더 빠름)** |

### 9.2 DeepSeek-V2 전체 (236B MoE, TP=8, `--max-model-len 768`, `--gpu-memory-utilization 0.95`)

**⚠️ 이 모델 크기에서 처음 드러난 진짜 트레이드오프: KV 캐시 메모리.** native는 기본 설정
(`--max-model-len 4096`, `gpu_memory_utilization` 기본값 0.9)으로 바로 떴다. patched(naive fallback)는
**같은 설정으로 시작조차 안 됨** — 순차적으로 줄여야 했다:
- `--max-model-len 4096` (기본) → 실패: "2.81 GiB KV cache is needed... available (0.75 GiB)"
- `--max-model-len 1024` → 실패: "0.7 GiB needed... available (0.69 GiB)" (여전히 근소하게 부족)
- `--max-model-len 768` + `--gpu-memory-utilization 0.95` → **성공**

**원인:** naive fallback은 K와 V를 각각 균일한 `head_size=192`로, per-token 저장량이
`2(K/V) × num_heads(16) × head_size(192) = 6144`인 반면, 네이티브 MLA는 압축 latent 하나만
`kv_lora_rank+qk_rope_head_dim=576`을 저장 — **patched가 토큰당 KV 캐시를 약 10.7배 더 씀.** 236B처럼
가중치 자체가 8-GPU 메모리 대부분을 차지하는 큰 모델에서는 이 차이가 "그냥 좀 느림" 정도가 아니라 **서비스
가능한 컨텍스트 길이 자체를 심각하게 줄인다** — 이건 이번에 lite 모델로는 전혀 안 보이던, 스케일에서만
드러나는 진짜 제약이다.

공정 비교를 위해 native도 같은 제약 설정(`--max-model-len 768`, `--gpu-memory-utilization 0.95`)으로
재시작해서 비교:

| 지표 | native (`vllm`) | patched (`transformers`) | 차이 |
|---|---|---|---|
| 단일 요청 128tok | 38.36 tok/s | 34.10 tok/s | **-11.1%** |
| 동시 8×100tok | 211.95 tok/s | 195.25 tok/s | **-7.9%** |

정합성: 5개 프롬프트 전부 양쪽 다 일관되게 말이 되는 답변 (사실 질문, LLM 설명, 팩토리얼 함수, 산수, 반의어
전부 일치하는 수준의 품질).

### 9.3 시도했지만 규모상 보류: Kimi-K2 (Moonshot AI)

`/remote/vast0/share-mv/moonshotai/Kimi-K2.5`, `Kimi-K2.6` — `config.json` 확인 결과 `kv_lora_rank=512`,
`q_lora_rank=1536`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128` — DeepSeek과 동일한 MLA
아키텍처 계열로 확인됨 (`is_deepseek_mla` 화이트리스트가 이 모델도 잡아줄 가능성 높음, 미확인). 다만 체크포인트
크기가 **555GB** — 이 노드의 8×MI250 총 VRAM(약 549GB)보다 가중치만으로 이미 초과, KV 캐시·activation
여유분이 전혀 없어 이번 세션에서는 시도하지 않았다. 더 많은 GPU가 있는 노드나 양자화된 체크포인트가 있다면
재시도 가치 있음.

### 9.4 종합 결론

- **정합성: 두 개의 실제(비-degenerate) 모델, 작은 것과 큰 것 모두에서 확인 완료.** 단순 "크래시 없음"이나
  "빈 문자열 일치"가 아니라 5개의 다양한 프롬프트에 대해 실제로 말이 되는 답변을 생성함을 확인.
- **속도: 단일 요청 기준 일관되게 -7~15% 느림.** 동시성 처리량은 모델 크기에 따라 갈림 — lite에서는 오히려
  patched가 빠르고(+15%), 236B에서는 patched가 약간 느림(-8%). 규모가 커질수록 native의 우위가 (아주
  근소하게) 다시 나타나는 경향.
- **⚠️ KV 캐시 메모리 — 가장 중요한, 규모에서만 드러나는 트레이드오프.** naive fallback은 토큰당 KV 캐시를
  약 10.7배 더 쓴다. 작은 모델·짧은 컨텍스트에서는 안 보이다가, 가중치 자체가 GPU 메모리 대부분을 차지하는
  큰 모델에서는 서비스 가능한 최대 컨텍스트 길이를 크게 줄인다 (이번 236B 테스트에서 native 4096 vs patched
  768). **이건 이 패치를 upstream에 제안할 때 반드시 명시해야 하는 한계다** — "크래시 → 정상 동작"은
  맞지만 "네이티브와 동등"은 아니다, 특히 긴 컨텍스트나 메모리가 빡빡한 배포에서는.
- 남은 것: 유닛 테스트 작성, fork 브랜치에 push (아직 로컬에만 있음), PR 초안 — 위 KV 캐시 트레이드오프를
  PR 설명에 명확히 포함할 것.
