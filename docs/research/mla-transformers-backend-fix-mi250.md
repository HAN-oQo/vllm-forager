# Fix: DeepSeek MLA models crash under `--model-impl transformers` — root cause + working patch

- **날짜:** 2026-07-10
- **선행 리포트:** `docs/research/multi-model-transformers-backend-mi250-repro.md` — 여기서 DeepSeek-V3-Lite가
  `--model-impl transformers`에서 크래시하는 걸 처음 발견함
- **대상 백로그 항목:** `docs/CONTRIB_BACKLOG.md` — "Fix DeepSeek-V3 (MLA) shape crash in the Transformers-backend"
- **하드웨어:** `mi250-053` (gfx90a), 격리된 vLLM 소스 사본 (`/remote/vast0/herom/vllm-aotbug`, fork
  `HAN-oQo/vllm`에 연결됨, 다른 MI250 작업과 공유하지 않음)
- **최종 결과: 크래시 → 정상 서빙으로 수정 완료.** 다만 성능 트레이드오프 있음 (아래 §4).
- **fork 브랜치:** `HAN-oQo/vllm@wip/mla-transformers-backend-head-size-v` (아직 PR 아님, WIP)
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

| | `--model-impl vllm` (네이티브, 압축 MLA) | `--model-impl transformers` (패치 후, naive) | 차이 |
|---|---|---|---|
| 결정론적 출력 | 빈 문자열 (checkpoint 자체 특성) | **빈 문자열 — 동일** | 일치 |
| 단일 요청 256tok | 22.15 tok/s | 19.55 tok/s | **-11.8%** |
| 동시 16×200tok | 317.6 tok/s | 291.9 tok/s | **-8.1%** |
| 장문 prefill (1561 prompt tok) | 0.937s | 6.683s | **-613% (7.1배 느림)** |

**장문 prefill이 유독 크게 느려지는 건 예상된 결과다** — MLA의 압축 latent/absorption 트릭은 정확히 긴
시퀀스에서 attention FLOPs와 KV 캐시 메모리를 줄이려고 만든 것; naive(압축 해제) 경로는 이 이득을 전혀 못
본다. 단일 요청/동시성에서는 -8~12% 정도로 상대적으로 덜 벌어짐.

## 5. 결론

- **이건 성능 개선이 아니라 정확성 수정이다 — "크래시 → 정상 동작"**. DeepSeek 계열 모델은
  `--model-impl transformers`(ROCm 포함)에서 여전히 네이티브 대비 유의미하게 느릴 것으로 예상됨, 특히
  긴 컨텍스트에서.
- 패치는 작고 (두 파일, ~30줄), ROCm 전용이 아니며, vLLM이 이미 검증된 자기 자신의 native fallback 패턴을
  그대로 재사용 — 새 아키텍처나 core API 변경 없음.
- **아직 안 한 것:** 다른 MLA 모델(DeepSeek-V2, Kimi-K2 등)에서도 재현되는지 확인, 유닛 테스트 작성,
  vLLM 메인테이너에게 PR 제출 전 확인 (특히 "naive fallback 성능이 이 정도로 느린 게 받아들여질 만한가"
  — 크래시보다는 훨씬 낫지만).
