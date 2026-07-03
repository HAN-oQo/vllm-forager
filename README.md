# vllm-forager

> autonomous vLLM/ROCm contributor — 기여 포인트를 끊임없이 탐색(forage)해 고쳐 되돌려주길 반복한다.

vLLM을 **ROCm(MI250) 위에서 지속적으로 발전시키는 자율 기여 에이전트**.
추론 서빙 생태계(vLLM · SGLang · NVIDIA Dynamo · llm-d)의 이슈·PR을 **지속적으로** 추적해
추론 시장의 방향성과 핵심 기술을 정리하고, **자기 예측을 채점해 추적 기준을 스스로 진화시키며**,
그 신호로 **vLLM(ROCm) 기여 후보를 발굴 → 패치 생성·테스트 → 사람 검토 후 PR**까지 잇는
상시 가동·자기개선 에이전트.

> 상태: **M0 (bootstrapping)** — 수집기 뼈대 단계.

## 왜 ROCm

가용 하드웨어가 MI250 × 3 (AMD Instinct, ROCm). vLLM의 ROCm 경로는 CUDA보다 성숙도가 낮아
미해결 갭·버그가 더 많고, 재현할 AMD 장비가 없어 방치된 이슈가 많다 → **실물 하드웨어가 곧 진입장벽이자 우위**.

- 에이전트 런타임: **CPU면 충분**(GPU 불필요).
- MI250은 **vLLM 빌드·테스트·ROCm 버그 재현/검증에만** 사용.

## 빠른 시작

```bash
# 1) 가상환경
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) GitHub 토큰 설정 (rate limit 회피 + private 접근)
cp .env.example .env
# .env 를 열어 GITHUB_TOKEN 채우기 (repo, read 권한 PAT)

# 3) 최소 수집기 1회 실행 → data/*.jsonl 생성
python -m src.collector

# 4) 증분 수집 (state 기준으로 updated 이후만)
python -m src.collector
```

## 리포 구조

```
vllm-forager/
├── README.md
├── requirements.txt
├── .env.example
├── docs/
│   ├── PLAN.md        # 프로젝트 계획 (마일스톤 M0~M4)
│   └── CONTEXT.md     # 설계 결정 로그 — 작업 재개용 컨텍스트
├── src/
│   ├── __init__.py
│   ├── config.py      # 추적 대상 레포·경로 설정
│   └── collector.py   # GitHub 이슈/PR 증분 수집기 (M0)
└── data/              # 수집 결과 (gitignore)
```

## 다음 단계

`docs/PLAN.md`의 마일스톤 순서대로. 우선 M0 = 수집 + 베이스라인 요약.
설계 배경과 지금까지의 의사결정은 `docs/CONTEXT.md` 참고.
