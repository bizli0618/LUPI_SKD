# LUPI-SKD 구현 현황

> 미구현 기능(Adaptive Top-K, 데이터 혼합 실험, Success 확장)은 [future_work.md](./future_work.md) 참조

---

## 1. 구현 완료 기능

```
┌─────────────────────────────────────────────────────────────┐
│                    ✅ 구현 완료                               │
├─────────────────────────────────────────────────────────────┤
│  • LUPI 입력 비대칭 (Teacher: Q+K, Student: Q)               │
│  • SKD Rollout (기본/최적화/Adaptive Gamma/Compiled/Debug)   │
│  • 병렬 Teacher 검증 + KV-cache crop                         │
│  • teacher_min_prob threshold                               │
│  • 한국 문화 데이터셋 200개 (한/영)                            │
│  • DDP 학습 + 노트북 학습 파이프라인                           │
│  • WandB 로깅 (gamma_history, acceptance_rate 등)            │
│  • 체크포인트 resume 기능                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Rollout 함수 버전별 비교

| 파일 | 설명 | 핵심 특징 | 상태 |
|------|------|----------|------|
| `lupi_rollout.py` | 기본 버전 | 순차적 Teacher 검증 O(γ) forward | ✅ |
| `lupi_rollout_optimized.py` | 최적화 버전 | 병렬 Teacher 검증 O(1), KV-cache crop | ✅ |
| `lupi_rollout_adaptive.py` | **Adaptive Gamma** | 동적 블록 크기 (성공↑γ, 실패↓γ) | ✅ |
| `lupi_rollout_compiled.py` | torch.compile | 실험적 컴파일 지원 | ✅  |
| `lupi_rollout_debug.py` | 디버그 버전 | 상세 로깅 | ✅ |

**Adaptive Gamma 로직** (`lupi_rollout_adaptive.py:345-353`):
```python
if n_matches == len(draft_ids):
    current_gamma = min(max_gamma, current_gamma + 2)  # 전체 성공 → γ 증가
else:
    current_gamma = max(1, n_matches + 1)              # 실패 → γ 감소
```

---

## 3. LUPI 입력 비대칭 구조

| 구분 | Teacher | Student |
|------|---------|---------|
| **System Prompt** | 한국 문화 전문가 페르소나 | ❌ |
| **Few-shot** | 3개 (음식/역사/전통) | ❌ |
| **Knowledge** | `[Background Knowledge]\n{K}` | ❌ |
| **Query** | `{Q}` | `{Q}` 만 |
| **토큰 수 예시** | ~2,460 | ~47 |

**구현 위치**: `train/ddp_lupi_dataset.py:36-172`

---

## 4. 데이터셋

| 파일 | 언어 | 샘플 수 | 카테고리 (각 40개) |
|------|------|---------|-------------------|
| `korean_culture_train_200.json` | 한국어 | 200 | 전통예절, 역사/인물, 음식/발효, 지리/지역, 현대문화 |
| `korean_culture_train_200_en.json` | 영어 | 200 | 동일 |

**데이터 구조**:
```json
{
  "id": "tradition_001",
  "category": "tradition",
  "query": "...",
  "knowledge": "...",
  "sources": ["..."],
  "word_count": 312,
  "weakness_addressed": ["..."]
}
```

---

## 5. 학습 파이프라인

| 컴포넌트 | 파일 | 설명 |
|----------|------|------|
| DDP 학습 | `train/ddp_lupi_skd.py` | 분산 학습, GPU 2개씩 (T/S 분리), resume 지원 |
| 노트북 학습 | `notebooks/train_lupi_skd_full.ipynb` | 상세 로깅, WandB 연동 |
| 데이터셋 클래스 | `train/ddp_lupi_dataset.py` | `LUPIDataset`, `LUPICollator` |

**현재 기본 하이퍼파라미터**:
```python
# SKD
top_k = 5
gamma = 20
teacher_temperature = 0.7
student_temperature = 0.7
teacher_min_prob = 0.05

# Training
learning_rate = 1e-5
gradient_accumulation = 1
epochs = 5
max_new_tokens = 4096
```

---

## 6. WandB 로깅 메트릭

| 카테고리 | 메트릭 |
|----------|--------|
| **Training** | `train/loss`, `train/lr`, `train/step` |
| **Rollout** | `acc_rate/mean`, `acc_rate/first_token`, `acc_rate/full_match` |
| **Gamma** | `gamma/mean`, `gamma/max`, `gamma/min`, `gamma/std` |
| **Tokens** | `tokens/total_drafted`, `tokens/total_accepted`, `tokens/acceptance_ratio_global` |
| **Validation** | `val/loss` |

---

## 7. 체크포인트 Resume

**사용법**:
```bash
# 환경변수로 설정
RESUME_CHECKPOINT="checkpoints/lupi_skd/step_44" ./run_train.sh
```