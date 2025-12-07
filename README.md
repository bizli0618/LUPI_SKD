# LUPI-SKD: Learning Using Privileged Information with Speculative Knowledge Distillation

Speculative Knowledge Distillation(SKD)을 활용한 한국 문화 지식 주입 프레임워크입니다. Teacher 모델에만 지식(K)을 제공하고, Student 모델은 Query(Q)만으로 Teacher와 동일한 응답을 생성하도록 학습합니다.

## Key Features

- **LUPI 비대칭 입력**: Teacher (Q+K) vs Student (Q only)
- **5가지 Rollout 구현**:
  - `lupi_rollout.py` - 기본 순차 검증
  - `lupi_rollout_optimized.py` - 병렬 Teacher 검증 O(1)
  - `lupi_rollout_adaptive.py` - Adaptive Gamma (동적 블록 크기)
  - `lupi_rollout_compiled.py` - torch.compile 지원
  - `lupi_rollout_debug.py` - 상세 디버그 로깅
- **병렬 Teacher 검증**: O(1) forward pass + KV-cache crop
- **한국 문화 데이터셋**: 200개 Q-K 쌍 (한/영)
- **DDP 분산 학습**: GPU 2개씩 Teacher/Student 분리
- **WandB 로깅**: acceptance rate, gamma history 등

## Project Structure

```
LUPI-SKD/
├── train/
│   ├── ddp_lupi_skd.py          # DDP 학습 메인
│   ├── ddp_lupi_dataset.py      # LUPI 데이터셋/콜레이터
│   ├── ddp_skd.py               # 기본 SKD (non-LUPI)
│   ├── lupi_rollout.py          # 기본 rollout
│   ├── lupi_rollout_optimized.py # 병렬 검증
│   ├── lupi_rollout_adaptive.py  # Adaptive Gamma
│   ├── lupi_rollout_compiled.py  # torch.compile
│   ├── lupi_rollout_debug.py     # 디버그 버전
│   ├── run_kd_train.py          # KD 학습 런처
│   └── train_sft.py             # SFT 학습
├── eval/                         # 평가 스크립트
├── config/                       # 설정 파일
├── notebooks/                    # 학습 노트북
├── data/                         # 한국 문화 데이터셋
├── scripts/                      # 유틸리티 스크립트
├── transformers/                 # HuggingFace 수정 파일
└── docs/
    └── research/                 # 연구 노트
```

## Requirements

- Python 3.10+
- PyTorch 2.6.0+ with CUDA
- GPU: A100/H100 (BF16 + FlashAttention 지원 필요)

```bash
pip install -r requirements.txt

# FlashAttention (별도 설치)
pip install flash-attn --no-build-isolation
```

> **Note**: `requirements.txt`에 중복 항목이 있을 수 있습니다. 핵심 의존성: `torch`, `transformers`, `accelerate`, `trl`, `deepspeed`, `wandb`

## Quick Start

### 1. 노트북 학습 (단일/듀얼 GPU)

```bash
jupyter notebook notebooks/train_lupi_skd_full.ipynb
```

### 2. DDP 학습 (다중 GPU)

**GPU 요구사항**: 각 rank가 2개 GPU 사용 (Teacher + Student 분리)
- `nproc_per_node = 총 GPU 수 / 2`
- 예: 4 GPU → `nproc_per_node=2`, 8 GPU → `nproc_per_node=4`

```bash
# 4 GPU 예시 (2 ranks × 2 GPUs each)
torchrun --nproc_per_node=2 train/ddp_lupi_skd.py

# 환경변수로 모델 지정 (선택)
export TEACHER_MODEL="Qwen/Qwen3-32B"
export STUDENT_MODEL="Qwen/Qwen3-4B"
```

### 3. 체크포인트 Resume

```bash
RESUME_CHECKPOINT="checkpoints/lupi_skd/step_100" torchrun --nproc_per_node=2 train/ddp_lupi_skd.py
```

## Configuration

주요 하이퍼파라미터 (`train/ddp_lupi_skd.py`):

```python
# SKD 파라미터
top_k = 5                    # Top-K acceptance threshold
gamma = 20                   # Speculative block size (Adaptive Gamma가 동적 조정)
teacher_temperature = 0.7
student_temperature = 0.7
teacher_min_prob = 0.05      # Minimum probability threshold

# 학습 파라미터
learning_rate = 1e-5
epochs = 5
max_new_tokens = 4096
```

## Dataset

`data/korean_culture_train_200.json` / `korean_culture_train_200_en.json`:

```json
{
  "id": "tradition_001",
  "category": "tradition",
  "query": "설날에 세배를 드릴 때 어떤 예절을 지켜야 하나요?",
  "knowledge": "설날 세배는 윗어른께 새해 인사를 드리는 전통 예절입니다..."
}
```

5개 카테고리 (각 40개): 전통예절, 역사/인물, 음식/발효, 지리/지역, 현대문화

## Transformers Patch (Optional)

Speculative Decoding 사용 시 HuggingFace Transformers 수정 필요:

```bash
cp transformers/* /path/to/site-packages/transformers/generation/
```

## Documentation

- [구현 현황](docs/research/info.md) - 현재 구현된 기능
- [Future Work](docs/research/future_work.md) - 미구현 기능 설계 (Adaptive Top-K, 데이터 혼합 등)

## References

### Original SKD Paper

```bibtex
@misc{xu2025speculativeknowledgedistillation,
  title={Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling},
  author={Wenda Xu and Rujun Han and Zifeng Wang and Long T. Le and Dhruv Madeka and Lei Li and William Yang Wang and Rishabh Agarwal and Chen-Yu Lee and Tomas Pfister},
  year={2025},
  eprint={2410.11325},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2410.11325}
}
```

### Original Implementation

- [google-research/speculative_kd](https://github.com/google-research/google-research/tree/master/speculative_kd)

## License

Apache 2.0
