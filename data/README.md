# Data 디렉토리 구조

이 디렉토리는 **한국 문화 Knowledge 데이터 합성 프로젝트**의 산출물을 포함합니다.

### 최종 결과물
- **`korean_culture_train_200.json`** (283KB)
  - LUPI-SKD 학습용 최종 Train 데이터셋
  - 200개 Question-Knowledge 쌍 (5개 카테고리 × 40개)
  - Teacher: Query + Knowledge / Student: Query only

---

## 📊 데이터 통계

### 카테고리 분포
- 전통예절: 40개
- 역사/인물: 40개
- 음식/발효: 40개
- 지리/지역: 40개
- 현대문화: 40개
- **총계: 200개**

### Knowledge 길이
- 최소: 309자
- 최대: 662자
- 평균: 462.9자
- 500자 초과: 57개 (28.5%)

---

## 💡 사용 방법

### LUPI-SKD 학습
```python
import json

# 최종 데이터셋 로드
with open('korean_culture_train_200.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 각 항목 구조
for item in train_data:
    query = item['query']          # 질문
    knowledge = item['knowledge']  # 한국 문화 지식 (최대 500자)
    category = item['category']    # 카테고리

    # Teacher: query + knowledge
    # Student: query only
```
---

## 📝 참고사항

### 데이터 특징
- **언어:** 한국어
- **관점:** 한국 문화 중심 (과학 용어 최소화)
- **출처:** 한국민족문화대백과 > Wikipedia
- **용도:** LUPI-SKD 학습용 Train 데이터
