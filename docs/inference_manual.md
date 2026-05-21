# DiffMS 추론 매뉴얼 — 스펙트럼 → 분자 구조 예측

## 개요

DiffMS는 MS/MS 스펙트럼을 입력받아 분자 그래프(SMILES)를 생성하는 diffusion 모델이다.

```
MS/MS 스펙트럼
    ↓  MIST 인코더
스펙트럼 임베딩 벡터 y
    ↓  DiGress 그래프 diffusion (500 스텝)
후보 분자 20개 (SMILES)
```

- **인코더**: MIST (스펙트럼 → 벡터)
- **디코더**: DiGress-style discrete graph diffusion
- **원자 타입**: C, O, P, N, S, Cl, F, H
- **결합 타입**: 없음 / 단일 / 이중 / 삼중 / 방향족
- **학습 데이터**: MassSpecGym (MSG)
- **프리트레인 체크포인트**: `checkpoints/checkpoints/diffms_msg.ckpt`

---

## 환경 설정

```bash
conda create -y -c conda-forge -n diffms rdkit=2024.09.4 python=3.9
conda activate diffms
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 버전에 맞게
pip install -e .
```

---

## 데이터 준비 (SOP 데이터셋)

SOP 데이터셋은 `info_spectrum_sop.csv`에서 생성한다.  
이미 `data/sop/`가 존재하면 이 단계는 생략 가능하다.

```bash
PYTHONPATH=src python scripts/prepare_sop_input.py
```

생성 결과:

```
data/sop/
├── labels.tsv          # spec_id, smiles 매핑
├── split.tsv           # train / val / test 분할 (test: 296개)
├── spec_files/         # 스펙트럼 .ms 파일 (596개)
├── subformulae/        # 서브포뮬라 정보
├── atom_types.txt      # 원자 타입 분포
└── edge_types.txt      # 결합 타입 분포
```

선택 기준: 화합물당 (c_id, prec_type) 조합 중 `qualified=1` 또는 TIC 최고값 1개 유지.  
결과: 196개 화합물에서 298개 스펙트럼 (train/val 2 + test 296).

---

## 추론 실행

### 기본 실행 (296개 전체, 백그라운드)

```bash
WANDB_MODE=disabled PYTHONPATH=src \
python -u scripts/infer_sop.py \
  --num-samples 20 \
  --out outputs/sop_results \
  > /tmp/infer_sop.log 2>&1 &
```

GPU(RTX 3060 Ti 기준): 스펙트럼당 ~30초, 전체 ~2.5시간.

### 빠른 테스트 (3개 스펙트럼)

```bash
WANDB_MODE=disabled PYTHONPATH=src \
python -u scripts/infer_sop.py \
  --num-samples 20 \
  --max-test 3 \
  --out outputs/sop_test
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--num-samples` | 20 | 스펙트럼당 생성할 후보 분자 수 |
| `--checkpoint` | `checkpoints/checkpoints/diffms_msg.ckpt` | 모델 체크포인트 |
| `--out` | `outputs/sop_results` | 결과 저장 디렉토리 |
| `--device` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `--max-test` | None | 디버그용: 처리할 스펙트럼 수 제한 |
| `--nll-mc` | 5 | NLL 스코어링용 MC 샘플 수 (현재 미사용) |

---

## 진행 상황 확인

추론 중 언제든 확인 가능:

```bash
cat outputs/sop_results/progress.txt
```

출력 예시:

```
Progress  : 150/296  (50.7%)
Elapsed   : 1.2h  (30s/spec)
ETA       : 1h 13m
Last spec : 28s
---
SMILES validity (mean per spectrum): 77.5%
Max Tan   : 0.5301
  recall@1      : 7.8%
  recall@5      : 17.2%
  recall@10     : 20.6%
  recall@20     : 24.7%
```

로그 전체 확인:

```bash
tail -f /tmp/infer_sop.log
```

### 중간에 종료된 경우 (Resume)

결과 CSV가 존재하면 자동으로 이어서 실행된다. 동일한 명령을 그대로 재실행하면 된다.

```bash
# 동일 명령 재실행 → 이미 처리된 spec_id는 자동 스킵
WANDB_MODE=disabled PYTHONPATH=src \
python -u scripts/infer_sop.py --out outputs/sop_results
```

---

## 출력 파일

### `predictions.csv`

| 컬럼 | 설명 |
|------|------|
| `spec_id` | 스펙트럼 ID |
| `true_smiles` | 정답 SMILES |
| `valid` | 유효한 SMILES 수 (최대 20) |
| `max_tanimoto` | 20개 후보 중 정답과의 최대 Tanimoto 유사도 |
| `recall@1` | 1번 후보가 정답과 일치하면 1 |
| `recall@5` | 5개 중 하나라도 정답과 일치하면 1 |
| `recall@10` | 10개 중 하나라도 정답과 일치하면 1 |
| `recall@20` | 20개 중 하나라도 정답과 일치하면 1 |
| `elapsed_s` | 해당 스펙트럼 처리 시간(초) |
| `pred_1` ~ `pred_20` | 생성된 후보 SMILES (생성 순서, 무작위) |

### `progress.txt`

매 스펙트럼마다 갱신되는 진행 상황 파일. 위 내용 참고.

---

## 결과 해석

### 지표 설명

**recall@k**: `k`개 후보 중 하나라도 정답(InChIKey 앞 14자리 일치)이 있으면 1.  
`pred_1`~`pred_k`는 **무작위 순서**이므로 "상위 k개"가 아닌 "k번 생성 중 정답 포함 여부"다.

**SMILES validity**: 생성된 분자 중 RDKit으로 파싱 가능한 비율 (mean per spectrum).  
화학적으로 유효한 구조 여부를 나타내며, 정확도와는 별개다.

**max_tanimoto**: 20개 후보 중 정답과 가장 유사한 분자의 Morgan fingerprint Tanimoto 유사도.  
- 1.0: 정답과 동일한 구조 생성
- 0.5~0.9: 유사하지만 불일치
- < 0.3: 구조적으로 많이 다름

### SOP 기준선 (MSG 프리트레인, cross-dataset)

| 지표 | 값 |
|------|-----|
| SMILES validity | 77.5% |
| Max Tanimoto (mean) | 0.5301 |
| recall@1 | 7.8% |
| recall@5 | 17.2% |
| recall@10 | 20.6% |
| recall@20 | 24.7% |

MSG로 학습된 모델을 SOP에 직접 적용한 cross-dataset 결과다.  
약 4개 중 1개 스펙트럼에서 20번의 시도 안에 정답 구조를 생성한다.

---

## 결과 분석 (Python)

```python
import pandas as pd

df = pd.read_csv("outputs/sop_results/predictions.csv")

# 전체 recall 요약
recall_cols = [c for c in df.columns if c.startswith("recall@")]
print(df[recall_cols].mean() * 100)

# 정답을 맞춘 스펙트럼만 보기
hits = df[df["recall@20"] == 1]
print(hits[["spec_id", "true_smiles", "max_tanimoto"]])

# 특정 스펙트럼의 후보 확인
row = df[df["spec_id"] == "SOPID20210104386"].iloc[0]
pred_cols = [c for c in df.columns if c.startswith("pred_")]
for col in pred_cols:
    print(col, ":", row[col])
```

---

## 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: No module named 'models'` | PYTHONPATH 미설정 | `PYTHONPATH=src` 추가 |
| `Cannot convert MPS Tensor to float64` | MPS float64 미지원 | `PYTORCH_ENABLE_MPS_FALLBACK=1` 설정 |
| `CUDA error: device-side assert triggered` | 일부 스펙트럼에서 발생, CUDA 컨텍스트 오염 | 동일 명령 재실행 (resume 자동 적용) |
| WandB API key 오류 | WandB 로그인 필요 | `WANDB_MODE=disabled` 설정 |

---

## 디렉토리 구조

```
DiffMS/
├── checkpoints/checkpoints/
│   └── diffms_msg.ckpt         # 프리트레인 체크포인트
├── configs/
│   └── dataset/sop.yaml        # SOP 데이터셋 설정
├── data/sop/                   # SOP 데이터셋
├── scripts/
│   ├── prepare_sop_input.py    # CSV → data/sop/ 변환
│   ├── infer_sop.py            # 배치 추론 (메인)
│   └── rerank_by_nll.py        # NLL 기반 후보 재정렬 (실험용)
├── src/
│   ├── diffusion_model_spec2mol.py  # 모델 정의
│   └── spec2mol_main.py             # 학습/테스트 엔트리포인트
└── outputs/
    └── sop_results/
        ├── predictions.csv
        └── progress.txt
```
