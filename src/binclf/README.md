# 이원화(2클래스) 학습 파이프라인 사용설명서 — src/binclf/

우수/보통/불량 3클래스에서 **우수 vs 불량 2클래스**로 이원화한 학습 코드입니다.
**보통은 불량에 병합**했습니다 (멘토 피드백 "클래스 2개로 압축" 반영).

**런팟(리눅스) 실행을 기준**으로 작성했습니다. 기존 `src/resnet/`, `src/efficientnet/`,
`src/` 최상위(MobileNetV2) 3클래스 코드는 한 글자도 수정하지 않았습니다.

---

## 0. 라벨 정의 (가장 중요)

| model_label | class_name | 유래 | 장수 |
|---|---|---|---|
| 0 | 우수 | 기존 우수(class_id 1) | 3,266 (8.8%) |
| 1 | 불량 | 기존 보통(2) + 불량(3) 병합 | 34,019 (91.2%) |

- 데이터 정의 파일: `data/processed/metadata_split_binary3.csv` (깃에 포함)
  - 원본은 `metadata_split_binary3.xlsx`이며 `src/binclf/convert_binary_metadata.py`로
    변환·검증한 것. 런팟에서 변환을 다시 할 필요 없음 (git pull이면 끝).
  - split(train/validation/test)은 기존 `metadata_split.csv`와 완전히 동일 (재분리 안 함).
  - ⚠ `metadata_split_binary2.csv`는 한글이 깨진 손상본이므로 절대 쓰지 마세요.
- **1:10 불균형 주의**: 전부 "불량"으로 찍어도 accuracy ~91%가 나옵니다.
  accuracy가 아니라 **우수 F1 / macro F1**을 봐야 하며, 그래서 최고 모델 선택 기준도
  기존(validation accuracy)과 달리 **validation macro F1**입니다.

## 1. 설치 — 제일 먼저 할 일

```bash
pip install pandas scikit-learn pillow
```

### torch / torchvision 은 설치하지 마세요

런팟 PyTorch 템플릿에는 CUDA 빌드가 이미 깔려 있습니다. `pip install torch`를 치면
CPU 빌드로 덮어써서 GPU를 못 쓰게 되는 경우가 많습니다. 확인만 하세요:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

- `True`면 손대지 마세요. torchvision이 0.16 미만이면 `transforms.v2` 에러 → 업그레이드 필요.
- openpyxl은 필요 없습니다 (xlsx 변환은 로컬에서 이미 완료, 학습 코드는 CSV만 읽음).

## 2. 데이터 준비

```bash
cd /workspace/hn_old-building
git pull   # metadata_split_binary3.csv + src/binclf/ 코드가 이걸로 도착
```

| 준비물 | 위치 | 확인 |
|---|---|---|
| 이원화 CSV | `data/processed/metadata_split_binary3.csv` | 깃에 포함 |
| 변환 이미지 | `data/processed_images/` | 용량 문제로 깃에 없음 → 구글드라이브 zip을 이 위치에 풀기 |

데이터 로딩 점검 (모델별 preprocess를 직접 실행하면 분포/배치 shape 확인):

```bash
python src/binclf/resnet/preprocess_binclf_resnet.py
```

## 3. 실행 순서 (모델별 4단계)

세 모델(mobilenet / resnet / efficientnet) 모두 같은 구조입니다:
`train(baseline) → fine_tune → evaluate → (선택) tune_threshold`

같은 모델의 단계 간에는 **RUN_NAME(과 RESNET_ARCH)을 반드시 동일하게** 넘겨야 합니다.

### ResNet (RESNET_ARCH로 resnet18/resnet50 선택)

```bash
RESNET_ARCH=resnet18 RUN_NAME=bin1 NUM_EPOCHS=20 python src/binclf/resnet/train_binclf_resnet.py
RESNET_ARCH=resnet18 RUN_NAME=bin1 NUM_EPOCHS=20 python src/binclf/resnet/fine_tune_binclf_resnet.py
RESNET_ARCH=resnet18 RUN_NAME=bin1 python src/binclf/resnet/evaluate_binclf_resnet.py
RESNET_ARCH=resnet18 RUN_NAME=bin1 python src/binclf/resnet/tune_threshold_binclf_resnet.py
```

### EfficientNet-B0

```bash
RUN_NAME=bin1 NUM_EPOCHS=20 python src/binclf/efficientnet/train_binclf_efficientnet.py
RUN_NAME=bin1 NUM_EPOCHS=20 python src/binclf/efficientnet/fine_tune_binclf_efficientnet.py
RUN_NAME=bin1 python src/binclf/efficientnet/evaluate_binclf_efficientnet.py
RUN_NAME=bin1 python src/binclf/efficientnet/tune_threshold_binclf_efficientnet.py
```

### MobileNetV2

```bash
RUN_NAME=bin1 NUM_EPOCHS=20 python src/binclf/mobilenet/train_binclf_mobilenet.py
RUN_NAME=bin1 NUM_EPOCHS=20 python src/binclf/mobilenet/fine_tune_binclf_mobilenet.py
RUN_NAME=bin1 python src/binclf/mobilenet/evaluate_binclf_mobilenet.py
RUN_NAME=bin1 python src/binclf/mobilenet/tune_threshold_binclf_mobilenet.py
```

> 스크립트는 반드시 **파일 경로**로 실행하세요. `python -m src.binclf...`은 안 됩니다
> (같은 폴더의 preprocess 모듈을 직접 import하는 구조라서).

## 4. 전체 환경변수 정리표

| 변수 | 기본값 | 적용 스크립트 | 설명 |
|---|---|---|---|
| `RUN_NAME` | `default` | 전체 | 실험 이름. 저장 파일명에 포함 (단계 간 동일하게!) |
| `RESNET_ARCH` | `resnet18` | resnet만 | `resnet18` 또는 `resnet50` |
| `NUM_EPOCHS` | `10` | train/fine_tune | fine_tune은 patience 3 조기종료 있음 |
| `BATCH_SIZE` | resnet/effnet `128`, mobilenet `50` | 전체 | OOM 시 낮추기 (threshold 스크립트는 기본 64) |
| `EVAL_TARGET` | `finetuned` | evaluate/threshold | `baseline` 또는 `finetuned` |
| `RECALL_FLOOR` | `0.85` | tune_threshold | 불량 recall 하한 제약 |
| `DATA_DIR` | 프로젝트 루트 | 전체 | 데이터를 다른 위치에 뒀을 때만 지정 |
| `LIMIT_PER_SPLIT` | `0`(전체) | 전체 | **스모크 테스트 전용.** 실험 결과에 사용 금지 |

mobilenet의 batch 기본값 50은 기존 3클래스 실험(src/preprocess_data.py)과의
공정 비교를 위한 값입니다. GPU 여유가 있으면 `BATCH_SIZE=128`로 올려도 되지만
그 경우 결과표에 배치 크기를 명시하세요.

## 5. 백그라운드 실행 (SSH 끊김 대비)

```bash
nohup env RESNET_ARCH=resnet18 RUN_NAME=bin1 NUM_EPOCHS=20 \
  python src/binclf/resnet/train_binclf_resnet.py > train_bin1.log 2>&1 &

tail -f train_bin1.log   # 진행 상황 확인
```

## 6. 결과 파일 위치

| 산출물 | 경로 |
|---|---|
| baseline 모델 | `model/best_<arch>_binclf_baseline_<RUN_NAME>.pth` |
| finetuned 모델 | `model/best_<arch>_binclf_finetuned_<RUN_NAME>.pth` |
| 평가 리포트 | `test_results/binclf/<arch>/<RUN_NAME>/evaluate_<EVAL_TARGET>_result.txt` |
| 혼동행렬(2×2) | `test_results/binclf/<arch>/<RUN_NAME>/test_confusion_matrix_<EVAL_TARGET>.csv` |
| threshold 실험 | `test_results/binclf/<arch>/<RUN_NAME>_threshold/` |

`<arch>`는 `mobilenet_v2` / `resnet18` / `resnet50` / `efficientnet_b0`.
파일명의 `binclf` 토큰 덕에 기존 3클래스 실험 산출물과 절대 겹치지 않습니다.

## 7. 로컬 스모크 테스트 (클라우드 올리기 전 확인용)

```bash
LIMIT_PER_SPLIT=200 BATCH_SIZE=16 NUM_EPOCHS=1 RUN_NAME=smoke \
  python src/binclf/resnet/train_binclf_resnet.py
```

PowerShell(윈도우 로컬):

```powershell
$env:LIMIT_PER_SPLIT="200"; $env:BATCH_SIZE="16"; $env:NUM_EPOCHS="1"; $env:RUN_NAME="smoke"
python src/binclf/resnet/train_binclf_resnet.py
```

스모크 산출물(RUN_NAME=smoke)은 커밋하지 마세요.

## 8. 자주 나는 오류

| 증상 | 원인/해결 |
|---|---|
| `ModuleNotFoundError: sklearn` | 1번 설치 안 함. `pip install pandas scikit-learn pillow` |
| `metadata_split_binary3.csv를 찾을 수 없습니다` | git pull을 안 했거나 DATA_DIR이 틀림. 로컬 재생성은 `python src/binclf/convert_binary_metadata.py` |
| `model_label에 0/1 외 값이 있습니다` | 3클래스 CSV를 잘못 지정. binary3.csv인지 확인 |
| `이미지 폴더를 찾을 수 없습니다` | `data/processed_images/` 압축을 안 풂 |
| `CUDA out of memory` | `BATCH_SIZE=64`(resnet50은 더 낮게)로 하향 |
| `best_..._binclf_baseline_....pth 없음` | train 단계를 건너뛰었거나 RUN_NAME/RESNET_ARCH가 단계 간 다름 |
| `python -m` 실행 실패 | 파일 경로로 실행해야 함 (`python src/binclf/...py`) |

## 9. 결과 해석 시 주의할 점

- **accuracy는 바닥선이 ~91%** (전부 불량 예측). 반드시 우수 F1 / 우수 recall /
  macro F1을 같이 보세요. 핵심 지표는 불량 F1(놓치면 안 되는 클래스)과 우수 F1(이원화로
  살리려는 소수 클래스) 둘 다입니다.
- 최고 모델 선택 기준이 기존 3클래스 실험(validation accuracy)과 다릅니다
  (validation **macro F1**). 기존 결과와 나란히 놓을 때 이 차이를 표에 명시하세요.
- 클래스 가중치는 하드코딩이 아니라 train 분포에서 자동 계산됩니다 (약 [5.68, 0.55]).
- tune_threshold는 재학습 없이 운영점(불량 확률 임계값)만 조정하는 실험입니다.
  탐색은 validation에서만 하므로 test 누출이 없습니다.
