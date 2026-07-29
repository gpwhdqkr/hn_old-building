# ResNet 학습·평가 파이프라인 사용설명서

`src/resnet/` 에 추가된 5개 스크립트의 사용법입니다.
**런팟(리눅스) 실행을 기준**으로 작성했습니다.

윈도우 로컬에서 돌릴 경우의 차이는 맨 아래 [부록 A](#부록-a-윈도우-로컬에서-실행할-때) 를 보세요.

---

## 목차

1. [설치 — 제일 먼저 할 일](#1-설치--제일-먼저-할-일)
2. [데이터 준비](#2-데이터-준비)
3. [환경변수 사용법](#3-환경변수-사용법)
4. [실행 순서 (4단계)](#4-실행-순서-4단계)
   - [1단계 · baseline 학습](#1단계--baseline-학습-train_resnetpy)
   - [2단계 · 파인튜닝](#2단계--파인튜닝-fine_tune_resnetpy)
   - [3단계 · Test 평가 + 추론속도](#3단계--test-평가--추론속도-evaluate_resnetpy)
   - [4단계 · 모델 4종 비교표](#4단계--모델-4종-비교표-benchmark_inferencepy)
5. [전체 환경변수 정리표](#5-전체-환경변수-정리표)
6. [한 번에 돌리기 / 백그라운드 실행](#6-한-번에-돌리기--백그라운드-실행)
7. [결과 파일 위치](#7-결과-파일-위치)
8. [자주 나는 오류](#8-자주-나는-오류)
9. [결과 해석 시 주의할 점](#9-결과-해석-시-주의할-점)

---

## 1. 설치 — 제일 먼저 할 일

**아무것도 하기 전에 라이브러리부터 설치합니다.**

```bash
pip install pandas scikit-learn pillow
```

이 세 개가 없으면 1단계에서 바로 `ModuleNotFoundError` 가 납니다.

> import 이름과 패키지 이름이 다릅니다.
> `sklearn` → `scikit-learn`, `PIL` → `pillow`.
> `pip install sklearn` 은 빈 껍데기 패키지라 에러가 납니다.

### torch / torchvision 은 설치하지 마세요

런팟 PyTorch 템플릿에는 CUDA 빌드가 **이미 깔려 있습니다.**
여기서 `pip install torch` 를 치면 CPU 빌드나 CUDA 버전이 안 맞는 걸로
덮어써서 GPU를 못 쓰게 되는 경우가 많습니다.

설치 상태를 먼저 확인하세요.

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

- `True` 가 나오면 → 손대지 마세요.
- torchvision 이 **0.16 미만**이면 → `transforms.v2` 에서 에러가 납니다. 업그레이드 필요.
- `False` 가 나오거나 torch 자체가 없으면 → 그때만 아래를 실행합니다.
  (`cu121` 부분은 `nvidia-smi` 로 확인한 CUDA 버전에 맞추세요)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 설치 최종 확인

```bash
python -c "import torch, torchvision, pandas, sklearn, PIL; print('전부 OK')"
```

`전부 OK` 가 뜨면 다음 단계로 넘어갑니다.

### 필요한 패키지 목록

| import 이름 | pip 패키지 | 비고 |
|---|---|---|
| `torch` | (런팟 기본 제공) | 설치 금지 |
| `torchvision` | (런팟 기본 제공) | 0.16 이상 필요 |
| `pandas` | `pandas` | |
| `sklearn` | `scikit-learn` | 이름 다름 |
| `PIL` | `pillow` | 이름 다름 |

> 프로젝트의 다른 부분(웹앱, Grad-CAM, 고전 ML)은 `flask`, `pymongo`,
> `opencv-python`, `grad-cam`, `matplotlib`, `numpy` 도 쓰지만
> **ResNet 학습만 돌릴 거면 필요 없습니다.**

---

## 2. 데이터 준비

```bash
cd /workspace/hn_old-building
```

브랜치와 데이터를 확인합니다.

```bash
git branch --show-current
ls src/resnet/
ls data/processed/metadata_split.csv
ls data/processed_images | head -3
```

| 필요한 것 | 위치 | 비고 |
|---|---|---|
| 스크립트 5개 | `src/resnet/` | **`Dev` 브랜치에만 있음** (`main` 에는 없음) |
| split CSV | `data/processed/metadata_split.csv` | 깃에 포함되어 있음 |
| 변환 이미지 | `data/processed_images/` | 용량 문제로 깃에 없음 → 구글드라이브에서 받아 이 위치에 풀기 |

### 데이터 로딩 점검 (선택)

`preprocess_resnet.py` 는 데이터로더 정의 파일이라 보통 직접 실행할 필요가 없지만,
데이터가 제대로 읽히는지 미리 확인하고 싶으면 돌려보세요.

```bash
python src/resnet/preprocess_resnet.py
```

정상이면 Train/Validation/Test 장수와 첫 배치 형태가 출력됩니다.

```
이미지 묶음 형태 : torch.Size([128, 3, 224, 224])
정답 묶음 형태 : torch.Size([128])
```

> **전처리와 split은 수정하지 마세요.**
> transform(증강/정규화)은 팀원 `preprocess_data.py`, EfficientNet 버전과
> 완전히 동일하고, split도 팀원의 `metadata_split.csv` 를 그대로 씁니다.
> 그래야 test 결과 차이를 "모델 차이"로만 해석할 수 있습니다.

---

## 3. 환경변수 사용법

리눅스는 명령 앞에 `VAR=값` 을 붙입니다. **그 명령 한 번에만** 적용됩니다.

```bash
RESNET_ARCH=resnet18 RUN_NAME=r18_e20 python src/resnet/train_resnet.py
```

매번 치기 번거로우면 세션 전체에 걸어둘 수 있습니다. 팟 재시작 전까지 유지됩니다.

```bash
export RESNET_ARCH=resnet18 RUN_NAME=r18_e20 NUM_EPOCHS=20
```

### 주의사항

- **`set` 이 아니라 `export` 입니다.** 리눅스에서 `set "RESNET_ARCH=resnet18"` 은
  환경변수를 만들지 않습니다(위치 매개변수에 문자열이 들어갈 뿐). 에러도 안 나고
  조용히 실패하니 `echo "[$RESNET_ARCH]"` 로 확인하세요.
- **`export` 값은 계속 남아있습니다.** 다른 실험을 시작할 때 이전 `RUN_NAME` 이
  남아있으면 결과 파일을 덮어쓰거나 엉뚱한 baseline을 불러옵니다.
- **`RUN_NAME` 은 꼭 지정하세요.** 미지정 시 `default` 로 잡혀서, 나중에 다른 조건으로
  또 `default` 를 돌리면 이전 결과가 사라집니다.

---

## 4. 실행 순서 (4단계)

1단계 → 2단계 → 3단계는 **순서대로** 실행해야 합니다.
각 단계에서 `RESNET_ARCH` 와 `RUN_NAME` 을 **똑같이** 넘겨야 앞 단계 결과물을 찾습니다.

### 1단계 · baseline 학습 (`train_resnet.py`)

ImageNet 사전학습 가중치를 불러와 특징 추출부는 전부 동결하고,
마지막 분류층(`fc`)만 우수/보통/불량 3등급 분류로 학습합니다.

```bash
RESNET_ARCH=resnet18 RUN_NAME=r18_e20 NUM_EPOCHS=20 python src/resnet/train_resnet.py
```

**결과물** → `model/best_resnet18_baseline_r18_e20.pth`

validation **정확도**가 가장 높았던 시점의 모델이 저장됩니다
(팀원 MobileNetV2 / EfficientNet과 동일 기준).
불량 F1은 매 epoch 참고용으로 출력만 되고 모델 선택에는 쓰지 않습니다.

resnet50을 돌릴 때는 배치를 낮춥니다.

```bash
RESNET_ARCH=resnet50 RUN_NAME=r50_e20 NUM_EPOCHS=20 BATCH_SIZE=64 python src/resnet/train_resnet.py
```

---

### 2단계 · 파인튜닝 (`fine_tune_resnet.py`)

1단계 baseline을 불러와 마지막 특징 블록(`layer4`)까지 동결을 풀고
아주 작은 학습률(1e-5)로 조금 더 학습합니다.

**1단계를 먼저 실행해야 합니다.** `RESNET_ARCH` 와 `RUN_NAME` 이 1단계와 같아야
그 baseline 파일을 찾습니다.

```bash
RESNET_ARCH=resnet18 RUN_NAME=r18_e20 NUM_EPOCHS=20 python src/resnet/fine_tune_resnet.py
```

**결과물** → `model/best_resnet18_finetuned_r18_e20.pth`

조기 종료(patience)가 있어서 `NUM_EPOCHS=20` 이어도 개선이 멈추면 먼저 끝납니다.
한 번도 baseline보다 좋아지지 않으면 baseline 상태 그대로 저장됩니다.

---

### 3단계 · Test 평가 + 추론속도 (`evaluate_resnet.py`)

등급별 F1 / 정확도 / 혼동행렬에 더해 **추론속도**를 함께 측정합니다.
(멘토 피드백 "모델의 추론속도를 결과표에 추가" 반영)

```bash
RESNET_ARCH=resnet18 RUN_NAME=r18_e20 EVAL_TARGET=finetuned python src/resnet/evaluate_resnet.py
```

**결과물**
- `test_results/resnet18/r18_e20/evaluate_finetuned_result.txt`
- `test_results/resnet18/r18_e20/test_confusion_matrix_finetuned.csv`

화면 출력이 그대로 txt로 저장되므로 `| tee` 같은 건 필요 없습니다.

`EVAL_TARGET=baseline` 으로 바꾸면 1단계 모델을 평가합니다.

#### 속도를 세 층으로 나눠 재는 이유

기존 `evaluate_augmented.py` 는 dataloader 루프 전체를 감싼 수치 하나뿐이었는데,
거기엔 이미지 파일 읽기·전처리 시간이 섞여 있습니다. 그건 "이 서버 디스크가
빠른가"에 가까운 값이지 "이 모델이 빠른가"가 아닙니다. 그래서 분리했습니다.

| 측정 항목 | 의미 |
|---|---|
| **단일 이미지 지연** (batch=1, 중앙값/p95) | 실서비스 체감 기준. **가장 중요** |
| **배치 처리량** (데이터 로딩 제외, 순수 forward) | 대량 일괄 추론 기준 |
| **end-to-end** (dataloader 포함) | 기존 실험과 비교 연속성 유지용 |

---

### 4단계 · 모델 4종 비교표 (`benchmark_inference.py`)

MobileNetV2 / EfficientNet-B0 / ResNet18 / ResNet50 을 한 번에 로드해
성능(F1)과 추론속도를 **한 표로** 뽑습니다. 멘토 피드백에 대한 최종 산출물입니다.

```bash
RESNET18_RUN=r18_e20 RESNET50_RUN=r50_e20 EFFICIENTNET_RUN=default python src/resnet/benchmark_inference.py
```

**결과물**
- `test_results/inference_benchmark/<RUN_NAME>/benchmark.txt`
- `test_results/inference_benchmark/<RUN_NAME>/benchmark.csv`

#### 실행 전 모델 파일 확인

모델별로 실험 이름을 따로 받는 이유는, 모델마다 다른 `RUN_NAME` 으로
학습했을 수 있기 때문입니다. 찾는 파일 이름은 다음과 같습니다.

```
model/best_mobilenet_v2_finetuned.pth                      ← 팀원 것, 실험 이름 없음
model/best_efficientnet_b0_finetuned_<EFFICIENTNET_RUN>.pth
model/best_resnet18_finetuned_<RESNET18_RUN>.pth
model/best_resnet50_finetuned_<RESNET50_RUN>.pth
```

```bash
ls -1 model/
```

없는 모델은 **조용히 빠지지 않고** "측정 안 됨"으로 결과에 명시됩니다.
(조용히 빠지면 표를 보는 사람이 "측정 안 됨"을 "없음"으로 오해하게 되므로)

#### CPU 지연도 재는 이유

배포 환경이 CPU라면 **런팟 GPU 수치는 배포 수치가 아닙니다.**
`MEASURE_CPU=0` 으로 끌 수 있지만, 배포 근거로 쓸 자료라면 켜두세요.

결과 파일에는 GPU 모델명과 torch/CUDA 버전이 함께 기록됩니다.
하드웨어가 명시되지 않은 속도 수치는 근거로 쓸 수 없기 때문입니다.

---

## 5. 전체 환경변수 정리표

| 환경변수 | 기본값 | 적용 스크립트 | 설명 |
|---|---|---|---|
| `RESNET_ARCH` | `resnet18` | train / fine_tune / evaluate | `resnet18` 또는 `resnet50` |
| `RUN_NAME` | `default` | 전부 | 실험 이름. 파일명·폴더명에 붙음 |
| `NUM_EPOCHS` | `10` | train / fine_tune | epoch 수 |
| `BATCH_SIZE` | `128` | 전부 | resnet50 OOM 시 `64` |
| `EVAL_TARGET` | `finetuned` | evaluate / benchmark | `finetuned` 또는 `baseline` |
| `RESNET18_RUN` | `default` | benchmark | resnet18 실험 이름 |
| `RESNET50_RUN` | `default` | benchmark | resnet50 실험 이름 |
| `EFFICIENTNET_RUN` | `default` | benchmark | EfficientNet-B0 실험 이름 |
| `MEASURE_CPU` | `1` | benchmark | `0` 이면 CPU 지연 측정 생략 |

---

## 6. 한 번에 돌리기 / 백그라운드 실행

### resnet18 전체 파이프라인

```bash
export RESNET_ARCH=resnet18 RUN_NAME=r18_e20 NUM_EPOCHS=20 && python src/resnet/train_resnet.py && python src/resnet/fine_tune_resnet.py && EVAL_TARGET=finetuned python src/resnet/evaluate_resnet.py
```

### SSH가 끊겨도 계속 돌리기

학습이 길면 `nohup` 으로 띄우세요.

```bash
nohup env RESNET_ARCH=resnet18 RUN_NAME=r18_e20 NUM_EPOCHS=20 python src/resnet/train_resnet.py > train_r18_e20.log 2>&1 &
```

```bash
tail -f train_r18_e20.log
```

### GPU 사용률 확인

```bash
watch -n 1 nvidia-smi
```

GPU-Util 이 계속 낮으면 데이터 로딩 병목입니다.
워커 수는 CPU 코어 수에 맞춰 자동 계산되지만(코어의 절반, 최소 4 최대 32),
그래도 낮으면 `BATCH_SIZE` 를 올려보세요.

---

## 7. 결과 파일 위치

```
model/
  best_<ARCH>_baseline_<RUN_NAME>.pth        ← 1단계 결과
  best_<ARCH>_finetuned_<RUN_NAME>.pth       ← 2단계 결과

test_results/
  <ARCH>/<RUN_NAME>/
    evaluate_<EVAL_TARGET>_result.txt        ← 3단계 결과 (화면 출력 전문)
    test_confusion_matrix_<EVAL_TARGET>.csv
  inference_benchmark/<RUN_NAME>/
    benchmark.txt                            ← 4단계 결과
    benchmark.csv
```

`RUN_NAME` 으로 폴더가 나뉘므로 여러 실험을 돌려도 서로 덮어쓰지 않습니다.
기존 `test_results/efficientnet_b0/` 와도 분리됩니다.

---

## 8. 자주 나는 오류

| 증상 | 원인과 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | [1번](#1-설치--제일-먼저-할-일) 설치를 안 함. `pip install pandas scikit-learn pillow` |
| `can't open file '.../srcresnetpreprocess_resnet.py'` | 백슬래시를 씀. 리눅스는 **슬래시** — `src/resnet/...` |
| `FileNotFoundError: metadata_split.csv` | 프로젝트 루트가 아닌 곳에서 실행했거나 깃에서 파일을 못 받음 |
| `이미지 폴더를 찾을 수 없습니다` | `data/processed_images/` 압축을 안 풂 |
| `RuntimeError: CUDA out of memory` | `BATCH_SIZE=64` 로 낮추기. resnet50이 EfficientNet-B0보다 activation 메모리를 훨씬 많이 씀 |
| `FileNotFoundError: best_resnet18_baseline_*.pth` | 2단계를 1단계 없이 실행했거나 `RUN_NAME` 이 1단계와 다름 |
| `python -m src.resnet.train_resnet` 실패 | `-m` 으로는 안 됨. 스크립트가 `from preprocess_resnet import ...` 로 같은 폴더를 직접 import하므로 **파일 경로**로 실행 |
| 환경변수를 지정했는데 반영이 안 됨 | `set` 이 아니라 `export`. `echo "[$RUN_NAME]"` 으로 확인 |
| GPU-Util 이 5% 근처 | 데이터 로딩 병목. `BATCH_SIZE` 를 올리거나 CPU 코어가 많은 팟인지 확인 |

---

## 9. 결과 해석 시 주의할 점

### 핵심 지표

`classification_report` 의 **"불량" 행 f1-score** 입니다.
프로젝트 목표가 불량 탐지이므로 불량이 positive 입니다.

팀 기존 수치는 다음과 같습니다.

| 모델 | 불량 F1 | 보통 F1 | Macro |
|---|---|---|---|
| MobileNetV2 baseline | 0.9088 | 0.3814 | 0.6626 |
| EfficientNet-B0 finetuned | 0.9087 | 0.3907 | 0.6589 |

ResNet18도 이 근처에서 나와야 정상입니다.
**크게 벗어나면 모델 성능 차이가 아니라 새 코드를 먼저 의심하세요.**

### 파인튜닝 규모 차이 (결과표에 반드시 명시)

"마지막 특징 블록만 푼다"는 방침은 EfficientNet과 같지만,
푸는 규모가 모델마다 다릅니다.

| 모델 | 푸는 부분 | 학습 파라미터 | 전체 대비 |
|---|---|---|---|
| EfficientNet-B0 | `features[-1]` | 0.41M | 8% |
| ResNet18 | `layer4` | 8.4M | **72%** |
| ResNet50 | `layer4` | 15.0M | **59%** |

ResNet의 `layer4` 는 위치상으로는 대응되지만 사실상 모델 대부분을 푸는 셈이라
EfficientNet보다 공격적인 파인튜닝입니다. lr이 1e-5로 낮고 train이 26k장이라
학습 자체는 안정적이지만, **결과표에 이 조건 차이를 적어야 공정한 비교**가 됩니다.
(그래서 체크포인트에 `trainable_param_count` 를 기록하고 evaluate가 출력합니다)

### 속도 측정의 신뢰성

측정 코드는 다음을 지키고 있습니다.

- **워밍업 후 측정** — cuDNN 알고리즘 선택과 CUDA 초기화가 첫 몇 회를 크게 부풀림
- **`torch.cuda.synchronize()`** — CUDA는 비동기 실행이라 이걸 빼면 '커널 제출 시간'만
  재게 되어 숫자가 실제보다 훨씬 빠르게 나옴 (속도 측정의 가장 흔한 실수)
- **평균이 아닌 중앙값** — 클라우드는 다른 사용자 부하로 가끔 크게 튀는데 평균은 거기 끌려감
- **모든 모델에 동일한 배치 크기·해상도·데이터로더** 적용

**속도 숫자는 측정한 하드웨어에서만 유효합니다.** 결과 파일에 GPU 모델명과
torch/CUDA 버전이 함께 기록되는 이유입니다.

---

## 부록 A. 윈도우 로컬에서 실행할 때

리눅스식 `VAR=값 python ...` 인라인 문법은 윈도우에서 동작하지 않습니다.

**CMD**

```
set "RESNET_ARCH=resnet18"
set "RUN_NAME=r18_e20"
python src\resnet\train_resnet.py
```

값에 따옴표를 감싸세요. `set RUN_NAME=r18_e20 && python ...` 은 값 뒤에
공백이 붙어 `r18_e20 ` 이라는 이름의 폴더가 생깁니다.

**PowerShell**

```powershell
$env:RESNET_ARCH="resnet18"; $env:RUN_NAME="r18_e20"
python src\resnet\train_resnet.py
```

> **로컬 CPU 학습은 현실적이지 않습니다.** GPU가 없으면 자동으로 CPU를 쓰지만
> train 26k장 기준 매우 느립니다. 로컬은 코드가 도는지 확인하는 용도
> (`NUM_EPOCHS=1`)로만 쓰고, 실제 학습은 런팟에서 하세요.

---

## 부록 B. 파일별 역할 요약

| 파일 | 역할 | 직접 실행 |
|---|---|---|
| `preprocess_resnet.py` | 데이터셋·데이터로더 정의 | 점검용으로만 |
| `train_resnet.py` | 1단계 baseline 학습 (`fc` 만) | O |
| `fine_tune_resnet.py` | 2단계 파인튜닝 (`layer4` 해제) | O |
| `evaluate_resnet.py` | 3단계 Test 평가 + 추론속도 | O |
| `benchmark_inference.py` | 모델 4종 성능·속도 비교표 | O |

`RESNET_ARCH` 환경변수로 resnet18/resnet50을 전환합니다.
두 모델은 `fc`/`layer4` 구조가 같아서 코드가 한 벌로 처리됩니다.
