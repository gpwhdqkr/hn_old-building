# v3 학습 파이프라인 (이원화 binclf_v3 단일 과제 — middle_id 완전 제거)

v2 파이프라인에서 midclf(결함 종류 7클래스)를 완전히 걷어낸 이원화 전용 버전.
7×7 혼동행렬을 다루지 않기로 한 결정(2026-07-28)에 따른 것으로,
**bbox 기반 defect-aware crop 등 v2의 개선점은 전부 유지**한다.

- **binclf_v3**: 이원화 — 0=우수(8.8%) / 1=불량(보통+불량 병합, 91.2%)
- middle_id는 CSV 컬럼에서부터 존재하지 않음 (`metadata_split_v3.csv`)

## v2와 다른 점 / 같은 점

| 항목 | v3 |
|---|---|
| 과제 | binclf 하나 (midclf 스크립트 자체가 없음) |
| 라벨 CSV | `metadata_split_v3.csv` — v2에서 middle_id/middle_name 컬럼만 제거, **split 배정은 행 단위 동일** (`src/make_metadata_split_v3.py`로 생성) |
| 이미지 캐시 | v2와 공유 (`data/processed_images_v2/`, 재전처리 불필요) |
| annotation | v2와 공유 (`annotations_v2.jsonl` — bbox 필드 항상 존재) |
| 학습 로직 | v2와 동일 (448 입력, defect-aware crop, 시드 고정, AMP, cosine, CLAHE 옵션) |
| 산출물 토큰 | `binclf_v3` — 체크포인트/결과 폴더가 v2와 절대 충돌하지 않음 |

## 필요한 데이터 (DATA_DIR 아래)

| 경로 | 반입 방법 |
|---|---|
| `data/processed/metadata_split_v3.csv` | git 포함. 없으면 `python src/make_metadata_split_v3.py` |
| `data/processed/annotations_v2.jsonl` | git 포함 (34MB) |
| `data/processed/image_spec_v2.csv` | git 포함 |
| `data/processed_images_v2/` (**14GB**) | git 미포함 — zip 반입 (v2와 동일, 이미 있으면 그대로) |

## 실행 순서 (런팟 리눅스, `python -m` 금지, ARCH/RUN_NAME 단계 간 동일)

```bash
cd /workspace/hn_old-building

ARCH=resnet50 RUN_NAME=v3r50a python src/v3/train_binclf_v3.py
ARCH=resnet50 RUN_NAME=v3r50a NUM_EPOCHS=15 python src/v3/fine_tune_binclf_v3.py
ARCH=resnet50 RUN_NAME=v3r50a python src/v3/evaluate_binclf_v3.py
ARCH=resnet50 RUN_NAME=v3r50a python src/v3/tune_threshold_binclf_v3.py   # 선택
ARCH=resnet50 RUN_NAME=v3r50a python src/v3/layercam_binclf_v3.py         # 선택
```

`layercam_binclf_v3.py`: 학습된 모델에 LayerCAM(다층 융합 히트맵)을 적용해
결함 위치를 실제로 가리키는지 GT bbox로 채점(pointing game + 에너지 집중도)하고
오버레이 이미지를 저장한다. 재학습 불필요. **실학습 모델로 채점해야 의미가 있고**,
점수가 낮으면 위치 표시 용도로는 detection 모델이 필요하다는 신호다.

환경변수/트러블슈팅은 `src/v2/README.md`와 동일 (DATA_DIR, BATCH_SIZE=64→OOM시 32,
USE_CLAHE, EVAL_RESIZE_MODE=centercrop 기본, LIMIT_PER_SPLIT은 스모크 전용).

**ARCH 지원 백본 (v2보다 2종 추가)**: `resnet50`(기본) / `resnet18` /
`efficientnet_b0` / `efficientnet_b2` / `mobilenet_v2` / `convnext_tiny`.
resnet18(layer4+fc)과 mobilenet_v2(features[-1]+classifier)의 fine-tune 해제
범위는 구버전 `src/binclf/` 검증 관례를 그대로 계승했다.

## 산출물 위치

| 파일 | 내용 |
|---|---|
| `model/best_<ARCH>_binclf_v3_<baseline\|finetuned>_<RUN_NAME>.pth` | 체크포인트 |
| `test_results/binclf_v3/<ARCH>/<RUN_NAME>/` | 히스토리/평가 전문/혼동행렬/샘플별 예측 CSV |
| `test_results/binclf_v3/<ARCH>/<RUN_NAME>_threshold/` | 임계값 튜닝 (RECALL_FLOOR 기본 0.85) |

## 로컬 스모크 (윈도우 CPU — 완주 확인용, 실험 결과 사용 금지)

```powershell
$env:DATA_DIR="D:/hn_old-building"; $env:RUN_NAME="smoke"; $env:ARCH="resnet50"
$env:LIMIT_PER_SPLIT="64"; $env:BATCH_SIZE="8"; $env:NUM_WORKERS="0"; $env:NUM_EPOCHS="1"
$env:OPENBLAS_NUM_THREADS="1"
python src/v3/train_binclf_v3.py
python src/v3/fine_tune_binclf_v3.py
python src/v3/evaluate_binclf_v3.py
python src/v3/tune_threshold_binclf_v3.py
```

2026-07-28 로컬 스모크 완주 확인됨 (train→fine_tune→evaluate→tune_threshold,
결과가 `test_results/binclf_v3/resnet50/smoke*`에 생성되는 것 검증).

## 결과 해석 주의 (v2와 동일)

- test는 불량 91.2% — accuracy 바닥선이 91%이므로 **우수 F1과 macro F1**로 판단.
- 기존 binclf(224 정방형)나 v2 결과와 비교할 때 전처리·해상도 차이를 명시할 것.
