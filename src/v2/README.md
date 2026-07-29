# v2 학습 파이프라인 (이원화 binclf_v2 + 결함종류 midclf_v2)

전처리 v2 산출물(종횡비 유지 768 캐시 + 폴리곤/bbox)로 두 모델을 학습한다.

- **binclf_v2**: 이원화 — 0=우수(3,266장, 8.8%) / 1=불량(보통+불량 병합, 34,019장, 91.2%)
- **midclf_v2**: 결함 종류 7클래스 — C(균열)/P(박리,박락)/X(철근노출)/F(대지)/T(마감)/L(생활)/W(창호), 추론은 softmax 최고 확률

기존 `src/binclf/` 대비 개선: 입력 448(종횡비 유지 캐시), defect-aware crop(불량은 폴리곤/bbox 중심 크롭 → 라벨 노이즈 방지), 시드 고정, epoch 히스토리 CSV, 샘플별 예측 CSV, cosine 스케줄러 + AMP, CLAHE 옵션.

## 0. 사전 확정 사항

- **EVAL_RESIZE_MODE 기본값 = `centercrop`** — `measure_centercrop_miss_v2.py` 실측 결과
  (val/test 불량 10,389장 전수: 완전 누락 0.45%, coverage<0.5 비율 2.32% — 기준(2%/10%) 이내).
  리포트: `test_results/v2_checks/centercrop_miss_report.txt`
- annotation 좌표는 두 형태를 모두 사용: `polygon`(평탄화 [x1,y1,...]) 45,790개 + `bbox`([x0,y0,x1,y1] 코너) 31,642개

## 1. 필요한 데이터 (DATA_DIR 아래)

| 경로 | 반입 방법 |
|---|---|
| `data/processed/metadata_split_v2.csv` | git 포함 여부 확인, 없으면 로컬에서 복사 |
| `data/processed/annotations_v2.jsonl` | 〃 (34MB) |
| `data/processed/image_spec_v2.csv` | 〃 |
| `data/processed_images_v2/` (**14GB**) | git 미포함 — zip으로 반입해 이 위치에 해제 (기존 processed_images와 동일 방식) |

## 2. 런팟 설치

PyTorch 템플릿 팟 기준. **torch/torchvision 재설치 금지** (CUDA 빌드가 CPU 빌드로 덮이는 사고 방지).

```
pip install pandas scikit-learn pillow opencv-python-headless
python -c "import torch, torchvision, cv2; print(torch.__version__, torch.cuda.is_available())"
```

## 3. 실행 순서 (리눅스 인라인 env)

`python -m` 사용 금지 — 반드시 파일 경로로 실행. 단계 간 `ARCH`/`RUN_NAME` 동일 필수.

```bash
cd /workspace/hn_old-building

# (선택) 사전 측정 재확인 — 이미 centercrop으로 확정됨
python src/v2/measure_centercrop_miss_v2.py

# 이원화
ARCH=resnet50 RUN_NAME=v2r50a python src/v2/train_binclf_v2.py
ARCH=resnet50 RUN_NAME=v2r50a NUM_EPOCHS=15 python src/v2/fine_tune_binclf_v2.py
ARCH=resnet50 RUN_NAME=v2r50a python src/v2/evaluate_binclf_v2.py
ARCH=resnet50 RUN_NAME=v2r50a python src/v2/tune_threshold_binclf_v2.py   # 선택

# 결함종류 7클래스
ARCH=resnet50 RUN_NAME=v2r50a python src/v2/train_midclf_v2.py
ARCH=resnet50 RUN_NAME=v2r50a NUM_EPOCHS=15 python src/v2/fine_tune_midclf_v2.py
ARCH=resnet50 RUN_NAME=v2r50a python src/v2/evaluate_midclf_v2.py
```

백그라운드: `nohup env ARCH=resnet50 RUN_NAME=v2r50a python src/v2/train_binclf_v2.py > train_v2r50a.log 2>&1 &` + `tail -f`.

CLAHE(음영 완화) 비교 실험: `USE_CLAHE=1 RUN_NAME=v2r50a_clahe`로 같은 순서를 한 번 더 — **학습·평가 전 단계에서 같은 값**을 써야 한다 (체크포인트에 기록되며 불일치 시 evaluate가 경고).

## 4. 환경변수

| env | 기본값 | 설명 |
|---|---|---|
| `DATA_DIR` | 저장소 루트 | data/, model/, test_results/ 상위 |
| `RUN_NAME` | default | 모든 산출물 파일명에 포함 (실험 덮어쓰기 방지) |
| `ARCH` | resnet50 | resnet50 / efficientnet_b0 / efficientnet_b2 / convnext_tiny (convnext는 AdamW 자동) |
| `SEED` | 42 | 셔플/크롭/초기화 재현용 (완전 결정성은 아님) |
| `BATCH_SIZE` | 64 | 448 입력 기준. **CUDA OOM 시 32** |
| `NUM_WORKERS` | 자동 | 로컬 윈도우 스모크 2, 디버깅 0 권장 |
| `NUM_EPOCHS` | train 5 / fine_tune 15 | fine_tune은 patience로 조기 종료 |
| `PATIENCE` | 3 | fine_tune 조기 종료 |
| `LR_HEAD` / `LR_BACKBONE` / `LR_HEAD_FINETUNE` | 1e-3 / 1e-5 / 1e-4 | 차등 학습률 |
| `USE_SCHEDULER` | 1 | fine_tune cosine. 0이면 기존 binclf와 동일 조건 |
| `USE_AMP` | 1 | CUDA에서만 유효 (CPU는 자동 무시) |
| `USE_CLAHE` | 0 | 1이면 LAB L채널 CLAHE — train/평가 동일값 필수 |
| `EVAL_RESIZE_MODE` | centercrop | 실측으로 확정된 기본값. letterbox는 비교 실험용 |
| `DEFECT_CROP_P` | 1.0 | 불량 이미지에 defect-aware crop 적용 확률 |
| `LIMIT_PER_SPLIT` | 0 | 스모크 전용 — 실험 결과에 절대 사용 금지 |
| `EVAL_TARGET` | finetuned | evaluate/tune_threshold 대상 (baseline 가능) |
| `RECALL_FLOOR` | 0.85 | tune_threshold의 불량 recall 하한 |

## 5. 산출물 위치

| 파일 | 내용 |
|---|---|
| `model/best_<ARCH>_<binclf_v2\|midclf_v2>_<baseline\|finetuned>_<RUN_NAME>.pth` | 체크포인트 (state_dict + 실험 메타, weights_only=True 호환) |
| `test_results/<binclf_v2\|midclf_v2>/<ARCH>/<RUN_NAME>/train_history_*.csv` | epoch별 loss/F1/lr (v2 신규) |
| `.../evaluate_<EVAL_TARGET>_result.txt` | 화면 출력 전문 (기존 binclf 포맷 유지 — ppt 파서 호환) |
| `.../test_confusion_matrix_<EVAL_TARGET>.csv` | binclf 2x2 / midclf 7x7(middle_name 라벨) |
| `.../predictions_<EVAL_TARGET>.csv` | 샘플별 확률/예측 (v2 신규 — 오답 분석용) |
| `test_results/binclf_v2/<ARCH>/<RUN_NAME>_threshold/` | 임계값 튜닝 스윕/혼동행렬/val 예측 |

## 6. 로컬 스모크 (윈도우, CPU)

```powershell
$env:DATA_DIR="D:/hn_old-building"; $env:RUN_NAME="smoke"; $env:ARCH="resnet50"
$env:LIMIT_PER_SPLIT="64"; $env:BATCH_SIZE="8"; $env:NUM_WORKERS="2"; $env:NUM_EPOCHS="1"
python src/v2/train_binclf_v2.py
python src/v2/fine_tune_binclf_v2.py
python src/v2/evaluate_binclf_v2.py
python src/v2/tune_threshold_binclf_v2.py
```

smoke 산출물(model/·test_results/의 `smoke` 이름)은 커밋 금지. 이 저장소 로컬 PC에서는
OpenBLAS 메모리 오류가 나면 `$env:OPENBLAS_NUM_THREADS="1"`을 추가할 것.

## 7. 자주 나는 문제

| 증상 | 원인/해결 |
|---|---|
| CUDA OOM | `BATCH_SIZE=32` (448 입력은 224의 약 4배 메모리) |
| `metadata_split_v2.csv 없음` | DATA_DIR 확인. 워크트리에서 실행 시 반드시 `DATA_DIR=D:\hn_old-building` 지정 |
| `이미지 캐시 폴더 없음` | processed_images_v2 14GB 반입 확인 |
| cv2 import 오류 | `pip install opencv-python-headless` (USE_CLAHE=1일 때만 필요) |
| baseline .pth 없음 | train을 같은 ARCH/RUN_NAME으로 먼저 실행 |
| 학습이 기존보다 느림 | 정상 — 448 입력은 224 대비 연산 3~4배. AMP(기본 켜짐)로 상쇄 |
| 평가 결과에 [경고] 전처리 불일치 | 학습 때와 USE_CLAHE/EVAL_RESIZE_MODE가 다름 — 의도한 ablation이 아니면 env를 맞출 것 |

## 8. 결과 해석 주의

- 이원화 test는 불량 91.2% — accuracy 바닥선이 91%이므로 **우수 F1과 macro F1**로 판단.
- midclf는 분포가 비교적 균형이지만 `train_history`의 `val_f1_최저클래스`로 최약 클래스를 추적할 것.
- 기존 binclf(224, 정방형 왜곡) 결과와 비교할 때는 "전처리+해상도가 모두 다르다"는 점을 명시.
