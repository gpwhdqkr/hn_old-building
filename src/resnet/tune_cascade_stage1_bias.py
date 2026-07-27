# ============================================================
# [사용법]
# 캐스케이드 1단계의 불량 판정 문턱(STAGE1_BIAS)을 validation에서 탐색하는 스크립트.
#
# 배경: 캐스케이드 평가 결과 최종 보통 F1의 병목은 1단계였음.
# 1단계가 실제 보통 620장 중 265장(42.7%)을 불량으로 걸러내 2단계가 볼 기회가 없음.
# 1단계의 정상(0) logit에 bias를 더해 불량 판정을 보수적으로 만들면
# 경계의 보통이 2단계로 넘어가 살아날 수 있다. 반대급부(불량이 2단계로 새어
# 들어가 보통 precision 하락)와의 최적점을 validation에서만 찾는다.
#
# 절차 (tune_threshold_resnet.py와 동일한 원칙 — test 정보 누출 없음):
#   1. validation 전체를 두 모델에 1회씩 통과시켜 1단계 logit과 2단계 예측을 저장
#   2. 저장된 값으로 bias 0.00~3.00을 훑으며 최종 3-클래스 지표 계산 (재추론 없음)
#   3. "불량 recall >= RECALL_FLOOR" 제약 안에서 보통 F1 최대인 bias 확정
#   4. 확정된 bias로 test를 1회 평가하고 보정 전(bias=0)과 나란히 기록
#
# 실행 (런팟 = 리눅스):
#   RESNET_ARCH=resnet18 RUN_NAME=cas1 python src/resnet/tune_cascade_stage1_bias.py
#
# 실행 (윈도우 로컬, 데이터가 원본 저장소에 있을 때):
#   $env:RESNET_ARCH="resnet18"; $env:RUN_NAME="cas1"; $env:DATA_DIR="D:/hn_old-building"
#   python src/resnet/tune_cascade_stage1_bias.py
#
# [환경변수]
#   RESNET_ARCH  : resnet18 (기본) 또는 resnet50
#   RUN_NAME     : train_cascade.py 학습 때 쓴 실험 이름 (기본 "default")
#   RECALL_FLOOR : 불량 recall 하한 (기본 0.85)
#   DATA_DIR     : data/ 와 model/ 이 있는 프로젝트 루트 (기본: 이 저장소)
#   BATCH_SIZE   : 배치 크기 (기본 64)
#   NUM_WORKERS  : 데이터 로딩 워커 수 (기본 4)
#
# 결과 (기존 결과와 다른 새 폴더 — 덮어쓰기 없음):
#   test_results/<RESNET_ARCH>/cascade_<RUN_NAME>_stage1bias/
#     stage1_bias_tuning_result.txt
#     validation_bias_sweep.csv
#     test_confusion_matrix_bias0.csv      (보정 전)
#     test_confusion_matrix_tuned.csv      (보정 후)
#
# 확정된 bias는 evaluate_cascade.py의 STAGE1_BIAS 환경변수로 재현할 수 있다.
# GPU 불필요 — 추론만 하므로 CPU로도 동작한다.
# ============================================================

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet50

# 전처리는 학습 때와 완전히 동일해야 하므로 팀 공용 정의를 그대로 가져옴
from preprocess_resnet import RAW_PATH_MARKER, evaluation_transform

project_dir = Path(__file__).resolve().parent.parent.parent

RESNET_ARCH = os.environ.get("RESNET_ARCH", "resnet18")
RUN_NAME = os.environ.get("RUN_NAME", "default")

RECALL_FLOOR = float(os.environ.get("RECALL_FLOOR", 0.85))

data_dir = Path(os.environ.get("DATA_DIR", project_dir))

batch_size = int(os.environ.get("BATCH_SIZE", 64))
num_workers = int(os.environ.get("NUM_WORKERS", 4))

metadata_path = data_dir / "data" / "processed" / "metadata_split.csv"
processed_images_dir = data_dir / "data" / "processed_images"

stage1_model_path = (
    data_dir / "model"
    / f"best_{RESNET_ARCH}_cascade_stage1_{RUN_NAME}.pth"
)
stage2_model_path = (
    data_dir / "model"
    / f"best_{RESNET_ARCH}_cascade_stage2_{RUN_NAME}.pth"
)

results_dir = (
    project_dir / "test_results" / RESNET_ARCH
    / f"cascade_{RUN_NAME}_stage1bias"
)

class_names = ["우수", "보통", "불량"]

# 탐색할 bias 후보: 0.0(= 보정 없음) ~ 3.0을 0.05 간격으로
bias_candidates = [round(step * 0.05, 2) for step in range(61)]


class EvaluationDataset(Dataset):
    """지정된 split의 이미지와 원본 3-클래스 라벨을 반환하는 데이터셋."""

    def __init__(self, dataframe, images_dir, transform):
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        # CSV의 팀원 PC 경로에서 raw/images/ 뒤 상대 경로만 잘라 실제 위치로 변환
        posix_path = str(row["image_path"]).replace("\\", "/")
        marker_index = posix_path.find(RAW_PATH_MARKER)

        if marker_index == -1:
            raise ValueError(
                f"이미지 경로에서 '{RAW_PATH_MARKER}'를 찾을 수 없습니다: "
                f"{row['image_path']}"
            )

        relative_path = posix_path[marker_index + len(RAW_PATH_MARKER):]
        image_path = self.images_dir / relative_path

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image = self.transform(image)
        label = int(row["model_label"])

        return image, label


def load_cascade_model(model_path, device):
    """이진 분류(출력 2개) ResNet 구조를 만들어 체크포인트를 덮어씌운다."""
    if RESNET_ARCH == "resnet18":
        model = resnet18(weights=None)
    elif RESNET_ARCH == "resnet50":
        model = resnet50(weights=None)
    else:
        raise ValueError(
            f"지원하지 않는 RESNET_ARCH: {RESNET_ARCH} (resnet18/resnet50만 가능)"
        )

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 2)

    if not model_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {model_path}\n"
            "train_cascade.py를 같은 RESNET_ARCH / RUN_NAME으로 "
            "STAGE=1, STAGE=2 각각 먼저 실행하세요."
        )

    checkpoint = torch.load(
        model_path, map_location=device, weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def collect_cascade_outputs(stage1_model, stage2_model, loader, device):
    """한 split 전체를 두 모델에 통과시켜
    (1단계 logit, 2단계 예측, 정답) 텐서를 반환한다. 추론은 여기서 1회뿐."""
    stage1_logits_list = []
    stage2_predictions_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            stage1_logits_list.append(stage1_model(images).cpu())
            stage2_predictions_list.append(
                stage2_model(images).argmax(dim=1).cpu()
            )
            labels_list.append(labels)

    return (
        torch.cat(stage1_logits_list),
        torch.cat(stage2_predictions_list),
        torch.cat(labels_list)
    )


def combine_with_bias(stage1_logits, stage2_predictions, bias):
    """bias를 적용한 1단계 판정과 2단계 예측을 최종 3-클래스 예측으로 합성한다."""
    shifted = stage1_logits.clone()
    shifted[:, 0] += bias
    stage1_predictions = shifted.argmax(dim=1)

    # 1단계가 불량(1)이면 최종 불량(2), 정상(0)이면 2단계 판정(0=우수, 1=보통)
    return torch.where(
        stage1_predictions == 1,
        torch.full_like(stage1_predictions, 2),
        stage2_predictions
    ).numpy()


def metrics_for(labels_numpy, predictions):
    """최종 3-클래스 예측의 주요 지표를 계산한다."""
    return {
        "accuracy": (predictions == labels_numpy).mean(),
        "average_f1": f1_score(
            labels_numpy, predictions, labels=[1], average=None,
            zero_division=0
        )[0],
        "excellent_f1": f1_score(
            labels_numpy, predictions, labels=[0], average=None,
            zero_division=0
        )[0],
        "defect_recall": recall_score(
            labels_numpy, predictions, labels=[2], average=None,
            zero_division=0
        )[0],
        "defect_f1": f1_score(
            labels_numpy, predictions, labels=[2], average=None,
            zero_division=0
        )[0],
        "macro_f1": f1_score(
            labels_numpy, predictions, average="macro", zero_division=0
        ),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("사용 장치 :", device)
    print("1단계 모델 :", stage1_model_path.name)
    print("2단계 모델 :", stage2_model_path.name)
    print("데이터 위치 :", data_dir)
    print("불량 recall 하한 :", RECALL_FLOOR)

    stage1_model = load_cascade_model(stage1_model_path, device)
    stage2_model = load_cascade_model(stage2_model_path, device)

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")

    validation_data = metadata[
        metadata["split"] == "validation"
    ].reset_index(drop=True)

    test_data = metadata[
        metadata["split"] == "test"
    ].reset_index(drop=True)

    print("Validation 장수 :", len(validation_data))
    print("Test 장수 :", len(test_data))

    def build_loader(dataframe):
        return DataLoader(
            EvaluationDataset(
                dataframe, processed_images_dir, evaluation_transform
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # ---------- 1단계: validation 1회 추론 후 bias 탐색 ----------
    print("\nValidation 추론 중 (모델 2개 × 1회)...")
    validation_stage1_logits, validation_stage2_predictions, validation_labels = (
        collect_cascade_outputs(
            stage1_model, stage2_model, build_loader(validation_data), device
        )
    )

    validation_labels_numpy = validation_labels.numpy()

    sweep_rows = []
    for bias in bias_candidates:
        predictions = combine_with_bias(
            validation_stage1_logits, validation_stage2_predictions, bias
        )
        result = metrics_for(validation_labels_numpy, predictions)
        sweep_rows.append({
            "bias": bias,
            "보통_F1": round(result["average_f1"], 4),
            "우수_F1": round(result["excellent_f1"], 4),
            "불량_recall": round(result["defect_recall"], 4),
            "불량_F1": round(result["defect_f1"], 4),
            "macro_F1": round(result["macro_f1"], 4),
            "accuracy": round(result["accuracy"], 4),
        })

    sweep_table = pd.DataFrame(sweep_rows)

    eligible = sweep_table[sweep_table["불량_recall"] >= RECALL_FLOOR]

    if eligible.empty:
        raise RuntimeError(
            f"불량 recall {RECALL_FLOOR} 이상을 만족하는 bias가 없습니다. "
            "RECALL_FLOOR를 낮춰서 다시 실행하세요."
        )

    best_row = eligible.loc[eligible["보통_F1"].idxmax()]
    chosen_bias = float(best_row["bias"])

    print("\n[validation 탐색 결과]")
    print(f"선택된 bias : {chosen_bias}")
    print(f"validation 보통 F1 : {best_row['보통_F1']:.4f}")
    print(f"validation 불량 recall : {best_row['불량_recall']:.4f}")

    # ---------- 2단계: 확정된 bias로 test 1회 평가 ----------
    print("\nTest 추론 중 (모델 2개 × 1회)...")
    test_stage1_logits, test_stage2_predictions, test_labels = (
        collect_cascade_outputs(
            stage1_model, stage2_model, build_loader(test_data), device
        )
    )

    test_labels_numpy = test_labels.numpy()

    baseline_predictions = combine_with_bias(
        test_stage1_logits, test_stage2_predictions, 0.0
    )
    tuned_predictions = combine_with_bias(
        test_stage1_logits, test_stage2_predictions, chosen_bias
    )

    baseline = metrics_for(test_labels_numpy, baseline_predictions)
    tuned = metrics_for(test_labels_numpy, tuned_predictions)

    baseline_report = classification_report(
        test_labels_numpy, baseline_predictions,
        labels=[0, 1, 2], target_names=class_names,
        digits=4, zero_division=0
    )
    tuned_report = classification_report(
        test_labels_numpy, tuned_predictions,
        labels=[0, 1, 2], target_names=class_names,
        digits=4, zero_division=0
    )

    matrix_index = ["실제_우수", "실제_보통", "실제_불량"]
    matrix_columns = ["예측_우수", "예측_보통", "예측_불량"]

    baseline_matrix = pd.DataFrame(
        confusion_matrix(
            test_labels_numpy, baseline_predictions, labels=[0, 1, 2]
        ),
        index=matrix_index, columns=matrix_columns
    )
    tuned_matrix = pd.DataFrame(
        confusion_matrix(
            test_labels_numpy, tuned_predictions, labels=[0, 1, 2]
        ),
        index=matrix_index, columns=matrix_columns
    )

    # ---------- 결과 저장 ----------
    results_dir.mkdir(parents=True, exist_ok=True)

    sweep_table.to_csv(
        results_dir / "validation_bias_sweep.csv",
        index=False, encoding="utf-8-sig"
    )
    baseline_matrix.to_csv(
        results_dir / "test_confusion_matrix_bias0.csv",
        encoding="utf-8-sig"
    )
    tuned_matrix.to_csv(
        results_dir / "test_confusion_matrix_tuned.csv",
        encoding="utf-8-sig"
    )

    lines = []
    record = lines.append

    record("==========================================")
    record("캐스케이드 1단계 bias 보정 실험 결과")
    record("==========================================")
    record(f"모델 종류 : {RESNET_ARCH}")
    record(f"실험 이름 : {RUN_NAME}")
    record(f"사용 장치 : {device}")
    record(f"1단계 모델 : {stage1_model_path.name}")
    record(f"2단계 모델 : {stage2_model_path.name}")
    record("")
    record("[bias 결정 절차]")
    record("bias 탐색은 validation에서만 수행 (test 누출 없음)")
    record(f"탐색 범위 : 0.00 ~ 3.00 (0.05 간격, {len(bias_candidates)}개)")
    record(f"제약 조건 : 불량 recall >= {RECALL_FLOOR}")
    record(f"선택된 bias : {chosen_bias} (1단계 정상 logit에 더함)")
    record(f"validation 보통 F1 : {best_row['보통_F1']:.4f}")
    record(f"validation 불량 recall : {best_row['불량_recall']:.4f}")
    record("")
    record("재현 방법 :")
    record(f"  STAGE1_BIAS={chosen_bias} RESNET_ARCH={RESNET_ARCH} "
           f"RUN_NAME={RUN_NAME} python src/resnet/evaluate_cascade.py")
    record("")
    record("===== Test 결과 비교 (같은 체크포인트, 1단계 문턱만 다름) =====")
    record("")
    record(f"{'지표':<14}{'보정 전(bias=0)':>16}{'보정 후':>12}")
    record(
        f"{'보통 F1':<14}"
        f"{baseline['average_f1']:>16.4f}{tuned['average_f1']:>12.4f}"
    )
    record(
        f"{'우수 F1':<14}"
        f"{baseline['excellent_f1']:>16.4f}{tuned['excellent_f1']:>12.4f}"
    )
    record(
        f"{'불량 recall':<14}"
        f"{baseline['defect_recall']:>16.4f}{tuned['defect_recall']:>12.4f}"
    )
    record(
        f"{'불량 F1':<14}"
        f"{baseline['defect_f1']:>16.4f}{tuned['defect_f1']:>12.4f}"
    )
    record(
        f"{'Macro F1':<14}"
        f"{baseline['macro_f1']:>16.4f}{tuned['macro_f1']:>12.4f}"
    )
    record(
        f"{'Accuracy':<14}"
        f"{baseline['accuracy']:>16.4f}{tuned['accuracy']:>12.4f}"
    )
    record("")
    record("===== 보정 전 등급별 상세 =====")
    record(baseline_report)
    record("")
    record("===== 보정 후 등급별 상세 =====")
    record(tuned_report)
    record("")
    record("===== 보정 전 혼동행렬 =====")
    record(baseline_matrix.to_string())
    record("")
    record("===== 보정 후 혼동행렬 =====")
    record(tuned_matrix.to_string())

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path = results_dir / "stage1_bias_tuning_result.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n결과 저장 위치 :", results_dir)


if __name__ == "__main__":
    main()
