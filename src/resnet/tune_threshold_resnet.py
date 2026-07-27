# ============================================================
# [사용법]
# 결정 임계값(threshold) 보정 스크립트 — 재학습 없이 기존 체크포인트만 사용.
#
# 이미 학습된 ResNet 체크포인트를 읽어 validation 셋에서 "보통(1)" logit에
# 더할 보정치(bias)를 탐색하고, 확정된 bias 하나로 test를 1회 평가합니다.
# bias 탐색은 validation에서만 하므로 test 정보 누출(leakage)이 없습니다.
#
# 기존 코드/체크포인트/평가 결과는 일절 수정하지 않습니다.
# 결과는 test_results/<RESNET_ARCH>/<RUN_NAME>_threshold/ 아래에만 새로 저장됩니다.
#
# 실행 (런팟 = 리눅스):
#   RESNET_ARCH=resnet18 RUN_NAME=r18_e20 python src/resnet/tune_threshold_resnet.py
#
# 실행 (윈도우 PowerShell):
#   $env:RESNET_ARCH="resnet18"; $env:RUN_NAME="r18_e20"
#   python src/resnet/tune_threshold_resnet.py
#
# [환경변수]
#   RESNET_ARCH  : resnet18 (기본) 또는 resnet50
#   RUN_NAME     : 학습 때 쓴 실험 이름 (기본 "default")
#   EVAL_TARGET  : baseline 또는 finetuned (기본 "finetuned")
#   RECALL_FLOOR : 불량 recall 하한 (기본 0.85).
#                  "불량 recall이 이 값 이상"인 bias 중에서 보통 F1 최대를 선택
#   DATA_DIR     : data/ 와 model/ 이 있는 프로젝트 루트 (기본: 이 저장소).
#                  워크트리처럼 데이터가 없는 곳에서 돌릴 때 원본 저장소를 지정
#   BATCH_SIZE   : 배치 크기 (기본 64 — CPU에서도 무난한 값)
#   NUM_WORKERS  : 데이터 로딩 워커 수 (기본 4)
#
# GPU가 없어도 됩니다. 학습이 아니라 추론 2회(validation 1회 + test 1회)뿐이라
# CPU로도 동작하며, bias 탐색은 저장해 둔 logit에 대한 배열 연산이라 수 초면 끝납니다.
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

# 전처리는 기존 학습/평가와 완전히 동일해야 하므로 팀 공용 정의를 그대로 가져옴
from preprocess_resnet import RAW_PATH_MARKER, evaluation_transform

# 이 파일이 src/resnet/ 안에 있으므로 세 단계 올라가면 프로젝트 폴더
project_dir = Path(__file__).resolve().parent.parent.parent

RESNET_ARCH = os.environ.get("RESNET_ARCH", "resnet18")
RUN_NAME = os.environ.get("RUN_NAME", "default")
EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")

# 불량 recall 하한: 이 값 밑으로 떨어뜨리는 bias는 후보에서 제외
RECALL_FLOOR = float(os.environ.get("RECALL_FLOOR", 0.85))

# 데이터/모델 위치: 기본은 이 저장소, 워크트리에서 돌릴 때는 원본 저장소를 지정
data_dir = Path(os.environ.get("DATA_DIR", project_dir))

batch_size = int(os.environ.get("BATCH_SIZE", 64))
num_workers = int(os.environ.get("NUM_WORKERS", 4))

metadata_path = data_dir / "data" / "processed" / "metadata_split.csv"
processed_images_dir = data_dir / "data" / "processed_images"

best_model_path = (
    data_dir
    / "model"
    / f"best_{RESNET_ARCH}_{EVALUATION_TARGET}_{RUN_NAME}.pth"
)

# 결과는 기존 평가 결과와 다른 폴더에 저장 (덮어쓰기 없음)
results_dir = (
    project_dir / "test_results" / RESNET_ARCH / f"{RUN_NAME}_threshold"
)

class_names = ["우수", "보통", "불량"]

# 탐색할 bias 후보: 0.0(= 기존 argmax 그대로) ~ 3.0을 0.05 간격으로
bias_candidates = [round(step * 0.05, 2) for step in range(61)]


class ThresholdDataset(Dataset):
    """metadata_split.csv 한 split의 이미지와 정답을 반환하는 데이터셋.

    preprocess_resnet.BuildingDataset과 동일한 동작이지만, 이미지 폴더 위치를
    모듈 전역이 아니라 생성자 인자로 받는다. (DATA_DIR로 원본 저장소를
    지정해도 동작해야 하므로 — 기존 코드는 수정하지 않는다.)
    """

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


def create_resnet_model():
    """저장된 체크포인트를 덮어씌울 빈 ResNet 구조를 만든다 (evaluate_resnet.py와 동일)."""
    if RESNET_ARCH == "resnet18":
        model = resnet18(weights=None)
    elif RESNET_ARCH == "resnet50":
        model = resnet50(weights=None)
    else:
        raise ValueError(
            f"지원하지 않는 RESNET_ARCH: {RESNET_ARCH} (resnet18/resnet50만 가능)"
        )

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 3)

    return model


def collect_logits(model, loader, device):
    """한 split 전체를 추론해 (logits, labels) 텐서를 반환한다. 모델 수정 없음."""
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            all_logits.append(outputs.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def metrics_with_bias(logits, labels, bias):
    """보통(1) logit에 bias를 더했을 때의 주요 지표를 계산한다."""
    shifted = logits.clone()
    shifted[:, 1] += bias
    predictions = shifted.argmax(dim=1).numpy()

    labels_numpy = labels.numpy()

    return {
        "bias": bias,
        "predictions": predictions,
        "accuracy": (predictions == labels_numpy).mean(),
        "average_f1": f1_score(
            labels_numpy, predictions, labels=[1], average=None,
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
    print("평가 모델 :", best_model_path.name)
    print("데이터 위치 :", data_dir)
    print("불량 recall 하한 :", RECALL_FLOOR)

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {best_model_path}\n"
            "RESNET_ARCH / RUN_NAME / EVAL_TARGET / DATA_DIR 값을 확인하세요."
        )

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")

    validation_data = metadata[
        metadata["split"] == "validation"
    ].reset_index(drop=True)

    test_data = metadata[
        metadata["split"] == "test"
    ].reset_index(drop=True)

    print("Validation 장수 :", len(validation_data))
    print("Test 장수 :", len(test_data))

    validation_loader = DataLoader(
        ThresholdDataset(
            validation_data, processed_images_dir, evaluation_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        ThresholdDataset(
            test_data, processed_images_dir, evaluation_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = create_resnet_model()

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ---------- 1단계: validation 1회 추론 후 bias 탐색 ----------
    print("\nValidation 추론 중 (1회만 실행됨)...")
    validation_logits, validation_labels = collect_logits(
        model, validation_loader, device
    )

    sweep_rows = []
    for bias in bias_candidates:
        result = metrics_with_bias(
            validation_logits, validation_labels, bias
        )
        sweep_rows.append({
            "bias": result["bias"],
            "보통_F1": round(result["average_f1"], 4),
            "불량_recall": round(result["defect_recall"], 4),
            "불량_F1": round(result["defect_f1"], 4),
            "macro_F1": round(result["macro_f1"], 4),
            "accuracy": round(result["accuracy"], 4),
        })

    sweep_table = pd.DataFrame(sweep_rows)

    # 불량 recall 하한을 지키는 후보 중 보통 F1이 최대인 bias 선택
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
    print("\nTest 추론 중 (1회만 실행됨)...")
    test_logits, test_labels = collect_logits(model, test_loader, device)

    baseline = metrics_with_bias(test_logits, test_labels, 0.0)
    adjusted = metrics_with_bias(test_logits, test_labels, chosen_bias)

    labels_numpy = test_labels.numpy()

    baseline_report = classification_report(
        labels_numpy, baseline["predictions"],
        labels=[0, 1, 2], target_names=class_names,
        digits=4, zero_division=0
    )

    adjusted_report = classification_report(
        labels_numpy, adjusted["predictions"],
        labels=[0, 1, 2], target_names=class_names,
        digits=4, zero_division=0
    )

    matrix_index = ["실제_우수", "실제_보통", "실제_불량"]
    matrix_columns = ["예측_우수", "예측_보통", "예측_불량"]

    baseline_matrix = pd.DataFrame(
        confusion_matrix(
            labels_numpy, baseline["predictions"], labels=[0, 1, 2]
        ),
        index=matrix_index,
        columns=matrix_columns
    )

    adjusted_matrix = pd.DataFrame(
        confusion_matrix(
            labels_numpy, adjusted["predictions"], labels=[0, 1, 2]
        ),
        index=matrix_index,
        columns=matrix_columns
    )

    # ---------- 결과 저장 (기존 결과와 다른 새 폴더) ----------
    results_dir.mkdir(parents=True, exist_ok=True)

    sweep_table.to_csv(
        results_dir / "validation_bias_sweep.csv",
        index=False,
        encoding="utf-8-sig"
    )

    baseline_matrix.to_csv(
        results_dir / "test_confusion_matrix_argmax.csv",
        encoding="utf-8-sig"
    )

    adjusted_matrix.to_csv(
        results_dir / "test_confusion_matrix_threshold.csv",
        encoding="utf-8-sig"
    )

    lines = []
    record = lines.append

    record("==========================================")
    record("결정 임계값(threshold) 보정 실험 결과")
    record("==========================================")
    record(f"모델 파일 : {best_model_path.name}")
    record(f"모델 종류 : {RESNET_ARCH} ({EVALUATION_TARGET})")
    record(f"실험 이름 : {RUN_NAME}")
    record(f"사용 장치 : {device}")
    record("")
    record("[bias 결정 절차]")
    record("bias 탐색은 validation에서만 수행 (test 누출 없음)")
    record(f"탐색 범위 : 0.00 ~ 3.00 (0.05 간격, {len(bias_candidates)}개)")
    record(f"제약 조건 : 불량 recall >= {RECALL_FLOOR}")
    record(f"선택된 bias : {chosen_bias} (보통 logit에 더함)")
    record(f"validation 보통 F1 : {best_row['보통_F1']:.4f}")
    record(f"validation 불량 recall : {best_row['불량_recall']:.4f}")
    record("")
    record("===== Test 결과 비교 (같은 체크포인트, 결정 규칙만 다름) =====")
    record("")
    record(f"{'지표':<14}{'기존 argmax':>12}{'threshold 보정':>16}")
    record(
        f"{'보통 F1':<14}"
        f"{baseline['average_f1']:>12.4f}"
        f"{adjusted['average_f1']:>16.4f}"
    )
    record(
        f"{'불량 recall':<14}"
        f"{baseline['defect_recall']:>12.4f}"
        f"{adjusted['defect_recall']:>16.4f}"
    )
    record(
        f"{'불량 F1':<14}"
        f"{baseline['defect_f1']:>12.4f}"
        f"{adjusted['defect_f1']:>16.4f}"
    )
    record(
        f"{'Macro F1':<14}"
        f"{baseline['macro_f1']:>12.4f}"
        f"{adjusted['macro_f1']:>16.4f}"
    )
    record(
        f"{'Accuracy':<14}"
        f"{baseline['accuracy']:>12.4f}"
        f"{adjusted['accuracy']:>16.4f}"
    )
    record("")
    record("===== 기존 argmax 등급별 상세 =====")
    record(baseline_report)
    record("")
    record("===== threshold 보정 등급별 상세 =====")
    record(adjusted_report)
    record("")
    record("===== 기존 argmax 혼동행렬 =====")
    record(baseline_matrix.to_string())
    record("")
    record("===== threshold 보정 혼동행렬 =====")
    record(adjusted_matrix.to_string())

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path = results_dir / "threshold_tuning_result.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n결과 저장 위치 :", results_dir)


if __name__ == "__main__":
    main()
