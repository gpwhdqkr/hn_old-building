# ============================================================
# [사용법]
# 2단계 캐스케이드 Test 평가 스크립트.
#
# train_cascade.py로 학습한 두 체크포인트를 이어 붙여 test를 1회 평가한다:
#   1단계 모델이 "불량"으로 판정 → 최종 불량
#   1단계 모델이 "정상"으로 판정 → 2단계 모델이 우수/보통을 판정
#
# 실행 (런팟 = 리눅스):
#   RESNET_ARCH=resnet18 RUN_NAME=cas1 python src/resnet/evaluate_cascade.py
#
# [환경변수]
#   RESNET_ARCH : resnet18 (기본) 또는 resnet50
#   RUN_NAME    : 학습 때 쓴 실험 이름 (기본 "default")
#   BATCH_SIZE  : 배치 크기 (기본 128)
#   DATA_DIR    : data/ 와 model/ 이 있는 프로젝트 루트 (기본: 이 저장소)
#
# 결과 (기존 평가 결과와 다른 새 폴더 — 덮어쓰기 없음):
#   test_results/<RESNET_ARCH>/cascade_<RUN_NAME>/cascade_result.txt
#   test_results/<RESNET_ARCH>/cascade_<RUN_NAME>/test_confusion_matrix_cascade.csv
#
# 출력 내용:
#   - 최종 3-클래스 성능 (기존 evaluate_resnet.py와 같은 형식이라 나란히 비교 가능)
#   - 1단계 단독 성능 (불량 vs 정상 이진)
#   - 2단계 단독 성능 (실제 우수/보통 이미지만 대상)
#   → 최종 성능이 낮으면 어느 단계가 병목인지 바로 확인할 수 있음
# ============================================================

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet50

# 전처리는 학습 때와 완전히 동일해야 하므로 팀 공용 정의를 그대로 가져옴
from preprocess_resnet import RAW_PATH_MARKER, evaluation_transform, num_workers

project_dir = Path(__file__).resolve().parent.parent.parent

RESNET_ARCH = os.environ.get("RESNET_ARCH", "resnet18")
RUN_NAME = os.environ.get("RUN_NAME", "default")

batch_size = int(os.environ.get("BATCH_SIZE", 128))

data_dir = Path(os.environ.get("DATA_DIR", project_dir))

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

# 결과는 기존 평가 결과와 다른 폴더에 저장 (덮어쓰기 없음)
results_dir = (
    project_dir / "test_results" / RESNET_ARCH / f"cascade_{RUN_NAME}"
)

class_names = ["우수", "보통", "불량"]


class EvaluationDataset(Dataset):
    """test split의 이미지와 원본 3-클래스 라벨을 반환하는 데이터셋."""

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

    return model, checkpoint


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("사용 장치 :", device)
    print("1단계 모델 :", stage1_model_path.name)
    print("2단계 모델 :", stage2_model_path.name)

    stage1_model, stage1_checkpoint = load_cascade_model(
        stage1_model_path, device
    )
    stage2_model, stage2_checkpoint = load_cascade_model(
        stage2_model_path, device
    )

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")

    test_data = metadata[
        metadata["split"] == "test"
    ].reset_index(drop=True)

    print("Test 장수 :", len(test_data))

    test_loader = DataLoader(
        EvaluationDataset(
            test_data, processed_images_dir, evaluation_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_labels = []
    stage1_predictions = []   # 0=정상(우수+보통), 1=불량
    final_predictions = []    # 최종 3-클래스 (0=우수, 1=보통, 2=불량)

    # 두 모델에 같은 배치를 통과시키고 1단계 판정에 따라 결과를 합성한다.
    # (1단계가 정상으로 본 이미지만 2단계 결과를 쓰면 되므로, 배치 전체를
    #  두 모델 모두에 넣고 나중에 골라 쓰는 쪽이 GPU에서는 더 단순하고 빠름)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)

            stage1_output = stage1_model(images).argmax(dim=1)
            stage2_output = stage2_model(images).argmax(dim=1)

            # 1단계가 불량(1)이라 하면 최종 불량(2),
            # 정상(0)이라 하면 2단계 판정(0=우수, 1=보통)을 그대로 사용
            combined = torch.where(
                stage1_output == 1,
                torch.full_like(stage1_output, 2),
                stage2_output
            )

            all_labels.extend(labels.tolist())
            stage1_predictions.extend(stage1_output.cpu().tolist())
            final_predictions.extend(combined.cpu().tolist())

    # ---------- 최종 3-클래스 성능 ----------
    labels_series = pd.Series(all_labels)
    final_series = pd.Series(final_predictions)

    accuracy = (labels_series == final_series).mean()

    final_report = classification_report(
        all_labels, final_predictions,
        labels=[0, 1, 2], target_names=class_names,
        digits=4, zero_division=0
    )

    final_matrix = pd.DataFrame(
        confusion_matrix(all_labels, final_predictions, labels=[0, 1, 2]),
        index=["실제_우수", "실제_보통", "실제_불량"],
        columns=["예측_우수", "예측_보통", "예측_불량"]
    )

    per_class_f1 = f1_score(
        all_labels, final_predictions,
        labels=[0, 1, 2], average=None, zero_division=0
    )
    macro_f1 = f1_score(
        all_labels, final_predictions, average="macro", zero_division=0
    )

    # ---------- 1단계 단독 성능 (불량 vs 정상) ----------
    stage1_true = [1 if label == 2 else 0 for label in all_labels]

    stage1_report = classification_report(
        stage1_true, stage1_predictions,
        labels=[0, 1], target_names=["정상(우수+보통)", "불량"],
        digits=4, zero_division=0
    )

    # ---------- 실제 우수/보통 이미지에 대한 최종 예측 분석 ----------
    # 최종 예측이 불량(2)인 경우는 1단계에서 잘못 걸러진 것 → 캐스케이드 손실,
    # 우수/보통 간 오류는 2단계 모델의 구분 실패로 해석할 수 있음
    stage2_true = [
        label
        for label in all_labels
        if label in (0, 1)
    ]

    stage2_final_predictions = [
        final_prediction
        for label, final_prediction in zip(all_labels, final_predictions)
        if label in (0, 1)
    ]

    stage2_report = classification_report(
        stage2_true, stage2_final_predictions,
        labels=[0, 1, 2], target_names=class_names,
        digits=4, zero_division=0
    )

    # ---------- 결과 저장 ----------
    results_dir.mkdir(parents=True, exist_ok=True)

    final_matrix.to_csv(
        results_dir / "test_confusion_matrix_cascade.csv",
        encoding="utf-8-sig"
    )

    lines = []
    record = lines.append

    record("==========================================")
    record("2단계 캐스케이드 Test 평가 결과")
    record("==========================================")
    record(f"모델 종류 : {RESNET_ARCH}")
    record(f"실험 이름 : {RUN_NAME}")
    record(f"사용 장치 : {device}")
    record(f"1단계 모델 : {stage1_model_path.name}")
    record(f"  - 저장 시 validation macro F1 : "
           f"{stage1_checkpoint.get('validation_macro_f1', '정보없음')}")
    record(f"2단계 모델 : {stage2_model_path.name}")
    record(f"  - 저장 시 validation macro F1 : "
           f"{stage2_checkpoint.get('validation_macro_f1', '정보없음')}")
    record("")
    record("===== 최종 3-클래스 성능 =====")
    record(f"Test 장수 : {len(all_labels)}")
    record(f"Test 정확도 : {accuracy:.2%}")
    record(f"우수 F1 : {per_class_f1[0]:.4f}")
    record(f"보통 F1 : {per_class_f1[1]:.4f}")
    record(f"불량 F1 : {per_class_f1[2]:.4f}")
    record(f"Macro F1 : {macro_f1:.4f}")
    record("")
    record(final_report)
    record("")
    record("===== 혼동행렬 =====")
    record(final_matrix.to_string())
    record("")
    record("===== 1단계 단독 성능 (불량 vs 정상 이진) =====")
    record(stage1_report)
    record("")
    record("===== 실제 우수/보통 이미지에 대한 최종 예측 =====")
    record("(예측_불량 행은 1단계에서 불량으로 잘못 걸러진 손실을 의미)")
    record(stage2_report)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path = results_dir / "cascade_result.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n결과 저장 위치 :", results_dir)


if __name__ == "__main__":
    main()
