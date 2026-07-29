# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# v3 이원화 임계값 튜닝 — 5-crop TTA 서빙 규격판.
#
# 서빙(app/ai_engine.py)이 중앙 1크롭 → 5크롭(네 모서리+중앙) "최대 불량 확률"
# 판정으로 바뀌면서 점수 분포가 위로 이동했다 (max >= 중앙값이 항상 성립).
# 기존 tune_threshold_binclf_v3.py의 임계값은 중앙 크롭 분포 기준이므로,
# 이 스크립트로 validation 점수를 5-crop max로 다시 뽑아 임계값을 재선정한다.
# 선택 규칙은 기존과 동일: 불량 recall >= RECALL_FLOOR 필터 → macro F1 최대
# → 동률 시 우수 F1 최대. (기존 스크립트는 수정하지 않음 — 비교용으로 보존)
#
# 실행 (런팟 = 리눅스):
#   ARCH=convnext_tiny RUN_NAME=v3cta python src/v3/tune_threshold_binclf_v3_5crop.py
# (PowerShell):
#   $env:ARCH="convnext_tiny"; $env:RUN_NAME="v3cta"; python src/v3/tune_threshold_binclf_v3_5crop.py
#
# [환경변수] ARCH RUN_NAME EVAL_TARGET(finetuned) RECALL_FLOOR(0.85) BATCH_SIZE — README 참고
# 주의: 이미지당 5크롭이라 실효 배치가 5배 — OOM 시 BATCH_SIZE를 1/5로 낮출 것.
#
# 결과 (test_results/binclf_v3/<ARCH>/<RUN_NAME>_threshold_5crop/):
#   validation_threshold_sweep.csv
#   test_confusion_matrix_default05.csv / test_confusion_matrix_threshold.csv
#   predictions_val_<EVAL_TARGET>.csv   — 샘플별 5-crop max 확률 + 중앙 크롭 확률
#   threshold_tuning_result.txt
# ============================================================

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import functional as TF
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from model_factory_v3 import create_model
from pipeline_v3 import ARCH, RUN_NAME, checkpoint_path, get_device, synchronize
from preprocess_v3 import (
    BIN_CLASS_NAMES,
    EVAL_RESIZE_SHORT,
    INPUT_SIZE,
    _eval_post_transform,
    apply_clahe,
    batch_size,
    load_split_dataframes,
    num_workers,
    processed_images_dir,
    set_seed,
    test_results_root,
    use_clahe,
)

EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")
RECALL_FLOOR = float(os.environ.get("RECALL_FLOOR", 0.85))

# 탐색 범위: 0.02 ~ 0.98, 0.01 간격 (기존 binclf와 동일)
THRESHOLDS = [round(0.02 + 0.01 * index, 2) for index in range(97)]

# 이미지당 5크롭이므로 실효 forward 배치 = batch_size * 5.
# 기본 batch_size(64)면 320장 상당 — GPU 24GB에서 448 입력 기준 무리 없음.
IMAGES_PER_BATCH = max(1, batch_size // 5)


def five_crop_offsets(resized_width, resized_height):
    """서빙(app/ai_engine.py._five_crop_offsets)과 동일한 5-crop 오프셋.

    네 모서리 + 중앙(마지막). 짧은 변이 EVAL_RESIZE_SHORT(512)라
    두 변 모두 INPUT_SIZE(448) 이상이 보장된다.
    """
    span_x = resized_width - INPUT_SIZE
    span_y = resized_height - INPUT_SIZE
    return [
        (0, 0), (span_x, 0), (0, span_y), (span_x, span_y),
        (span_x // 2, span_y // 2),
    ]


class FiveCropEvalDataset(Dataset):
    """평가 경로(캐시 이미지 → 짧은 변 512 리사이즈)에서 5크롭을 뜯는 Dataset.

    반환: (crops [5, 3, 448, 448], label) — 크롭 순서는 five_crop_offsets와 동일,
    마지막(인덱스 4)이 중앙 크롭이라 기존 파이프라인과의 비교 분석에 쓴다.
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self._clahe = None  # cv2.CLAHE는 피클 불가 → 워커에서 lazy 생성

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = processed_images_dir / row["image_relpath"]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if use_clahe:
            if self._clahe is None:
                import cv2
                self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = apply_clahe(image, self._clahe)

        image = TF.resize(image, EVAL_RESIZE_SHORT, antialias=True)
        resized_width, resized_height = image.size

        crops = torch.stack([
            _eval_post_transform(
                image.crop((x, y, x + INPUT_SIZE, y + INPUT_SIZE))
            )
            for x, y in five_crop_offsets(resized_width, resized_height)
        ])

        return crops, int(row["model_label"])


def collect_probabilities_5crop(model, data_loader, device):
    """(정답, 5-crop 최대 불량 확률, 중앙 크롭 불량 확률) 리스트 3개를 반환한다."""
    labels_list = []
    max_probabilities = []
    center_probabilities = []

    with torch.no_grad():
        for crops, labels in data_loader:
            image_count, crop_count = crops.shape[0], crops.shape[1]
            flat = crops.view(-1, *crops.shape[2:]).to(device)  # [N*5, 3, H, W]

            outputs = model(flat)
            defect_probs = torch.softmax(outputs, dim=1)[:, 1]
            defect_probs = defect_probs.view(image_count, crop_count)  # [N, 5]

            labels_list.extend(labels.tolist())
            max_probabilities.extend(defect_probs.max(dim=1).values.cpu().tolist())
            center_probabilities.extend(defect_probs[:, -1].cpu().tolist())

    synchronize(device)
    return labels_list, max_probabilities, center_probabilities


def metrics_at_threshold(labels, defect_probabilities, threshold):
    """threshold 적용 시 (우수 F1, 우수 recall, 불량 recall, 불량 F1, macro F1, accuracy)."""
    predictions = [
        1 if probability >= threshold else 0
        for probability in defect_probabilities
    ]
    per_class_f1 = f1_score(
        labels, predictions, labels=[0, 1], average=None, zero_division=0
    )
    per_class_recall = recall_score(
        labels, predictions, labels=[0, 1], average=None, zero_division=0
    )
    accuracy = sum(
        1 for label, prediction in zip(labels, predictions) if label == prediction
    ) / len(labels)

    return (
        float(per_class_f1[0]), float(per_class_recall[0]),
        float(per_class_recall[1]), float(per_class_f1[1]),
        float(per_class_f1.mean()), accuracy,
        predictions,
    )


def confusion_table(labels, predictions):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=[f"실제_{name}" for name in BIN_CLASS_NAMES],
        columns=[f"예측_{name}" for name in BIN_CLASS_NAMES],
    )


def main():
    result_lines = []

    def record(text=""):
        print(text)
        result_lines.append(str(text))

    model_path = checkpoint_path("binclf", EVALUATION_TARGET)
    if not model_path.exists():
        raise FileNotFoundError(
            f"모델을 찾을 수 없습니다: {model_path}\n"
            "먼저 같은 ARCH/RUN_NAME으로 학습을 실행하세요."
        )

    output_dir = test_results_root / "binclf_v3" / ARCH / f"{RUN_NAME}_threshold_5crop"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    record("===== 임계값 튜닝 (5-crop TTA 서빙 규격) =====")
    record(f"모델 파일 : {model_path.name}")
    record(f"불량 recall 하한 (RECALL_FLOOR) : {RECALL_FLOOR}")
    record(f"이미지 배치 : {IMAGES_PER_BATCH} (x5크롭 = 실효 {IMAGES_PER_BATCH * 5})")
    record(f"사용 장치 : {device}")

    set_seed()
    _, validation_data, test_data = load_split_dataframes("binclf")

    loader_kwargs = dict(
        batch_size=IMAGES_PER_BATCH, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    validation_loader = DataLoader(FiveCropEvalDataset(validation_data), **loader_kwargs)
    test_loader = DataLoader(FiveCropEvalDataset(test_data), **loader_kwargs)

    model = create_model(ARCH, 2, use_pretrained_weights=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ---- validation에서 threshold 탐색 (test 누출 없음) ----
    validation_labels, validation_max_probs, validation_center_probs = (
        collect_probabilities_5crop(model, validation_loader, device)
    )

    # validation 샘플별 확률 저장 — 중앙 크롭 확률도 함께 (분포 이동 분석용)
    predictions_val_path = output_dir / f"predictions_val_{EVALUATION_TARGET}.csv"
    pd.DataFrame({
        "source_data_id": validation_data["source_data_id"].tolist(),
        "image_relpath": validation_data["image_relpath"].tolist(),
        "true_label": validation_labels,
        "prob_불량_5crop_max": [round(value, 6) for value in validation_max_probs],
        "prob_불량_center": [round(value, 6) for value in validation_center_probs],
    }).to_csv(predictions_val_path, index=False, encoding="utf-8-sig")

    sweep_rows = []
    for threshold in THRESHOLDS:
        good_f1, good_recall, defect_recall, defect_f1, macro_f1, accuracy, _ = (
            metrics_at_threshold(validation_labels, validation_max_probs, threshold)
        )
        sweep_rows.append({
            "threshold": threshold,
            "우수_F1": round(good_f1, 6),
            "우수_recall": round(good_recall, 6),
            "불량_recall": round(defect_recall, 6),
            "불량_F1": round(defect_f1, 6),
            "macro_F1": round(macro_f1, 6),
            "accuracy": round(accuracy, 6),
        })

    sweep_frame = pd.DataFrame(sweep_rows)
    sweep_path = output_dir / "validation_threshold_sweep.csv"
    sweep_frame.to_csv(sweep_path, index=False, encoding="utf-8-sig")

    # 선택 규칙: 불량 recall >= 하한 → macro F1 최대 → 동률 시 우수 F1 최대
    eligible = sweep_frame[sweep_frame["불량_recall"] >= RECALL_FLOOR]
    if eligible.empty:
        record(f"[경고] 불량 recall >= {RECALL_FLOOR}를 만족하는 threshold가 없습니다. "
               "전체 중 macro F1 최대를 선택합니다.")
        eligible = sweep_frame

    best_row = eligible.sort_values(
        ["macro_F1", "우수_F1"], ascending=False
    ).iloc[0]
    best_threshold = float(best_row["threshold"])

    record("")
    record(f"선택된 threshold : {best_threshold:.2f}")
    record(f"  (validation 기준 macro F1 {best_row['macro_F1']:.4f}, "
           f"우수 F1 {best_row['우수_F1']:.4f}, 불량 recall {best_row['불량_recall']:.4f})")
    record("")
    record("서빙 적용법: 환경변수 MODEL_THRESHOLD로 주입 —")
    record(f"  MODEL_THRESHOLD={best_threshold:.2f} python app/main.py")

    # ---- test에 적용 ----
    test_labels, test_max_probs, _ = collect_probabilities_5crop(
        model, test_loader, device
    )

    record("")
    record("===== Test (5-crop max): 기본 0.5 vs 선택 threshold =====")

    for tag, threshold, matrix_name in [
        ("기본 0.5", 0.5, "test_confusion_matrix_default05.csv"),
        (f"threshold {best_threshold:.2f}", best_threshold,
         "test_confusion_matrix_threshold.csv"),
    ]:
        good_f1, good_recall, defect_recall, defect_f1, macro_f1, accuracy, preds = (
            metrics_at_threshold(test_labels, test_max_probs, threshold)
        )
        record("")
        record(f"[{tag}]")
        record(f"  우수 F1 : {good_f1:.4f} | 우수 recall : {good_recall:.4f}")
        record(f"  불량 F1 : {defect_f1:.4f} | 불량 recall : {defect_recall:.4f}")
        record(f"  Macro F1 : {macro_f1:.4f} | Accuracy : {accuracy:.2%}")

        table = confusion_table(test_labels, preds)
        record(table)
        table.to_csv(output_dir / matrix_name, encoding="utf-8-sig")

    result_path = output_dir / "threshold_tuning_result.txt"
    result_path.write_text("\n".join(result_lines), encoding="utf-8-sig")

    print("")
    print("스윕 저장 :", sweep_path)
    print("결과 저장 :", result_path)
    print("val 예측 저장 :", predictions_val_path)


if __name__ == "__main__":
    main()
