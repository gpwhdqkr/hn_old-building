# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# v2 이원화 임계값 튜닝 (선택 단계). GPU가 없어도 됩니다 (추론 2회뿐).
#
# 기본 argmax(=0.5 기준) 대신 "불량 확률 >= threshold면 불량"의 threshold를
# validation에서만 탐색해 (test 누출 없음) test에 적용해 봅니다.
# 선택 규칙: 불량 recall >= RECALL_FLOOR 필터 → macro F1 최대 → 동률 시 우수 F1 최대.
#
# 실행 (런팟 = 리눅스):
#   ARCH=resnet50 RUN_NAME=v2r50a python src/v2/tune_threshold_binclf_v2.py
# (PowerShell):
#   $env:ARCH="resnet50"; $env:RUN_NAME="v2r50a"; python src/v2/tune_threshold_binclf_v2.py
#
# [환경변수] ARCH RUN_NAME EVAL_TARGET(finetuned) RECALL_FLOOR(0.85) — README 참고
#
# 결과 (test_results/binclf_v2/<ARCH>/<RUN_NAME>_threshold/):
#   validation_threshold_sweep.csv
#   test_confusion_matrix_default05.csv / test_confusion_matrix_threshold.csv
#   predictions_val_<EVAL_TARGET>.csv        — validation 샘플별 확률 (v2 신규)
#   threshold_tuning_result.txt
# ============================================================

import os

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from model_factory_v2 import create_model
from pipeline_v2 import ARCH, RUN_NAME, checkpoint_path, get_device, synchronize
from preprocess_v2 import (
    BIN_CLASS_NAMES,
    build_dataloaders,
    load_split_dataframes,
    set_seed,
    test_results_root,
)

EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")
RECALL_FLOOR = float(os.environ.get("RECALL_FLOOR", 0.85))

# 탐색 범위: 0.02 ~ 0.98, 0.01 간격 (기존 binclf와 동일)
THRESHOLDS = [round(0.02 + 0.01 * index, 2) for index in range(97)]


def collect_probabilities(model, data_loader, device):
    """(정답 리스트, 불량 확률 리스트)를 반환한다."""
    labels_list = []
    defect_probabilities = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            labels_list.extend(labels.tolist())
            defect_probabilities.extend(probabilities[:, 1].cpu().tolist())

    synchronize(device)
    return labels_list, defect_probabilities


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

    output_dir = test_results_root / "binclf_v2" / ARCH / f"{RUN_NAME}_threshold"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    record("===== 임계값 튜닝 =====")
    record(f"모델 파일 : {model_path.name}")
    record(f"불량 recall 하한 (RECALL_FLOOR) : {RECALL_FLOOR}")
    record(f"사용 장치 : {device}")

    set_seed()
    _, validation_loader, test_loader = build_dataloaders("binclf")
    _, validation_data, _ = load_split_dataframes("binclf")

    model = create_model(ARCH, 2, use_pretrained_weights=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ---- validation에서 threshold 탐색 (test 누출 없음) ----
    validation_labels, validation_probabilities = collect_probabilities(
        model, validation_loader, device
    )

    # validation 샘플별 확률 저장 (v2 신규)
    predictions_val_path = output_dir / f"predictions_val_{EVALUATION_TARGET}.csv"
    pd.DataFrame({
        "source_data_id": validation_data["source_data_id"].tolist(),
        "image_relpath": validation_data["image_relpath"].tolist(),
        "true_label": validation_labels,
        "prob_불량": [round(value, 6) for value in validation_probabilities],
    }).to_csv(predictions_val_path, index=False, encoding="utf-8-sig")

    sweep_rows = []
    for threshold in THRESHOLDS:
        good_f1, good_recall, defect_recall, defect_f1, macro_f1, accuracy, _ = (
            metrics_at_threshold(validation_labels, validation_probabilities, threshold)
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

    # ---- test에 적용 ----
    test_labels, test_probabilities = collect_probabilities(model, test_loader, device)

    record("")
    record("===== Test: 기본 0.5 vs 선택 threshold =====")

    for tag, threshold, matrix_name in [
        ("기본 0.5", 0.5, "test_confusion_matrix_default05.csv"),
        (f"threshold {best_threshold:.2f}", best_threshold,
         "test_confusion_matrix_threshold.csv"),
    ]:
        good_f1, good_recall, defect_recall, defect_f1, macro_f1, accuracy, preds = (
            metrics_at_threshold(test_labels, test_probabilities, threshold)
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
