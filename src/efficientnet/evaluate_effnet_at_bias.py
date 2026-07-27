# ============================================================
# [사용법]
# EfficientNet-B0을 지정한 고정 bias로 test 1회 평가하는 스크립트.
#
# tune_threshold_efficientnet.py가 "탐색"용이라면 이 스크립트는 "확정된 bias의
# test 성적표"용이다. bias 값은 반드시 validation에서 근거를 갖고 정한 값을 쓸 것
# (예: validation sweep에서 보통 recall >= 0.6을 처음 만족한 1.6).
# test를 보면서 bias를 고르면 조작(test 누출)이 된다.
#
# 실행 (윈도우 로컬):
#   $env:BIAS="1.6"; $env:DATA_DIR="D:/hn_old-building"
#   python src/efficientnet/evaluate_effnet_at_bias.py
#
# [환경변수]
#   BIAS         : 보통(1) logit에 더할 보정치 (필수)
#   BIAS_REASON  : 이 bias를 고른 근거 한 줄 (결과 파일에 기록, 기본 문구 있음)
#   RUN_NAME     : 학습 때 쓴 실험 이름 (기본 "epoch20")
#   EVAL_TARGET  : baseline 또는 finetuned (기본 "finetuned")
#   DATA_DIR / BATCH_SIZE / NUM_WORKERS : tune_threshold_efficientnet.py와 동일
#
# 결과 (bias 값이 폴더명에 붙어 기존 결과와 분리됨):
#   test_results/efficientnet_b0/<RUN_NAME>_bias<BIAS>/threshold_tuning_result.txt
#   (plot_effnet_threshold_recall.py가 그대로 파싱할 수 있는 형식)
# ============================================================

import os
from pathlib import Path

# tune_threshold_efficientnet가 import 시점에 경로를 굳히므로 그 전에 환경을 확인
BIAS = float(os.environ["BIAS"])
BIAS_REASON = os.environ.get(
    "BIAS_REASON",
    "validation sweep에서 보통 recall >= 0.6을 처음 만족한 지점"
)

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from tune_threshold_efficientnet import (
    RUN_NAME,
    ThresholdDataset,
    best_model_path,
    batch_size,
    class_names,
    collect_logits,
    create_efficientnet_model,
    evaluation_transform,
    metadata_path,
    metrics_with_bias,
    num_workers,
    processed_images_dir,
    project_dir,
)

results_dir = (
    project_dir / "test_results" / "efficientnet_b0"
    / f"{RUN_NAME}_bias{BIAS}"
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("사용 장치 :", device)
    print("평가 모델 :", best_model_path.name)
    print("고정 bias :", BIAS)

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")

    test_data = metadata[
        metadata["split"] == "test"
    ].reset_index(drop=True)

    print("Test 장수 :", len(test_data))

    test_loader = DataLoader(
        ThresholdDataset(
            test_data, processed_images_dir, evaluation_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = create_efficientnet_model()

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("\nTest 추론 중 (1회만 실행됨)...")
    test_logits, test_labels = collect_logits(model, test_loader, device)

    baseline = metrics_with_bias(test_logits, test_labels, 0.0)
    adjusted = metrics_with_bias(test_logits, test_labels, BIAS)

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
        index=matrix_index, columns=matrix_columns
    )
    adjusted_matrix = pd.DataFrame(
        confusion_matrix(
            labels_numpy, adjusted["predictions"], labels=[0, 1, 2]
        ),
        index=matrix_index, columns=matrix_columns
    )

    results_dir.mkdir(parents=True, exist_ok=True)

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
    record("고정 bias Test 평가 — EfficientNet-B0")
    record("==========================================")
    record(f"모델 파일 : {best_model_path.name}")
    record(f"실험 이름 : {RUN_NAME}")
    record(f"사용 장치 : {device}")
    record("")
    record("[bias 결정 근거]")
    record(f"선택된 bias : {BIAS} (보통 logit에 더함)")
    record(f"선택 근거 : {BIAS_REASON}")
    record("(bias는 validation에서만 결정, test는 이 1회만 평가)")
    record("")
    record("===== Test 결과 비교 (같은 체크포인트, 결정 규칙만 다름) =====")
    record("")
    record(f"{'지표':<14}{'기존 argmax':>12}{'threshold 보정':>16}")
    record(
        f"{'보통 recall':<14}"
        f"{baseline['average_recall']:>12.4f}"
        f"{adjusted['average_recall']:>16.4f}"
    )
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
