# -*- coding: utf-8 -*-
"""CNN 2종 비교 — EfficientNet-B0 vs MobileNetV2 ('불량' = positive).

불량을 positive로 보고 precision / recall / F1 을 나란히 비교한다.
    precision = TP/(TP+FP)   recall = TP/(TP+FN)   F1 = 2TP/(2TP+FP+FN)

수치는 코드에 적어두지 않고 혼동행렬에서 직접 계산한다.
읽는 파일 (이 스크립트와 같은 폴더 기준):
    test_confusion_matrix_finetuned.csv    (EfficientNet-B0 finetuned)
    test_result_baseline_김민권.txt         (MobileNetV2 baseline — 리포트 본문에서 파싱)

주의: EfficientNet 은 파인튜닝까지 마친 결과이고 MobileNetV2 는 파인튜닝 전
baseline 결과다. 학습 단계가 달라 동일 조건 비교가 아니라는 점을 그래프에도 적어둔다.

실행:
    python plot_cnn_comparison.py
결과:
    cnn_comparison.png (같은 폴더에 저장)
"""
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 라벨이 깨지지 않도록 윈도우 기본 한글 글꼴 사용
matplotlib.rcParams["font.family"] = "Malgun Gothic"
# 한글 글꼴 사용 시 음수 부호가 네모로 깨지는 것을 방지
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "cnn_comparison.png"

DEFECT_INDEX = 2  # 우수=0, 보통=1, 불량=2

cnn_sources = [
    ("EfficientNet-B0 (finetuned)", BASE_DIR / "test_confusion_matrix_finetuned.csv", "#C1121F"),
    ("MobileNetV2 (baseline)", BASE_DIR / "test_result_baseline_김민권.txt", "#2E5EAA"),
]


def read_confusion_matrix(path):
    """혼동행렬을 3x3 정수 리스트로 읽는다. CSV와 평가 리포트 텍스트를 모두 지원."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, index_col=0, encoding="utf-8-sig").values.tolist()

    text = path.read_text(encoding="utf-8")
    found = re.findall(r"실제_\S+\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if len(found) != 3:
        raise ValueError(f"{path.name}: 혼동행렬 3행을 찾지 못했습니다 (찾은 행 {len(found)})")
    return [[int(value) for value in row] for row in found]


def defect_metrics(path):
    """불량을 positive로 본 precision / recall / F1 과 건수를 계산한다."""
    matrix = read_confusion_matrix(path)

    true_positive = matrix[DEFECT_INDEX][DEFECT_INDEX]
    actual_defect = sum(matrix[DEFECT_INDEX])
    predicted_defect = sum(row[DEFECT_INDEX] for row in matrix)

    false_negative = actual_defect - true_positive   # 놓친 불량
    false_positive = predicted_defect - true_positive  # 헛경보

    return {
        "precision": true_positive / predicted_defect,
        "recall": true_positive / actual_defect,
        "f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
        "tp": true_positive,
        "fn": false_negative,
        "fp": false_positive,
        "actual": actual_defect,
    }


records = []
for display_name, source_path, color in cnn_sources:
    metrics = defect_metrics(source_path)
    metrics["name"] = display_name
    metrics["color"] = color
    records.append(metrics)
    print(f"{display_name:30s} P {metrics['precision']:.4f} | "
          f"R {metrics['recall']:.4f} | F1 {metrics['f1']:.4f} | "
          f"놓침 {metrics['fn']:,} / 헛경보 {metrics['fp']:,}")

metric_labels = ["precision", "recall", "F1"]
metric_keys = ["precision", "recall", "f1"]

figure, (left_axes, right_axes) = plt.subplots(
    1, 2,
    figsize=(15, 6.8),
    gridspec_kw={"width_ratios": [1.45, 1]},
)

# ---------------------------------------------------------------- 왼쪽: 지표 3종 비교
x_positions = np.arange(len(metric_labels))
bar_width = 0.34

for index, record in enumerate(records):
    offset = (index - 0.5) * bar_width
    values = [record[key] for key in metric_keys]

    bars = left_axes.bar(
        x_positions + offset,
        values,
        width=bar_width,
        label=record["name"],
        color=record["color"],
        edgecolor="white",
        linewidth=1.2,
    )

    for bar, value in zip(bars, values):
        left_axes.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

left_axes.set_title(
    "불량 = positive 기준 지표 비교",
    fontsize=14,
    fontweight="bold",
    pad=14,
)
left_axes.set_ylabel("점수", fontsize=12)
left_axes.set_xticks(x_positions)
left_axes.set_xticklabels(metric_labels, fontsize=13)
left_axes.set_ylim(0, 1.08)
left_axes.grid(axis="y", alpha=0.3, linestyle="--")
left_axes.set_axisbelow(True)
# 막대가 높아 그래프 안에 두면 값을 가리므로 x축 아래에 배치
left_axes.legend(
    fontsize=11,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.09),
    ncol=2,
    frameon=False,
)

# ---------------------------------------------------------------- 오른쪽: 놓친 불량 수
missed_positions = np.arange(len(records))
missed_values = [record["fn"] for record in records]

missed_bars = right_axes.bar(
    missed_positions,
    missed_values,
    width=0.5,
    color=[record["color"] for record in records],
    edgecolor="white",
    linewidth=1.2,
)

for bar, record in zip(missed_bars, records):
    right_axes.text(
        bar.get_x() + bar.get_width() / 2,
        record["fn"] + 8,
        f"{record['fn']:,}채",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )
    right_axes.text(
        bar.get_x() + bar.get_width() / 2,
        record["fn"] / 2,
        f"{record['tp']:,} / {record['actual']:,}\n적중",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
        linespacing=1.5,
    )

right_axes.set_title(
    "놓친 불량 건축물 수 (FN) — 낮을수록 좋음",
    fontsize=14,
    fontweight="bold",
    pad=14,
)
right_axes.set_ylabel("놓친 불량 수 (채)", fontsize=12)
right_axes.set_xticks(missed_positions)
right_axes.set_xticklabels(
    [record["name"].replace(" (", "\n(") for record in records],
    fontsize=11,
)
right_axes.set_ylim(0, max(missed_values) * 1.25)
right_axes.grid(axis="y", alpha=0.3, linestyle="--")
right_axes.set_axisbelow(True)

figure.suptitle(
    "CNN 2종 비교 — EfficientNet-B0 vs MobileNetV2 (test 5,558장 중 실제 불량 4,437채)",
    fontsize=16,
    fontweight="bold",
)

figure.text(
    0.5, 0.015,
    "EfficientNet-B0 은 파인튜닝까지 마친 결과, MobileNetV2 는 파인튜닝 전 baseline 결과입니다. "
    "학습 단계가 달라 동일 조건 비교가 아닙니다.",
    ha="center",
    fontsize=9.5,
    color="#666666",
)

figure.tight_layout(rect=[0, 0.035, 1, 0.94])
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")

print("\n저장 완료:", OUTPUT_PATH)
