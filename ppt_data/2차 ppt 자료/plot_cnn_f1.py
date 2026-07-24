# -*- coding: utf-8 -*-
"""CNN 2종의 '불량' F1 비교 막대그래프 (불량 = positive).

    F1 = 2TP / (2TP + FP + FN)
      TP = 실제 불량을 불량으로 맞힘
      FN = 실제 불량을 놓침
      FP = 우수·보통을 불량으로 잘못 봄

수치는 코드에 적어두지 않고 혼동행렬에서 직접 계산한다.
읽는 파일 (이 스크립트와 같은 폴더 기준):
    test_confusion_matrix_finetuned.csv    (EfficientNet-B0 finetuned)
    test_result_baseline_김민권.txt         (MobileNetV2 baseline)

실행:
    python plot_cnn_f1.py
결과:
    cnn_f1_comparison.png
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
OUTPUT_PATH = BASE_DIR / "cnn_f1_comparison.png"

DEFECT_INDEX = 2  # 우수=0, 보통=1, 불량=2

cnn_sources = [
    ("EfficientNet-B0\nfinetuned", BASE_DIR / "test_confusion_matrix_finetuned.csv", "#C1121F"),
    ("MobileNetV2\nbaseline", BASE_DIR / "test_result_baseline_김민권.txt", "#2E5EAA"),
]


def read_confusion_matrix(path):
    """혼동행렬을 3x3 정수 리스트로 읽는다. CSV와 평가 리포트 텍스트를 모두 지원."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, index_col=0, encoding="utf-8-sig").values.tolist()

    text = path.read_text(encoding="utf-8")
    found = re.findall(r"실제_\S+\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if len(found) != 3:
        raise ValueError(f"{path.name}: 혼동행렬 3행을 찾지 못했습니다")
    return [[int(value) for value in row] for row in found]


def defect_f1(path):
    matrix = read_confusion_matrix(path)

    true_positive = matrix[DEFECT_INDEX][DEFECT_INDEX]
    actual_defect = sum(matrix[DEFECT_INDEX])
    predicted_defect = sum(row[DEFECT_INDEX] for row in matrix)

    false_negative = actual_defect - true_positive
    false_positive = predicted_defect - true_positive

    return {
        "f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
        "precision": true_positive / predicted_defect,
        "recall": true_positive / actual_defect,
        "fn": false_negative,
        "fp": false_positive,
    }


records = []
for display_name, source_path, color in cnn_sources:
    metrics = defect_f1(source_path)
    metrics["name"] = display_name
    metrics["color"] = color
    records.append(metrics)
    print(f"{display_name.replace(chr(10), ' '):30s} 불량 F1 {metrics['f1']:.4f}  "
          f"(P {metrics['precision']:.4f} / R {metrics['recall']:.4f})")

gap = abs(records[0]["f1"] - records[1]["f1"])
print(f"\n두 모델 불량 F1 차이: {gap:.6f}")

names = [record["name"] for record in records]
f1_values = [record["f1"] for record in records]
colors = [record["color"] for record in records]

figure, axes = plt.subplots(figsize=(9, 7))

bars = axes.bar(names, f1_values, width=0.5, color=colors, edgecolor="white", linewidth=1.2)

# 막대 위에 F1 값, 막대 안에 precision / recall
for bar, record in zip(bars, records):
    axes.text(
        bar.get_x() + bar.get_width() / 2,
        record["f1"] + 0.015,
        f"{record['f1']:.4f}",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
    )
    axes.text(
        bar.get_x() + bar.get_width() / 2,
        record["f1"] / 2,
        f"precision {record['precision']:.4f}\nrecall {record['recall']:.4f}",
        ha="center",
        va="center",
        fontsize=12,
        color="white",
        fontweight="bold",
        linespacing=1.8,
    )

axes.set_title("CNN 2종 불량 F1 비교", fontsize=18, fontweight="bold", pad=16)
axes.set_ylabel("불량 F1", fontsize=13)
axes.set_ylim(0, 1.12)
axes.tick_params(axis="x", labelsize=12)
axes.grid(axis="y", alpha=0.3, linestyle="--")
axes.set_axisbelow(True)

# 두 값이 거의 같으므로 눈으로는 구분되지 않는다는 점을 명시한다
axes.annotate(
    f"차이 {gap:.4f} — 사실상 동률",
    xy=(0.5, 1.045),
    xycoords=("axes fraction", "data"),
    ha="center",
    fontsize=13,
    fontweight="bold",
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F2F2F2", edgecolor="#BBBBBB"),
)

figure.text(
    0.5, 0.035,
    "불량 = positive · test 5,558장 중 실제 불량 4,437채 기준",
    ha="center",
    fontsize=10.5,
    color="#444444",
)
figure.text(
    0.5, 0.008,
    "EfficientNet-B0 은 파인튜닝까지 마친 결과, MobileNetV2 는 파인튜닝 전 baseline 결과입니다.",
    ha="center",
    fontsize=9.5,
    color="#666666",
)

figure.tight_layout(rect=[0, 0.06, 1, 1])
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")

print("저장 완료:", OUTPUT_PATH)
