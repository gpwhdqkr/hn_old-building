# -*- coding: utf-8 -*-
"""노후 건축물 등급분류 — 모델별 '불량' Recall 비교 막대그래프.

이 프로젝트의 목표는 불량 건축물을 놓치지 않는 것이므로 불량을 positive로 본다.
    불량 Recall = 불량으로 맞힌 수 / 실제 불량 수
                = 혼동행렬의 (실제 불량, 예측 불량) / (실제 불량 행 합계)

값을 코드에 적어두지 않고 혼동행렬 CSV에서 직접 계산한다.
읽는 파일 (모두 이 스크립트와 같은 폴더 기준):
    test_confusion_matrix_finetuned.csv      (EfficientNet-B0 finetuned)
    test_result_baseline_김민권.txt           (MobileNetV2 baseline — 리포트 본문에서 파싱)
    classic ml/confusion_matrix_svm.csv
    classic ml/confusion_matrix_knn.csv
    classic ml/confusion_matrix_decision_tree.csv

실행:
    python plot_recall_comparison.py
결과:
    recall_comparison.png (같은 폴더에 저장)
"""
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# 한글 라벨이 깨지지 않도록 윈도우 기본 한글 글꼴 사용
matplotlib.rcParams["font.family"] = "Malgun Gothic"
# 한글 글꼴 사용 시 음수 부호가 네모로 깨지는 것을 방지
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "recall_comparison.png"

# (표시 이름, 혼동행렬 CSV 경로)
# 혼동행렬은 네 파일 모두 행=실제, 열=예측, 등급 순서 우수·보통·불량으로 동일하다.
# 헤더 문구만 다르므로(예측_불량 / 불량) 이름이 아닌 위치로 읽는다.
model_sources = [
    ("EfficientNet-B0\n(finetuned)", BASE_DIR / "test_confusion_matrix_finetuned.csv"),
    ("MobileNetV2\n(baseline)", BASE_DIR / "test_result_baseline_김민권.txt"),
    ("SVM", BASE_DIR / "classic ml" / "confusion_matrix_svm.csv"),
    ("Decision Tree", BASE_DIR / "classic ml" / "confusion_matrix_decision_tree.csv"),
    ("KNN", BASE_DIR / "classic ml" / "confusion_matrix_knn.csv"),
]

DEFECT_INDEX = 2  # 우수=0, 보통=1, 불량=2

# CNN 두 종은 색을 따로 줘서 고전 ML과 구분한다
CNN_NAMES = {"EfficientNet-B0\n(finetuned)", "MobileNetV2\n(baseline)"}


def read_confusion_matrix(path):
    """혼동행렬을 3x3 정수 리스트로 읽는다. CSV와 평가 리포트 텍스트를 모두 지원."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, index_col=0, encoding="utf-8-sig").values.tolist()

    # 평가 리포트 텍스트: '실제_불량   92   415   3930' 형태의 3줄을 찾는다
    text = path.read_text(encoding="utf-8")
    found = re.findall(r"실제_\S+\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if len(found) != 3:
        raise ValueError(f"{path.name}: 혼동행렬 3행을 찾지 못했습니다 (찾은 행 {len(found)})")
    return [[int(value) for value in row] for row in found]


def load_defect_recall(path):
    """불량 recall과 실제/적중/놓친 건수를 계산한다."""
    matrix = read_confusion_matrix(path)

    # 실제 불량 행
    actual_defect_row = matrix[DEFECT_INDEX]

    # 불량으로 올바르게 예측한 수 (대각 원소)
    correct_count = int(actual_defect_row[DEFECT_INDEX])

    # 실제 불량 전체 수
    total_count = int(sum(actual_defect_row))

    recall = correct_count / total_count
    missed_count = total_count - correct_count

    return recall, correct_count, total_count, missed_count


records = []
for display_name, source_path in model_sources:
    recall, correct, total, missed = load_defect_recall(source_path)
    records.append({
        "name": display_name,
        "recall": recall,
        "correct": correct,
        "total": total,
        "missed": missed,
    })
    print(f"{display_name.replace(chr(10), ' '):28s} "
          f"불량 recall {recall:.4f}  ({correct:,}/{total:,} 적중, {missed:,}채 놓침)")

# 성능이 높은 순으로 정렬
records.sort(key=lambda record: record["recall"], reverse=True)

names = [record["name"] for record in records]
recalls = [record["recall"] for record in records]

# CNN 2종은 붉은 계열로 강조, 고전 ML 3종은 회색으로 묶는다
colors = ["#C1121F" if record["name"] in CNN_NAMES else "#9AA5B1" for record in records]

figure, axes = plt.subplots(figsize=(13, 7))

bars = axes.bar(
    names,
    recalls,
    color=colors,
    width=0.6,
    edgecolor="white",
    linewidth=1.2,
)

# 막대 위: recall 값 / 막대 안: 적중·놓침 건수
for bar, record in zip(bars, records):
    axes.text(
        bar.get_x() + bar.get_width() / 2,
        record["recall"] + 0.018,
        f"{record['recall']:.4f}",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )

    axes.text(
        bar.get_x() + bar.get_width() / 2,
        record["recall"] / 2,
        f"{record['correct']:,} / {record['total']:,} 적중\n"
        f"{record['missed']:,}채 놓침",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
        linespacing=1.5,
    )

axes.set_title(
    "불량 건축물을 얼마나 잡아내는가 — 모델별 불량 Recall",
    fontsize=17,
    fontweight="bold",
    pad=16,
)

axes.set_ylabel("불량 Recall  (실제 불량 중 맞힌 비율)", fontsize=13)
axes.set_ylim(0, 1.0)
axes.tick_params(axis="x", labelsize=12)
axes.grid(axis="y", alpha=0.3, linestyle="--")
axes.set_axisbelow(True)

figure.text(
    0.5, 0.945,
    "불량 = positive · test 5,558장 중 실제 불량 4,437채 기준",
    ha="center",
    fontsize=11.5,
    color="#444444",
)

figure.text(
    0.5, 0.028,
    "붉은 막대 = CNN 2종 · 회색 막대 = 고전 ML 3종 "
    "(클래스당 2,000장·총 6,000장만 학습, 그레이스케일 + PCA 100차원 압축)",
    ha="center",
    fontsize=9.5,
    color="#666666",
)

figure.text(
    0.5, 0.005,
    "EfficientNet-B0 은 파인튜닝까지 마친 결과, MobileNetV2 는 파인튜닝 전 baseline 결과로 학습 단계가 다릅니다.",
    ha="center",
    fontsize=9.5,
    color="#666666",
)

figure.tight_layout(rect=[0, 0.04, 1, 0.93])
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")

print("\n저장 완료:", OUTPUT_PATH)
