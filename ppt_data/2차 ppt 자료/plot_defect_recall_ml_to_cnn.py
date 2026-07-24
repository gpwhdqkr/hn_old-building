# -*- coding: utf-8 -*-
"""모델별 '불량' Recall 막대그래프 — 고전 ML(왼쪽·회색) → CNN(오른쪽·빨강).

'모델 테스트 결과.xlsx' 에서 각 모델의 불량 recall(불량=positive)을 읽어
왼쪽에 고전 ML(회색), 오른쪽에 CNN(빨강)을 배치한다.
각 그룹은 recall 오름차순으로 정렬해 왼쪽→오른쪽으로 성능이 올라가며
'고전 ML → CNN' 흐름으로 읽히게 한다.

엑셀 열 구성(header 없이 읽음): col2=모델이름, col10=recall, col13=비고(적중 수)

실행:
    python plot_defect_recall_ml_to_cnn.py
결과 (같은 폴더에 저장):
    defect_recall_ml_to_cnn.png
"""
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

# 한글 라벨이 깨지지 않도록 윈도우 기본 한글 글꼴 사용, 음수 부호 깨짐 방지
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "모델 테스트 결과.xlsx"
OUTPUT_PATH = BASE_DIR / "defect_recall_ml_to_cnn.png"

NAME_COL, RECALL_COL, NOTE_COL = 2, 10, 13
CNN_KEYWORDS = ("EfficientNet", "MobileNet")

GRAY = "#8D99A6"   # 고전 ML
RED = "#C1121F"    # CNN
ZONE_GRAY = "#F0F2F4"
ZONE_RED = "#FBEAEA"


def load_models():
    """엑셀에서 (이름, recall, 적중수, CNN여부) 목록을 읽는다."""
    df = pd.read_excel(EXCEL_PATH, header=None)
    models = []
    for _, row in df.iterrows():
        recall = row[RECALL_COL]
        name = row[NAME_COL]
        # recall 이 숫자이고 이름이 있는 행만 데이터로 취급
        if not isinstance(recall, (int, float)) or pd.isna(recall):
            continue
        if not isinstance(name, str) or not name.strip():
            continue

        clean_name = name.replace("\n", " ").strip()
        is_cnn = any(keyword in clean_name for keyword in CNN_KEYWORDS)

        # 비고에서 '3,906/4,437' 형태의 적중/전체 수를 파싱(있으면)
        note = str(row[NOTE_COL])
        hit = re.search(r"([\d,]+)\s*/\s*([\d,]+)", note)
        correct = int(hit.group(1).replace(",", "")) if hit else None

        models.append({
            "name": clean_name,
            "recall": float(recall),
            "correct": correct,
            "is_cnn": is_cnn,
        })
    return models


# 보기 좋은 줄바꿈 라벨
DISPLAY_NAME = {
    "EfficientNet-B0 finetuned": "EfficientNet-B0\n(finetuned)",
    "MobileNetV2 baseline": "MobileNetV2\n(baseline)",
}

models = load_models()

# 고전 ML(왼쪽) → CNN(오른쪽), 각 그룹은 recall 오름차순 (왼쪽→오른쪽 상승)
classic = sorted([m for m in models if not m["is_cnn"]], key=lambda m: m["recall"])
cnn = sorted([m for m in models if m["is_cnn"]], key=lambda m: m["recall"])
ordered = classic + cnn

# x 위치: 고전 ML 은 0,1,2 / CNN 은 그룹 사이 간격을 두고 3.4, 4.4 ...
positions = []
for index in range(len(classic)):
    positions.append(index)
gap = 1.4
for index in range(len(cnn)):
    positions.append(len(classic) - 1 + gap + index)

names = [DISPLAY_NAME.get(m["name"], m["name"]) for m in ordered]
recalls = [m["recall"] for m in ordered]
colors = [RED if m["is_cnn"] else GRAY for m in ordered]

print("불량 Recall (고전 ML → CNN):")
for pos, m in zip(positions, ordered):
    tag = "CNN " if m["is_cnn"] else "고전ML"
    print(f"  x={pos:4.1f} [{tag}] {m['name']:26s} recall {m['recall']:.4f}")

figure, axes = plt.subplots(figsize=(12, 7))

# --- 두 영역 배경 음영 + 경계선 ---
boundary = (positions[len(classic) - 1] + positions[len(classic)]) / 2
left_edge = positions[0] - 0.7
right_edge = positions[-1] + 0.7
axes.axvspan(left_edge, boundary, color=ZONE_GRAY, zorder=0)
axes.axvspan(boundary, right_edge, color=ZONE_RED, zorder=0)
axes.axvline(boundary, color="#BBBBBB", linestyle="--", linewidth=1.2, zorder=1)

# --- 막대 ---
bars = axes.bar(
    positions, recalls, color=colors, width=0.72,
    edgecolor="white", linewidth=1.2, zorder=3,
)

# 막대 위 recall 값 / 막대 안 적중 수
for bar, m in zip(bars, ordered):
    axes.text(
        bar.get_x() + bar.get_width() / 2, m["recall"] + 0.015,
        f"{m['recall']:.4f}", ha="center", va="bottom",
        fontsize=14, fontweight="bold", zorder=4,
    )
    if m["correct"] is not None:
        axes.text(
            bar.get_x() + bar.get_width() / 2, m["recall"] / 2,
            f"{m['correct']:,}채\n적중", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold",
            linespacing=1.4, zorder=4,
        )

# --- 그룹 라벨(영역 상단) ---
classic_center = sum(positions[:len(classic)]) / len(classic)
cnn_center = sum(positions[len(classic):]) / len(cnn)
axes.text(classic_center, 1.05, "고전 ML", ha="center", va="center",
          fontsize=15, fontweight="bold", color="#5A6472")
axes.text(cnn_center, 1.05, "CNN", ha="center", va="center",
          fontsize=15, fontweight="bold", color=RED)

# --- 진행 방향 화살표: 고전 ML → CNN ---
axes.annotate(
    "", xy=(cnn_center, 0.965), xytext=(classic_center, 0.965),
    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2.2),
    zorder=5,
)
axes.text((classic_center + cnn_center) / 2, 0.99,
          "불량 검출력 향상", ha="center", va="bottom",
          fontsize=11, color="#333333")

axes.set_title(
    "모델별 불량 Recall — 고전 ML → CNN",
    fontsize=18, fontweight="bold", pad=30,
)
axes.set_ylabel("불량 Recall  (실제 불량 중 잡아낸 비율)", fontsize=13)
axes.set_ylim(0, 1.12)
axes.set_xlim(left_edge, right_edge)
axes.set_xticks(positions)
axes.set_xticklabels(names, fontsize=12)
axes.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
axes.set_axisbelow(True)

# 색 범례
legend_handles = [
    Patch(facecolor=GRAY, label="고전 ML"),
    Patch(facecolor=RED, label="CNN"),
]
axes.legend(handles=legend_handles, loc="lower right", fontsize=12,
            framealpha=0.9)

figure.text(
    0.5, 0.015,
    "불량 = positive · test 5,558장 중 실제 불량 4,437채 기준 · "
    "고전 ML(SVM·Decision Tree·KNN)은 그레이스케일 + PCA 100차원 압축 학습",
    ha="center", fontsize=9.5, color="#666666",
)

figure.tight_layout(rect=[0, 0.03, 1, 1])
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(figure)
print("\n저장 완료:", OUTPUT_PATH.name)
