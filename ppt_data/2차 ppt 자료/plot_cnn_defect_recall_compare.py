# -*- coding: utf-8 -*-
"""CNN 2종의 '불량' Recall 비교 막대그래프.

'모델 테스트 결과.xlsx' 에서 CNN 두 모델(EfficientNet-B0 finetuned,
MobileNetV2 baseline)의 불량 recall(불량=positive)만 읽어 비교한다.
두 값이 매우 근소하므로 차이를 상단에 함께 표시한다.

엑셀 열 구성(header 없이 읽음): col2=모델이름, col10=recall, col13=비고(적중 수)

실행:
    python plot_cnn_defect_recall_compare.py
결과 (같은 폴더에 저장):
    cnn_defect_recall_compare.png
"""
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# 한글 라벨이 깨지지 않도록 윈도우 기본 한글 글꼴 사용, 음수 부호 깨짐 방지
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "모델 테스트 결과.xlsx"
OUTPUT_PATH = BASE_DIR / "cnn_defect_recall_compare.png"

NAME_COL, RECALL_COL, NOTE_COL = 2, 10, 13
CNN_KEYWORDS = ("EfficientNet", "MobileNet")
TOTAL_DEFECT = 4437  # 실제 불량(테스트) 총 수

# 표시 이름과 막대 색 (왼쪽=빨강, 오른쪽=파랑)
DISPLAY = {
    "EfficientNet-B0 finetuned": ("EfficientNet-B0\n(finetuned)", "#C1121F"),
    "MobileNetV2 baseline": ("MobileNetV2\n(baseline)", "#2E86AB"),
}


def load_cnn_models():
    """엑셀에서 CNN 두 모델의 (이름, recall, 적중수)를 읽는다."""
    df = pd.read_excel(EXCEL_PATH, header=None)
    models = []
    for _, row in df.iterrows():
        recall, name = row[RECALL_COL], row[NAME_COL]
        if not isinstance(recall, (int, float)) or pd.isna(recall):
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        clean_name = name.replace("\n", " ").strip()
        if not any(keyword in clean_name for keyword in CNN_KEYWORDS):
            continue
        note = str(row[NOTE_COL])
        hit = re.search(r"([\d,]+)\s*/\s*([\d,]+)", note)
        correct = int(hit.group(1).replace(",", "")) if hit else None
        models.append({
            "name": clean_name,
            "recall": float(recall),
            "correct": correct,
        })
    return models


models = load_cnn_models()
# recall 오름차순 (왼쪽 낮음 → 오른쪽 높음)
models.sort(key=lambda m: m["recall"])

labels = [DISPLAY[m["name"]][0] for m in models]
colors = [DISPLAY[m["name"]][1] for m in models]
recalls = [m["recall"] for m in models]
positions = [0, 1]

print("CNN 2종 불량 Recall:")
for m in models:
    missed = TOTAL_DEFECT - m["correct"] if m["correct"] is not None else None
    print(f"  {m['name']:26s} recall {m['recall']:.4f}  "
          f"(적중 {m['correct']:,} / 놓침 {missed:,})")

figure, axes = plt.subplots(figsize=(8, 7))

bars = axes.bar(
    positions, recalls, color=colors, width=0.52,
    edgecolor="white", linewidth=1.4, zorder=3,
)

# 막대 위 recall 값 / 막대 안 적중·놓침 수
for bar, m in zip(bars, models):
    axes.text(
        bar.get_x() + bar.get_width() / 2, m["recall"] + 0.012,
        f"{m['recall']:.4f}", ha="center", va="bottom",
        fontsize=17, fontweight="bold", zorder=4,
    )
    if m["correct"] is not None:
        missed = TOTAL_DEFECT - m["correct"]
        axes.text(
            bar.get_x() + bar.get_width() / 2, m["recall"] / 2,
            f"{m['correct']:,}채 적중\n({missed:,}채 놓침)",
            ha="center", va="center", fontsize=12.5,
            color="white", fontweight="bold", linespacing=1.5, zorder=4,
        )

# --- 두 막대 top 사이 차이 표시(브래킷) ---
higher = max(recalls)
bracket_y = higher + 0.06
axes.plot([0, 0, 1, 1],
          [higher + 0.03, bracket_y, bracket_y, higher + 0.03],
          color="#333333", linewidth=1.4, zorder=5)
diff = abs(recalls[1] - recalls[0])
diff_hit = abs(models[1]["correct"] - models[0]["correct"])
axes.text(
    0.5, bracket_y + 0.008,
    f"차이 +{diff:.4f}  ·  불량 {diff_hit}채 더 검출",
    ha="center", va="bottom", fontsize=12.5, fontweight="bold",
    color="#333333", zorder=5,
)

axes.set_title(
    "CNN 2종 불량 Recall 비교",
    fontsize=18, fontweight="bold", pad=14,
)
axes.set_ylabel("불량 Recall  (실제 불량 중 잡아낸 비율)", fontsize=13)
axes.set_ylim(0, 1.05)
axes.set_xlim(-0.6, 1.6)
axes.set_xticks(positions)
axes.set_xticklabels(labels, fontsize=13)
axes.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
axes.set_axisbelow(True)

figure.text(
    0.5, 0.015,
    "불량 = positive · test 5,558장 중 실제 불량 4,437채 기준",
    ha="center", fontsize=10, color="#666666",
)

figure.tight_layout(rect=[0, 0.03, 1, 1])
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(figure)
print("\n저장 완료:", OUTPUT_PATH.name)
