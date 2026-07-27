# -*- coding: utf-8 -*-
"""등급별 Recall — EfficientNet-B0 원본(argmax) vs threshold 보정 비교 그룹 막대그래프.

같은 체크포인트(best_efficientnet_b0_finetuned_epoch20.pth)에서 결정 규칙만 바꾼
두 결과를 한 좌표평면에 놓는다 (같은 test 5,558장).
    ① 원본 (argmax)       <- threshold_tuning_result.txt의 첫 번째 등급별 표
    ② threshold 보정      <- 같은 파일의 두 번째 등급별 표
형식은 plot_recall_three_models.py와 동일 (기존 차트와 나란히 비교 가능).

실행:
    python ppt_data/plot_effnet_threshold_recall.py
    # 결과 파일이 다른 저장소에 있을 때:
    RESULT_DIR=<threshold 결과 폴더> python ppt_data/plot_effnet_threshold_recall.py
결과 (이 파일과 같은 폴더에 저장):
    effnet_threshold_recall.png
"""
import os
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 한글 라벨이 깨지지 않도록 윈도우 기본 한글 글꼴 사용, 음수 부호 깨짐 방지
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = Path(os.environ.get(
    "RESULT_DIR",
    BASE_DIR.parent / "test_results" / "efficientnet_b0" / "epoch20_threshold"
))
REPORT_PATH = RESULT_DIR / "threshold_tuning_result.txt"
# OUTPUT_NAME으로 파일명을 바꾸면 기존 차트를 덮어쓰지 않고 새로 저장됨
OUTPUT_PATH = BASE_DIR / os.environ.get(
    "OUTPUT_NAME", "effnet_threshold_recall.png"
)

GRADES = ["우수", "보통", "불량"]
# 등급마다 고정 색 — 다른 차트들과 동일
GRADE_COLORS = {"우수": "#2E86AB", "보통": "#F4A261", "불량": "#C1121F"}

# 결과 파일 안 등급별 표 순서: 첫 번째 = 원본 argmax, 두 번째 = threshold 보정
groups = [
    ("① 원본 (argmax)", 0),
    ("② threshold 보정", 1),
]

REPORT_RE = re.compile(
    r"precision\s+recall\s+f1-score\s+support(.*?)weighted avg", re.S
)


def parse_recalls(text, occurrence):
    """결과 파일에서 occurrence번째 등급별 표의 recall들을 파싱한다."""
    block = REPORT_RE.findall(text)[occurrence]
    recalls = {}
    for grade in GRADES:
        match = re.search(
            rf"{grade}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", block)
        if not match:
            raise ValueError(f"'{grade}' 행을 찾지 못했습니다 (표 {occurrence})")
        recalls[grade] = float(match.group(2))  # recall = 2번째 수치
    return recalls


report_text = REPORT_PATH.read_text(encoding="utf-8")

# 선택된 bias 값을 제목에 표기
bias_match = re.search(r"선택된 bias : ([\d.]+)", report_text)
chosen_bias = bias_match.group(1) if bias_match else "?"

BAR_WIDTH = 0.28          # 막대 폭
GROUP_GAP = 1.35          # 그룹 중심 간 간격
OFFSETS = [-BAR_WIDTH, 0.0, BAR_WIDTH]

figure, axes = plt.subplots(figsize=(11, 6.5))

group_centers = []
for group_index, (group_label, occurrence) in enumerate(groups):
    recalls = parse_recalls(report_text, occurrence)
    center = group_index * GROUP_GAP
    group_centers.append(center)

    print(f"[{group_label}]")
    for grade, offset in zip(GRADES, OFFSETS):
        recall = recalls[grade]
        x = center + offset
        axes.bar(
            x, recall, width=BAR_WIDTH,
            color=GRADE_COLORS[grade],
            edgecolor="white", linewidth=1.0,
        )
        # 막대 위 recall 값
        axes.text(
            x, recall + 0.015, f"{recall:.4f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
        # 막대 안 등급명 (색만으로 구분하지 않도록 직접 표기)
        axes.text(
            x, recall / 2, grade,
            ha="center", va="center", fontsize=11,
            color="white", fontweight="bold",
        )
        print(f"    {grade}  recall {recall:.4f}")
    print()

axes.set_title(
    f"등급별 Recall — EfficientNet-B0 원본 vs threshold 보정 (bias={chosen_bias}, Test 5,558장)",
    fontsize=16, fontweight="bold", pad=14,
)
axes.set_ylabel("Recall  (실제 등급 중 맞힌 비율)", fontsize=13)
axes.set_ylim(0, 1.05)
axes.set_xticks(group_centers)
axes.set_xticklabels([label for label, _ in groups], fontsize=13)
axes.grid(axis="y", alpha=0.3, linestyle="--")
axes.set_axisbelow(True)

legend_handles = [Patch(facecolor=GRADE_COLORS[grade], label=grade) for grade in GRADES]
axes.legend(handles=legend_handles, title="등급", fontsize=12,
            title_fontsize=12, loc="upper left",
            bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)

figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(figure)
print("저장 완료:", OUTPUT_PATH)
