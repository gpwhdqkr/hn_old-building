# -*- coding: utf-8 -*-
"""등급별(우수·보통·불량) Recall — 두 모델을 한 좌표평면에 묶은 그룹 막대그래프.

한 좌표평면(축 하나)에 두 그룹을 놓는다.
    ① MobileNetV2 (baseline)     <- test_result_baseline_김민권.txt
    ② EfficientNet-B0 (finetuned) <- evaluate_finetuned_output.txt
각 그룹 안의 우수·보통·불량 막대 3개는 서로 여백 없이 붙여 그린다.

리포트의 '등급별 성능' 표에서 recall(2번째 수치)을 파싱한다. 예)
    우수     0.6550    0.7465    0.6978       501
             ^precision ^recall  ^f1        ^support

실행:
    python plot_recall_grouped.py
결과 (같은 폴더에 저장):
    recall_by_grade_grouped.png
"""
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 한글 라벨이 깨지지 않도록 윈도우 기본 한글 글꼴 사용, 음수 부호 깨짐 방지
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "recall_by_grade_grouped.png"

# 좌 -> 우 순서: ① MobileNet, ② EfficientNet
# (그룹 라벨, 리포트 파일)
groups = [
    ("① MobileNetV2\n(baseline)", BASE_DIR / "test_result_baseline_김민권.txt"),
    ("② EfficientNet-B0\n(finetuned)", BASE_DIR / "evaluate_finetuned_output.txt"),
]

GRADES = ["우수", "보통", "불량"]
# 등급마다 고정 색 — 두 그룹에서 같은 등급은 같은 색으로 비교되게 한다
GRADE_COLORS = {"우수": "#2E86AB", "보통": "#F4A261", "불량": "#C1121F"}


def parse_recall(report_path):
    """리포트 '등급별 성능' 표에서 등급별 recall 을 파싱한다."""
    text = report_path.read_text(encoding="utf-8")
    recalls = {}
    for grade in GRADES:
        match = re.search(
            rf"{grade}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", text)
        if not match:
            raise ValueError(f"{report_path.name}: '{grade}' 행을 찾지 못했습니다")
        recalls[grade] = float(match.group(2))  # recall = 2번째 수치
    return recalls


BAR_WIDTH = 0.28          # 막대 폭
GROUP_GAP = 1.35          # 그룹 중심 간 간격 (그룹 사이 여백 확보)
# 그룹 안 3개 막대 중심 오프셋: 폭만큼 떨어지게 두면 서로 딱 붙는다(여백 0)
OFFSETS = [-BAR_WIDTH, 0.0, BAR_WIDTH]

figure, axes = plt.subplots(figsize=(11, 6.5))

group_centers = []
for group_index, (group_label, report_path) in enumerate(groups):
    recalls = parse_recall(report_path)
    center = group_index * GROUP_GAP
    group_centers.append(center)

    print(f"[{group_label.splitlines()[0]}]")
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
        # 막대 안 등급명 (가로)
        axes.text(
            x, recall / 2, grade,
            ha="center", va="center", fontsize=11,
            color="white", fontweight="bold",
        )
        print(f"    {grade}  recall {recall:.4f}")
    print()

axes.set_title(
    "등급별 Recall — MobileNetV2 vs EfficientNet-B0",
    fontsize=17, fontweight="bold", pad=14,
)
axes.set_ylabel("Recall  (실제 등급 중 맞힌 비율)", fontsize=13)
axes.set_ylim(0, 1.0)
axes.set_xticks(group_centers)
axes.set_xticklabels([label for label, _ in groups], fontsize=13)
axes.grid(axis="y", alpha=0.3, linestyle="--")
axes.set_axisbelow(True)

# 등급 색 범례 — 막대를 가리지 않게 그래프 바깥(오른쪽)에 둔다
legend_handles = [Patch(facecolor=GRADE_COLORS[grade], label=grade) for grade in GRADES]
axes.legend(handles=legend_handles, title="등급", fontsize=12,
            title_fontsize=12, loc="upper left",
            bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)

figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(figure)
print("저장 완료:", OUTPUT_PATH.name)
