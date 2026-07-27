# -*- coding: utf-8 -*-
"""등급별(우수·보통·불량) Recall — 세 모델을 한 좌표평면에 묶은 그룹 막대그래프.

한 좌표평면(축 하나)에 세 그룹을 놓는다 (모두 finetuned, 같은 test 5,558장).
    ① MobileNetV2      <- data/processed/test_result_finetuned.txt
    ② EfficientNet-B0  <- test_results/efficientnet_b0/epoch20/evaluate_finetuned_result.txt
    ③ ResNet-18        <- test_results/resnet18/r18_e20/evaluate_finetuned_result.txt
각 그룹 안의 우수·보통·불량 막대 3개는 서로 여백 없이 붙여 그린다.
(2차 ppt 자료/plot_recall_grouped.py와 동일한 형식 — 모델만 3개로 확장)

리포트의 '등급별 성능' 표에서 recall(2번째 수치)을 파싱한다. 예)
    우수     0.6550    0.7465    0.6978       501
             ^precision ^recall  ^f1        ^support

실행 (리포트 파일이 있는 프로젝트 루트를 PROJECT_DIR로 지정 가능):
    python ppt_data/plot_recall_three_models.py
    # 워크트리 등 리포트가 없는 곳에서 돌릴 때:
    PROJECT_DIR=D:/hn_old-building python ppt_data/plot_recall_three_models.py
결과 (이 파일과 같은 폴더에 저장):
    recall_three_models.png
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
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", BASE_DIR.parent))
OUTPUT_PATH = BASE_DIR / "recall_three_models.png"

# 좌 -> 우 순서 (그룹 라벨, 리포트 파일)
groups = [
    ("① MobileNetV2",
     PROJECT_DIR / "data" / "processed" / "test_result_finetuned.txt"),
    ("② EfficientNet-B0",
     PROJECT_DIR / "test_results" / "efficientnet_b0" / "epoch20"
     / "evaluate_finetuned_result.txt"),
    ("③ ResNet-18",
     PROJECT_DIR / "test_results" / "resnet18" / "r18_e20"
     / "evaluate_finetuned_result.txt"),
]

GRADES = ["우수", "보통", "불량"]
# 등급마다 고정 색 — 그룹이 달라도 같은 등급은 같은 색으로 비교되게 한다
# (2차 ppt 자료의 등급 색과 동일)
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

figure, axes = plt.subplots(figsize=(12.5, 6.5))

group_centers = []
for group_index, (group_label, report_path) in enumerate(groups):
    recalls = parse_recall(report_path)
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
    "등급별 Recall — MobileNetV2 vs EfficientNet-B0 vs ResNet-18 (finetuned, Test 5,558장)",
    fontsize=16, fontweight="bold", pad=14,
)
axes.set_ylabel("Recall  (실제 등급 중 맞힌 비율)", fontsize=13)
axes.set_ylim(0, 1.05)
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
print("저장 완료:", OUTPUT_PATH)
