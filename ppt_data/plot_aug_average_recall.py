# -*- coding: utf-8 -*-
"""보통 Recall — offline augmentation 실험 비교 그룹 막대그래프.

EfficientNet-B0 증강 실험 5종(noaug / aug2 / aug2fixed / aug3 / aug3fixed)의
baseline·finetuned 보통 recall을 한 좌표평면에서 비교한다 (같은 test 5,558장).
noaug가 "증강 없음" 기준선이므로, 다른 막대가 noaug보다 높아야 증강 효과가 있는 것.

리포트의 '등급별 성능' 표에서 보통 행의 recall(2번째 수치)을 파싱한다.

실행 (리포트 파일이 있는 프로젝트 루트를 PROJECT_DIR로 지정 가능):
    python ppt_data/plot_aug_average_recall.py
결과 (이 파일과 같은 폴더에 저장):
    aug_average_recall.png
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
AUG_DIR = PROJECT_DIR / "test_results" / "efficientnet_b0_augmented"
OUTPUT_PATH = BASE_DIR / "aug_average_recall.png"

# 좌 -> 우 순서 (실험 라벨, 폴더 이름)
configs = [
    ("증강 없음\n(noaug)", "noaug"),
    ("aug2", "aug2"),
    ("aug2fixed", "aug2fixed"),
    ("aug3", "aug3"),
    ("aug3fixed", "aug3fixed"),
]

# baseline/finetuned 두 계열 색 (등급 색과 혼동되지 않게 회색 + 보통 주황)
SERIES_COLORS = {"baseline": "#9E9E9E", "finetuned": "#F4A261"}


def parse_average_recall(report_path):
    """리포트 '등급별 성능' 표에서 보통 recall 을 파싱한다."""
    text = report_path.read_text(encoding="utf-8")
    match = re.search(r"보통\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", text)
    if not match:
        raise ValueError(f"{report_path.name}: '보통' 행을 찾지 못했습니다")
    return float(match.group(2))  # recall = 2번째 수치


BAR_WIDTH = 0.32
OFFSETS = {"baseline": -BAR_WIDTH / 2, "finetuned": BAR_WIDTH / 2}

figure, axes = plt.subplots(figsize=(12, 6.5))

noaug_finetuned_recall = None
group_labels = []
for group_index, (config_label, folder_name) in enumerate(configs):
    group_labels.append(config_label)

    for series in ["baseline", "finetuned"]:
        report_path = AUG_DIR / folder_name / f"evaluate_{series}_result.txt"
        recall = parse_average_recall(report_path)

        if folder_name == "noaug" and series == "finetuned":
            noaug_finetuned_recall = recall

        x = group_index + OFFSETS[series]
        axes.bar(
            x, recall, width=BAR_WIDTH,
            color=SERIES_COLORS[series],
            edgecolor="white", linewidth=1.0,
        )
        # 막대 위 recall 값
        axes.text(
            x, recall + 0.008, f"{recall:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
        print(f"{folder_name:>10} {series:>10}  보통 recall {recall:.4f}")

# 증강 없음(noaug finetuned) 기준선 — 이 선을 넘어야 증강 효과가 있는 것
axes.axhline(
    noaug_finetuned_recall, color="#555555",
    linestyle="--", linewidth=1.2, alpha=0.8,
)
axes.text(
    len(configs) - 0.42, noaug_finetuned_recall + 0.006,
    f"증강 없음(finetuned) 기준선 {noaug_finetuned_recall:.4f}",
    ha="right", va="bottom", fontsize=10.5, color="#555555",
)

axes.set_title(
    "보통 Recall — Offline Augmentation 효과 비교 (EfficientNet-B0, Test 5,558장)",
    fontsize=16, fontweight="bold", pad=14,
)
axes.set_ylabel("보통 Recall  (실제 보통 중 맞힌 비율)", fontsize=13)
axes.set_ylim(0, 0.65)
axes.set_xticks(range(len(configs)))
axes.set_xticklabels(group_labels, fontsize=12.5)
axes.grid(axis="y", alpha=0.3, linestyle="--")
axes.set_axisbelow(True)

legend_handles = [
    Patch(facecolor=SERIES_COLORS["baseline"], label="baseline"),
    Patch(facecolor=SERIES_COLORS["finetuned"], label="finetuned"),
]
axes.legend(handles=legend_handles, fontsize=12, loc="upper left",
            bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)

figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(figure)
print("저장 완료:", OUTPUT_PATH)
