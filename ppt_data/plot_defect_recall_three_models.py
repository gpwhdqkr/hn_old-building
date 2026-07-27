# -*- coding: utf-8 -*-
"""불량 Recall — 세 모델 비교 막대그래프 (모두 finetuned, 같은 test 5,558장).

    ① MobileNetV2      <- data/processed/test_result_finetuned.txt
    ② EfficientNet-B0  <- test_results/efficientnet_b0/epoch20/evaluate_finetuned_result.txt
    ③ ResNet-18        <- test_results/resnet18/r18_e20/evaluate_finetuned_result.txt

리포트의 '등급별 성능' 표에서 불량 행의 recall(2번째 수치)을 파싱한다.

실행 (리포트 파일이 있는 프로젝트 루트를 PROJECT_DIR로 지정 가능):
    python ppt_data/plot_defect_recall_three_models.py
결과 (이 파일과 같은 폴더에 저장):
    defect_recall_three_models.png
"""
import os
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# 한글 라벨이 깨지지 않도록 윈도우 기본 한글 글꼴 사용, 음수 부호 깨짐 방지
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", BASE_DIR.parent))
OUTPUT_PATH = BASE_DIR / "defect_recall_three_models.png"

# 좌 -> 우 순서 (모델 라벨, 리포트 파일)
models = [
    ("① MobileNetV2",
     PROJECT_DIR / "data" / "processed" / "test_result_finetuned.txt"),
    ("② EfficientNet-B0",
     PROJECT_DIR / "test_results" / "efficientnet_b0" / "epoch20"
     / "evaluate_finetuned_result.txt"),
    ("③ ResNet-18",
     PROJECT_DIR / "test_results" / "resnet18" / "r18_e20"
     / "evaluate_finetuned_result.txt"),
]

# 등급 고정 색 중 불량 색 (다른 ppt 차트들과 동일)
DEFECT_COLOR = "#C1121F"


def parse_defect_recall(report_path):
    """리포트 '등급별 성능' 표에서 불량 recall 을 파싱한다."""
    text = report_path.read_text(encoding="utf-8")
    match = re.search(r"불량\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", text)
    if not match:
        raise ValueError(f"{report_path.name}: '불량' 행을 찾지 못했습니다")
    return float(match.group(2))  # recall = 2번째 수치


figure, axes = plt.subplots(figsize=(9, 6))

labels = []
for index, (model_label, report_path) in enumerate(models):
    recall = parse_defect_recall(report_path)
    labels.append(model_label)

    axes.bar(
        index, recall, width=0.5,
        color=DEFECT_COLOR, edgecolor="white", linewidth=1.0,
    )
    # 막대 위 recall 값
    axes.text(
        index, recall + 0.012, f"{recall:.4f}",
        ha="center", va="bottom", fontsize=14, fontweight="bold",
    )
    print(f"{model_label}  불량 recall {recall:.4f}")

axes.set_title(
    "불량 Recall — 모델별 비교 (finetuned, Test 5,558장)",
    fontsize=16, fontweight="bold", pad=14,
)
axes.set_ylabel("불량 Recall  (실제 불량 중 맞힌 비율)", fontsize=13)
axes.set_ylim(0, 1.05)
axes.set_xticks(range(len(models)))
axes.set_xticklabels(labels, fontsize=13)
axes.grid(axis="y", alpha=0.3, linestyle="--")
axes.set_axisbelow(True)

figure.tight_layout()
figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(figure)
print("저장 완료:", OUTPUT_PATH)
