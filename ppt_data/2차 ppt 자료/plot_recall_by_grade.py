# -*- coding: utf-8 -*-
"""등급별(우수·보통·불량) Recall 막대그래프 2개.

두 평가 리포트에서 등급별 recall 을 직접 파싱해 각각 막대그래프로 그린다.
    evaluate_finetuned_output.txt   -> EfficientNet-B0 (finetuned)
    test_result_baseline_김민권.txt -> MobileNetV2 (baseline)

리포트의 '등급별 성능' 표에서 recall(2번째 수치)을 읽는다. 예)
    우수     0.6422    0.7166    0.6774       501
             ^precision ^recall  ^f1        ^support

실행:
    python plot_recall_by_grade.py
결과 (같은 폴더에 저장):
    recall_by_grade_finetuned.png
    recall_by_grade_baseline.png
"""
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# 한글 라벨이 깨지지 않도록 윈도우 기본 한글 글꼴 사용, 음수 부호 깨짐 방지
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent

# (표시 이름, 리포트 파일, 출력 파일)
sources = [
    ("EfficientNet-B0 (finetuned)",
     BASE_DIR / "evaluate_finetuned_output.txt",
     BASE_DIR / "recall_by_grade_finetuned.png"),
    ("MobileNetV2 (baseline)",
     BASE_DIR / "test_result_baseline_김민권.txt",
     BASE_DIR / "recall_by_grade_baseline.png"),
]

GRADES = ["우수", "보통", "불량"]
# 등급마다 고정 색 — 두 그래프에서 같은 등급은 같은 색으로 비교되게 한다
GRADE_COLORS = {"우수": "#2E86AB", "보통": "#F4A261", "불량": "#C1121F"}


def parse_recall(report_path):
    """리포트 '등급별 성능' 표에서 등급별 recall 과 support 를 파싱한다."""
    text = report_path.read_text(encoding="utf-8")
    result = {}
    for grade in GRADES:
        # 등급명 뒤 precision recall f1-score support
        match = re.search(
            rf"{grade}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", text)
        if not match:
            raise ValueError(f"{report_path.name}: '{grade}' 행을 찾지 못했습니다")
        recall = float(match.group(2))
        support = int(match.group(4))
        result[grade] = (recall, support)
    return result


for display_name, report_path, output_path in sources:
    parsed = parse_recall(report_path)
    recalls = [parsed[grade][0] for grade in GRADES]
    supports = [parsed[grade][1] for grade in GRADES]
    colors = [GRADE_COLORS[grade] for grade in GRADES]

    print(f"[{display_name}]")
    for grade in GRADES:
        recall, support = parsed[grade]
        print(f"    {grade}  recall {recall:.4f}  (실제 {support:,}장)")

    figure, axes = plt.subplots(figsize=(8, 6))
    bars = axes.bar(
        GRADES, recalls, color=colors, width=0.6,
        edgecolor="white", linewidth=1.5,
    )

    # 막대 위: recall 값 / 막대 안: 실제 장수
    for bar, recall, support in zip(bars, recalls, supports):
        axes.text(
            bar.get_x() + bar.get_width() / 2, recall + 0.02,
            f"{recall:.4f}", ha="center", va="bottom",
            fontsize=16, fontweight="bold",
        )
        axes.text(
            bar.get_x() + bar.get_width() / 2, recall / 2,
            f"실제\n{support:,}장", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold", linespacing=1.4,
        )

    axes.set_title(
        f"등급별 Recall — {display_name}",
        fontsize=17, fontweight="bold", pad=14,
    )
    axes.set_ylabel("Recall  (실제 등급 중 맞힌 비율)", fontsize=13)
    axes.set_xlabel("실제 등급", fontsize=13)
    axes.set_ylim(0, 1.0)
    axes.tick_params(axis="x", labelsize=14)
    axes.grid(axis="y", alpha=0.3, linestyle="--")
    axes.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"    저장 완료: {output_path.name}\n")
