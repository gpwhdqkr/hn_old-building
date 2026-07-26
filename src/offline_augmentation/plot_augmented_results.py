# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# offline augmentation 실험 결과를 우수/보통/불량 등급별 막대그래프로 그립니다.
# 증강 없는 대조군과 증강 실험군을 나란히 붙여 비교합니다.
#
# 실행 (윈도우 PowerShell):
#   python src/offline_augmentation/plot_augmented_results.py
#
# 비교할 실험을 바꾸고 싶으면:
#   $env:PLOT_RUNS="noaug,aug3,aug5"
#   python src/offline_augmentation/plot_augmented_results.py
#
# [환경변수]
#   PLOT_RUNS   비교할 RUN_NAME 목록, 쉼표로 구분 (기본 "noaug,aug3")
#   EVAL_TARGET baseline 또는 finetuned (기본 finetuned)
#
# [전제 조건]
# evaluate_augmented.py를 각 RUN_NAME으로 먼저 실행해서 혼동행렬 CSV가 있어야 함:
#   test_results/efficientnet_b0_augmented/<RUN_NAME>/
#       test_confusion_matrix_<EVAL_TARGET>.csv
#
# [수치를 코드에 적어두지 않는 이유]
# F1/recall/precision을 코드에 하드코딩하면 실험을 다시 돌렸을 때 그래프가
# 옛 숫자를 그리게 됩니다. 그래서 혼동행렬 CSV에서 직접 읽어 계산합니다
# (ppt_data의 기존 plot 스크립트들과 같은 방식).
#
# 결과:
#   test_results/efficientnet_b0_augmented/
#       augmentation_comparison_<EVAL_TARGET>.png
# ============================================================

import os
from pathlib import Path

import matplotlib
import pandas as pd

# 화면 없이 파일로만 저장하므로 GUI 백엔드를 쓰지 않음
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# 한글이 네모칸으로 깨지지 않게 윈도우 기본 한글 폰트를 지정
# (axes.unicode_minus를 끄지 않으면 음수 부호가 깨짐)
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

project_dir = Path(__file__).resolve().parent.parent.parent

# 비교할 실험 목록 (쉼표로 구분)
PLOT_RUNS = os.environ.get("PLOT_RUNS", "noaug,aug3")

EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")

results_root = project_dir / "test_results" / "efficientnet_b0_augmented"

output_path = results_root / f"augmentation_comparison_{EVALUATION_TARGET}.png"

# 등급 이름 (혼동행렬의 행/열 순서와 같아야 함)
class_names = ["우수", "보통", "불량"]

# 실험 이름을 그래프에 어떻게 표시할지 (없는 이름은 그대로 씀)
RUN_LABELS = {
    "noaug": "증강 없음 (대조군)",
    "aug2": "2배 증강 (가중치 재계산)",
    "aug2fixed": "2배 증강 (가중치 유지)",
    "aug3": "3배 증강 (가중치 재계산)",
    "aug3fixed": "3배 증강 (가중치 유지)",
    "aug5": "5배 증강",
    "aug7": "7배 증강"
}

# 막대 색 (대조군은 회색, 실험군은 보라 계열 — 어느 쪽이 실험인지 한눈에 보이게)
RUN_COLORS = ["#9CA3AF", "#6D28D9", "#F59E0B", "#0EA5E9", "#10B981"]


def load_confusion_matrix(run_name):
    """해당 실험의 혼동행렬 CSV를 읽어 3x3 숫자 목록으로 반환한다."""
    matrix_path = (
        results_root / run_name
        / f"test_confusion_matrix_{EVALUATION_TARGET}.csv"
    )

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"혼동행렬 CSV를 찾을 수 없습니다: {matrix_path}\n"
            f"먼저 해당 실험을 평가하세요:\n"
            f'  $env:RUN_NAME="{run_name}"; '
            f'$env:EVAL_TARGET="{EVALUATION_TARGET}"\n'
            f"  python src/offline_augmentation/evaluate_augmented.py"
        )

    # 첫 열이 '실제_우수' 같은 인덱스이므로 index_col=0
    confusion_table = pd.read_csv(
        matrix_path,
        index_col=0,
        encoding="utf-8-sig"
    )

    return confusion_table.to_numpy()


def compute_metrics(matrix):
    """혼동행렬에서 등급별 precision/recall/f1을 계산한다.

    혼동행렬은 행이 실제 등급, 열이 예측 등급이다.
        recall    = 맞게 맞춘 수 / 실제 그 등급인 수 (해당 행의 합)
        precision = 맞게 맞춘 수 / 그 등급이라고 예측한 수 (해당 열의 합)
    """
    metrics = {
        "precision": [],
        "recall": [],
        "f1": []
    }

    for class_index in range(len(class_names)):
        correct_count = float(matrix[class_index][class_index])

        actual_count = float(matrix[class_index].sum())
        predicted_count = float(matrix[:, class_index].sum())

        # 분모가 0이면 0으로 둠 (한 번도 그 등급으로 예측하지 않은 경우)
        recall = correct_count / actual_count if actual_count > 0 else 0.0
        precision = (
            correct_count / predicted_count
            if predicted_count > 0
            else 0.0
        )

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)

    return metrics


def draw_metric_panel(axes, metric_name, metric_title, run_names, metrics_by_run):
    """한 지표(F1/recall/precision)에 대한 등급별 묶음 막대그래프를 그린다."""
    class_count = len(class_names)
    run_count = len(run_names)

    # 등급 하나당 막대 run_count개를 나란히 놓기 위한 위치 계산
    bar_width = 0.8 / run_count
    class_positions = range(class_count)

    for run_index, run_name in enumerate(run_names):
        values = metrics_by_run[run_name][metric_name]

        # 묶음의 가운데가 등급 위치에 오도록 좌우로 밀어줌
        offset = (run_index - (run_count - 1) / 2) * bar_width

        bar_positions = [
            class_position + offset
            for class_position in class_positions
        ]

        bars = axes.bar(
            bar_positions,
            values,
            width=bar_width,
            label=RUN_LABELS.get(run_name, run_name),
            color=RUN_COLORS[run_index % len(RUN_COLORS)],
            edgecolor="white",
            linewidth=0.8
        )

        # 막대 위에 숫자를 적어 그래프만 봐도 값을 읽을 수 있게 함
        for bar, value in zip(bars, values):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    axes.set_title(metric_title, fontsize=13, pad=12)
    axes.set_xticks(list(class_positions))
    axes.set_xticklabels(class_names, fontsize=11)
    axes.set_ylim(0, 1.08)

    # 가로 눈금선만 남기고 테두리를 정리해 숫자가 잘 보이게
    axes.grid(axis="y", linestyle="--", alpha=0.35)
    axes.set_axisbelow(True)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)


def main():
    run_names = [
        run_name.strip()
        for run_name in PLOT_RUNS.split(",")
        if run_name.strip()
    ]

    if not run_names:
        raise ValueError(
            f"비교할 실험이 없습니다 (PLOT_RUNS={PLOT_RUNS!r})"
        )

    print("비교할 실험 :", run_names)
    print("평가 대상 :", EVALUATION_TARGET)

    metrics_by_run = {}

    for run_name in run_names:
        matrix = load_confusion_matrix(run_name)
        metrics_by_run[run_name] = compute_metrics(matrix)

        print(f"\n[{run_name}]")

        for class_index, class_name in enumerate(class_names):
            print(
                f"  {class_name} : "
                f"F1 {metrics_by_run[run_name]['f1'][class_index]:.4f} / "
                f"recall {metrics_by_run[run_name]['recall'][class_index]:.4f} / "
                f"precision {metrics_by_run[run_name]['precision'][class_index]:.4f}"
            )

    figure, axes_list = plt.subplots(
        1,
        3,
        figsize=(16, 5.5)
    )

    panels = [
        ("f1", "F1 점수"),
        ("recall", "Recall (재현율)"),
        ("precision", "Precision (정밀도)")
    ]

    for axes, (metric_name, metric_title) in zip(axes_list, panels):
        draw_metric_panel(
            axes,
            metric_name,
            metric_title,
            run_names,
            metrics_by_run
        )

    # 범례는 그림 전체에 하나만 (패널마다 반복되면 지저분해짐)
    handles, labels = axes_list[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(run_names),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.02)
    )

    figure.suptitle(
        f"offline augmentation 등급별 성능 비교 ({EVALUATION_TARGET})",
        fontsize=15
    )

    figure.tight_layout(rect=(0, 0.05, 1, 0.96))

    results_root.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print("\n==========================================")
    print("그래프 저장 :", output_path)


if __name__ == "__main__":
    main()
