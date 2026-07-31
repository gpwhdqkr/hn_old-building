# ============================================================
# [사용법]
# 학습해둔 모델들의 성능과 **추론속도**를 한 표로 비교하는 스크립트.
# 멘토 피드백("모델의 추론속도를 모델 결과표에 추가")에 대한 최종 산출물입니다.
#
# 실행 (런팟 = 리눅스):
#   RESNET18_RUN=r18_e20 RESNET50_RUN=r50_e20 EFFICIENTNET_RUN=default \
#     python src/resnet/benchmark_inference.py
#
# [주의: 윈도우 PowerShell은 리눅스식 인라인 환경변수를 못 씁니다]
#   $env:RESNET18_RUN="r18_e20"; $env:RESNET50_RUN="r50_e20"
#   python src/resnet/benchmark_inference.py
#
# [환경변수]
#   RESNET18_RUN     : resnet18 실험 이름 (기본 "default")
#   RESNET50_RUN     : resnet50 실험 이름 (기본 "default")
#   EFFICIENTNET_RUN : EfficientNet-B0 실험 이름 (기본 "default")
#   EVAL_TARGET      : "finetuned"(기본) 또는 "baseline" — 어느 단계 모델을 비교할지
#                      (MobileNetV2는 파일명에 실험 이름이 없어 이 값만 적용됨)
#   BATCH_SIZE       : 처리량 측정에 쓸 배치 크기 (기본 128)
#   RUN_NAME         : 결과 저장 폴더 이름 (기본 "default")
#   MEASURE_CPU      : "1"(기본)이면 CPU 지연도 측정, "0"이면 건너뜀
#
# 결과:
#   test_results/inference_benchmark/<RUN_NAME>/benchmark.txt
#   test_results/inference_benchmark/<RUN_NAME>/benchmark.csv
#
# [무엇을 재는가 — 세 층으로 나누는 이유]
# 기존 evaluate_augmented.py의 속도 측정은 dataloader 루프 전체를 감싼 수치 하나뿐이라
# 이미지 파일 읽기/전처리 시간이 섞여 있었습니다. 그건 "이 서버의 디스크가 빠른가"에
# 가까운 값이지 "이 모델이 빠른가"가 아닙니다. 그래서 아래처럼 분리합니다.
#   1) 단일 이미지 지연 : batch=1. 실제 앱에서 사진 한 장 넣었을 때 체감하는 시간.
#                         지표 중 가장 중요하며 중앙값과 p95를 함께 봅니다.
#   2) 배치 처리량      : GPU에 이미 올린 텐서로 forward만 반복. 순수 모델 속도.
#   3) CPU 지연         : 배포 환경이 CPU라면 이게 진짜 의사결정 근거입니다.
#                         런팟 GPU 수치는 배포 환경 수치가 아닙니다.
#
# [측정 신뢰성을 위해 지킨 것]
# - 워밍업 후 측정 : cuDNN 알고리즘 선택/CUDA 초기화가 첫 몇 번을 크게 부풀림
# - torch.cuda.synchronize() : CUDA는 비동기 실행이라 이걸 빼면 '커널 제출 시간'만
#                              재게 되어 숫자가 실제보다 훨씬 빠르게 나옴
# - 평균이 아닌 중앙값 : 클라우드는 다른 사용자 부하로 가끔 크게 튀는데 평균은 거기 끌려감
# - 모든 모델에 동일한 배치 크기/해상도/장치 적용
#
# [결과 해석 시 주의]
# 속도 숫자는 측정한 하드웨어에서만 유효합니다. 그래서 결과 파일에 GPU 모델명과
# torch/CUDA 버전을 같이 적습니다. 하드웨어가 안 적힌 속도 수치는 근거로 쓸 수 없습니다.
# ============================================================

import os
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torchvision.models import (
    efficientnet_b0,
    mobilenet_v2,
    resnet18,
    resnet50
)

from preprocess_resnet import batch_size, build_dataloaders

project_dir = Path(__file__).resolve().parent.parent.parent

model_dir = project_dir / "model"

# 비교할 학습 단계 (MobileNetV2는 파일명 규칙이 달라 아래 spec에서 따로 처리)
EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")

# 모델별 실험 이름 (모델마다 다른 RUN_NAME으로 학습했을 수 있으므로 따로 받음)
RESNET18_RUN = os.environ.get("RESNET18_RUN", "default")
RESNET50_RUN = os.environ.get("RESNET50_RUN", "default")
EFFICIENTNET_RUN = os.environ.get("EFFICIENTNET_RUN", "default")

# 결과 저장 폴더 이름
RUN_NAME = os.environ.get("RUN_NAME", "default")

# CPU 지연도 측정할지 여부
MEASURE_CPU = os.environ.get("MEASURE_CPU", "1") == "1"

results_dir = (
    project_dir / "test_results" / "inference_benchmark" / RUN_NAME
)

benchmark_text_path = results_dir / "benchmark.txt"
benchmark_csv_path = results_dir / "benchmark.csv"

class_labels = [0, 1, 2]

# ---- 측정 반복 횟수 ----
# 워밍업: 측정에 포함하지 않고 버리는 구간
WARMUP_ITERATIONS = 20
LATENCY_ITERATIONS = 100
THROUGHPUT_ITERATIONS = 30

# CPU는 GPU보다 훨씬 느려서 같은 횟수로 돌리면 시간이 오래 걸리므로 줄임
CPU_WARMUP_ITERATIONS = 5
CPU_LATENCY_ITERATIONS = 30


# ---- 모델별 구조 만들기 ----
# 각 모델은 분류층 이름이 달라서(fc vs classifier[1]) 따로 정의해야 한다.
# 학습 때와 똑같은 구조를 만들어야 저장된 가중치가 로드된다.

def build_mobilenet_v2():
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    return model


def build_efficientnet_b0():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    return model


def build_resnet18():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model


def build_resnet50():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model


# 비교 대상 목록. 파일이 없는 모델은 건너뛰되 그 사실을 결과에 남긴다.
# (조용히 빠지면 표를 보는 사람이 "측정 안 됨"을 "없음"으로 오해하게 됨)
model_specs = [
    {
        "label": "MobileNetV2 (팀원)",
        "filename": f"best_mobilenet_v2_{EVALUATION_TARGET}.pth",
        "builder": build_mobilenet_v2
    },
    {
        "label": "EfficientNet-B0",
        "filename": (
            f"best_efficientnet_b0_{EVALUATION_TARGET}_{EFFICIENTNET_RUN}.pth"
        ),
        "builder": build_efficientnet_b0
    },
    {
        "label": "ResNet18",
        "filename": f"best_resnet18_{EVALUATION_TARGET}_{RESNET18_RUN}.pth",
        "builder": build_resnet18
    },
    {
        "label": "ResNet50",
        "filename": f"best_resnet50_{EVALUATION_TARGET}_{RESNET50_RUN}.pth",
        "builder": build_resnet50
    }
]


def synchronize(device):
    """GPU 커널이 실제로 끝날 때까지 기다린다.

    CUDA 커널은 비동기로 실행되므로, 동기화 없이 시간을 재면 '커널을 제출하는 데
    걸린 시간'만 재게 되어 숫자가 실제보다 훨씬 빠르게 나온다.
    추론속도 측정에서 가장 흔한 실수라 모든 측정 구간에서 호출한다.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(sorted_values, ratio):
    """정렬된 값 목록에서 상위 ratio 지점의 값을 반환한다 (예: ratio=0.95 → p95)."""
    if not sorted_values:
        return float("nan")

    index = int(round(ratio * (len(sorted_values) - 1)))

    return sorted_values[index]


def measure_single_image_latency(
    model,
    device,
    warmup_iterations,
    measure_iterations
):
    """이미지 1장 추론 시간을 반복 측정해 (중앙값, p95) ms를 반환한다."""
    single_image = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(single_image)

        synchronize(device)

        elapsed_milliseconds = []

        for _ in range(measure_iterations):
            start_time = time.perf_counter()

            model(single_image)

            synchronize(device)

            elapsed_milliseconds.append(
                (time.perf_counter() - start_time) * 1000
            )

    elapsed_milliseconds.sort()

    return (
        statistics.median(elapsed_milliseconds),
        percentile(elapsed_milliseconds, 0.95)
    )


def measure_batch_throughput(model, device):
    """배치 추론 시 초당 처리 장수를 반환한다 (데이터 로딩 제외, 순수 forward)."""
    image_batch = torch.randn(batch_size, 3, 224, 224, device=device)

    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            model(image_batch)

        synchronize(device)

        start_time = time.perf_counter()

        for _ in range(THROUGHPUT_ITERATIONS):
            model(image_batch)

        synchronize(device)

        elapsed_seconds = time.perf_counter() - start_time

    return (batch_size * THROUGHPUT_ITERATIONS) / elapsed_seconds


def evaluate_on_test(model, test_loader, device):
    """test 세트 전체를 추론해 정확도/등급별 F1과 end-to-end 시간을 반환한다."""
    all_labels = []
    all_predictions = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    synchronize(device)

    end_to_end_seconds = time.perf_counter() - start_time

    correct_count = sum(
        1
        for label, prediction in zip(all_labels, all_predictions)
        if label == prediction
    )

    per_class_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=class_labels,
        average=None,
        zero_division=0
    )

    return {
        "test_count": len(all_labels),
        "accuracy": correct_count / len(all_labels),
        "excellent_f1": float(per_class_f1[0]),
        "moderate_f1": float(per_class_f1[1]),
        "defect_f1": float(per_class_f1[2]),
        "macro_f1": float(per_class_f1.mean()),
        "end_to_end_seconds": end_to_end_seconds
    }


def main():
    # 화면 출력과 결과 파일에 남길 내용을 한 번에 모음
    result_lines = []

    def record(text=""):
        print(text)
        result_lines.append(str(text))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    record("=" * 60)
    record("모델별 성능 / 추론속도 비교")
    record("=" * 60)
    record("")
    record("===== 측정 환경 =====")
    record(f"측정 장치 : {device}")

    if device.type == "cuda":
        record(f"GPU : {torch.cuda.get_device_name(0)}")
        record(f"CUDA 버전 : {torch.version.cuda}")
    else:
        record("GPU 없음 → CPU로 측정 (GPU 수치는 나오지 않음)")

    record(f"torch 버전 : {torch.__version__}")
    record(f"입력 해상도 : 224x224")
    record(f"배치 크기 (처리량 측정) : {batch_size}")
    record(f"비교 단계 (EVAL_TARGET) : {EVALUATION_TARGET}")
    record(
        f"워밍업 {WARMUP_ITERATIONS}회 후 "
        f"지연 {LATENCY_ITERATIONS}회 / 처리량 {THROUGHPUT_ITERATIONS}회 측정"
    )

    test_loader = None
    table_rows = []
    skipped_models = []

    for spec in model_specs:
        model_path = model_dir / spec["filename"]

        if not model_path.exists():
            skipped_models.append((spec["label"], spec["filename"]))
            continue

        record("")
        record("-" * 60)
        record(f"[{spec['label']}] {spec['filename']}")

        model = spec["builder"]()

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )
        file_size_megabytes = model_path.stat().st_size / (1024 * 1024)

        record(
            f"파라미터 : {parameter_count / 1e6:.1f}M / "
            f"파일 크기 : {file_size_megabytes:.1f}MB"
        )

        model = model.to(device)
        model.eval()

        # ---- test 세트 성능 (F1 등) ----
        # 데이터로더는 한 번만 만들어 모든 모델이 같은 것을 쓰게 한다
        # (전처리 조건이 모델마다 달라지면 비교가 성립하지 않음)
        if test_loader is None:
            _, _, test_loader = build_dataloaders()

        test_metrics = evaluate_on_test(model, test_loader, device)

        record(
            f"Test 정확도 : {test_metrics['accuracy']:.2%} | "
            f"불량 F1 : {test_metrics['defect_f1']:.4f} | "
            f"보통 F1 : {test_metrics['moderate_f1']:.4f} | "
            f"Macro F1 : {test_metrics['macro_f1']:.4f}"
        )

        # ---- 추론속도 ----
        latency_median_ms, latency_p95_ms = measure_single_image_latency(
            model,
            device,
            WARMUP_ITERATIONS,
            LATENCY_ITERATIONS
        )

        throughput_images_per_second = measure_batch_throughput(model, device)

        record(
            f"단일 이미지 지연 : {latency_median_ms:.2f}ms (중앙값) / "
            f"{latency_p95_ms:.2f}ms (p95)"
        )
        record(f"배치 처리량 : {throughput_images_per_second:.1f}장/초")
        record(
            f"end-to-end : {test_metrics['end_to_end_seconds']:.1f}초 "
            f"({test_metrics['test_count']}장, 데이터 로딩 포함)"
        )

        # ---- CPU 지연 ----
        # 배포 환경이 CPU라면 이 값이 실제 의사결정 근거가 된다
        cpu_latency_median_ms = None

        if MEASURE_CPU:
            if device.type == "cpu":
                # 이미 CPU로 측정했으므로 같은 값을 재사용 (중복 측정 불필요)
                cpu_latency_median_ms = latency_median_ms
            else:
                model = model.to("cpu")

                cpu_latency_median_ms, _ = measure_single_image_latency(
                    model,
                    torch.device("cpu"),
                    CPU_WARMUP_ITERATIONS,
                    CPU_LATENCY_ITERATIONS
                )

            record(f"CPU 단일 이미지 지연 : {cpu_latency_median_ms:.2f}ms (중앙값)")

        table_rows.append({
            "모델": spec["label"],
            "불량 F1": round(test_metrics["defect_f1"], 4),
            "보통 F1": round(test_metrics["moderate_f1"], 4),
            "우수 F1": round(test_metrics["excellent_f1"], 4),
            "Macro F1": round(test_metrics["macro_f1"], 4),
            "Accuracy": round(test_metrics["accuracy"], 4),
            "지연 중앙값(ms)": round(latency_median_ms, 2),
            "지연 p95(ms)": round(latency_p95_ms, 2),
            "처리량(장/초)": round(throughput_images_per_second, 1),
            "CPU 지연(ms)": (
                round(cpu_latency_median_ms, 2)
                if cpu_latency_median_ms is not None
                else None
            ),
            "params(M)": round(parameter_count / 1e6, 1),
            "크기(MB)": round(file_size_megabytes, 1)
        })

        # 다음 모델 측정에 영향이 없도록 GPU 메모리 정리
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    record("")
    record("=" * 60)

    if not table_rows:
        record("측정된 모델이 없습니다.")
        record("")
        record("model/ 폴더에서 아래 파일을 찾지 못했습니다:")

        for label, filename in skipped_models:
            record(f"  - {label} : {filename}")

        record("")
        record("먼저 학습을 실행하거나, 실험 이름 환경변수를 확인하세요.")
        record("  RESNET18_RUN / RESNET50_RUN / EFFICIENTNET_RUN")

        results_dir.mkdir(parents=True, exist_ok=True)
        benchmark_text_path.write_text(
            "\n".join(result_lines),
            encoding="utf-8-sig"
        )
        print("")
        print("결과 저장 :", benchmark_text_path)
        return

    comparison_table = pd.DataFrame(table_rows)

    record("===== 최종 비교표 =====")
    record(comparison_table.to_string(index=False))

    # 측정하지 못한 모델을 반드시 남긴다.
    # 조용히 빠지면 표가 "이게 전부"라고 거짓말을 하게 된다.
    if skipped_models:
        record("")
        record("===== 측정 제외 (모델 파일 없음) =====")

        for label, filename in skipped_models:
            record(f"  - {label} : model/{filename} 없음")

        record("(해당 모델을 학습했다면 실험 이름 환경변수를 확인하세요)")

    record("")
    record("===== 읽는 법 =====")
    record("- 불량 F1 : 프로젝트 핵심 지표 (불량 탐지가 목표, 불량이 positive)")
    record("- 지연 중앙값 : 사진 한 장 넣었을 때 체감 시간. 실서비스 기준 가장 중요")
    record("- 지연 p95 : 100번 중 느린 쪽 5번이 이 정도. 최악 응답시간 가늠용")
    record("- 처리량 : 대량 일괄 추론 시 초당 장수 (데이터 로딩 제외)")
    record("- CPU 지연 : 배포 환경이 CPU라면 이 값이 실제 근거")
    record("")
    record("속도 수치는 위에 적힌 측정 환경(GPU/torch 버전)에서만 유효합니다.")
    record("다른 하드웨어에서는 절대값이 달라지므로, 모델 간 상대 비교로 쓰세요.")

    results_dir.mkdir(parents=True, exist_ok=True)

    # utf-8-sig: 윈도우 메모장/엑셀에서도 한글이 깨지지 않게
    comparison_table.to_csv(
        benchmark_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    benchmark_text_path.write_text(
        "\n".join(result_lines),
        encoding="utf-8-sig"
    )

    print("")
    print("비교표 저장 :", benchmark_csv_path)
    print("결과 저장 :", benchmark_text_path)


if __name__ == "__main__":
    main()
