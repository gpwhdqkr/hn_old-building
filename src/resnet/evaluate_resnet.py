# ============================================================
# [사용법]
# ResNet 모델을 Test 데이터로 최종 평가하는 스크립트 (3단계).
# 등급별 F1/정확도/혼동행렬에 더해 **추론속도**를 함께 측정합니다.
# (멘토 피드백 "모델의 추론속도를 모델 결과표에 추가" 반영)
#
# 실행 (런팟 = 리눅스):
#   RESNET_ARCH=resnet18 RUN_NAME=r18_e20 python src/resnet/evaluate_resnet.py
#
# [주의: 윈도우 PowerShell은 리눅스식 인라인 환경변수를 못 씁니다]
#   $env:RESNET_ARCH="resnet18"; $env:RUN_NAME="r18_e20"; $env:EVAL_TARGET="finetuned"
#   python src/resnet/evaluate_resnet.py
#
# [환경변수]
#   RESNET_ARCH : resnet18 (기본) 또는 resnet50 — 학습 때와 같은 값
#   RUN_NAME    : 실험 이름 (기본 "default") — 학습 때와 같은 값
#   EVAL_TARGET : "finetuned" (기본) 또는 "baseline" — 어느 단계 모델을 평가할지
#   BATCH_SIZE  : 배치 크기 (기본 128) — 처리량 측정에도 이 값이 쓰임
#
# 결과:
#   test_results/<RESNET_ARCH>/<RUN_NAME>/evaluate_<EVAL_TARGET>_result.txt
#   test_results/<RESNET_ARCH>/<RUN_NAME>/test_confusion_matrix_<EVAL_TARGET>.csv
#   (화면에 나온 내용이 그대로 txt로도 저장되므로 tee가 필요 없음)
#
# [추론속도를 세 가지로 나눠 재는 이유]
# 기존 실험(evaluate_augmented.py)은 dataloader 루프 전체를 time.time()으로 감싼
# 수치 하나뿐이었는데, 거기엔 이미지 파일 읽기/전처리 시간이 섞여 있어서
# "모델이 빠른가"를 말해주지 못합니다. 그래서 아래 세 층으로 분리합니다.
#   1) 단일 이미지 지연  : 사진 한 장 넣었을 때 체감하는 시간 (실서비스 기준, 가장 중요)
#   2) 배치 처리량       : 대량 일괄 추론 시 초당 몇 장 (GPU 활용도 기준)
#   3) end-to-end        : dataloader 포함 전체 시간 (기존 실험과 비교 연속성 유지)
#
# [기존 결과와 비교하는 법]
# 팀원 evaluate_model.py / EfficientNet evaluate_efficientnet.py와 같은
# classification_report를 출력하므로 같은 항목끼리 비교하면 됩니다.
# - 핵심 지표: "불량" 행의 f1-score (프로젝트 목표 = 불량 탐지, 불량이 positive)
# - 참고 (팀 기존 수치):
#     MobileNetV2 baseline      불량 F1 0.9088 / 보통 F1 0.3814 / Macro 0.6626
#     EfficientNet-B0 finetuned 불량 F1 0.9087 / 보통 F1 0.3907 / Macro 0.6589
#   ResNet18이 이 근처에서 나와야 정상입니다. 크게 벗어나면 모델 성능 차이가 아니라
#   새 코드를 먼저 의심하세요.
# ============================================================

import os
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
from torch import nn
from torchvision.models import resnet18, resnet50

from preprocess_resnet import batch_size, build_dataloaders

# 학습할 ResNet 종류: 학습 때와 반드시 같은 값
RESNET_ARCH = os.environ.get("RESNET_ARCH", "resnet18")

# 평가할 모델 선택: "baseline" 또는 "finetuned" (미지정 시 "finetuned")
EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")

# 실험 이름: 학습 때와 반드시 같은 값을 써야 그 실험의 모델을 찾음
RUN_NAME = os.environ.get("RUN_NAME", "default")

project_dir = Path(__file__).resolve().parent.parent.parent

# 평가할 모델 경로
model_path = (
    project_dir / "model"
    / f"best_{RESNET_ARCH}_{EVALUATION_TARGET}_{RUN_NAME}.pth"
)

# 결과 저장 위치: test_results/<모델명>/<실험 이름>/ 아래
# (기존 test_results/efficientnet_b0/ 와 폴더가 나뉘어 서로 덮어쓰지 않음)
test_results_dir = (
    project_dir / "test_results" / RESNET_ARCH / RUN_NAME
)

confusion_matrix_path = (
    test_results_dir / f"test_confusion_matrix_{EVALUATION_TARGET}.csv"
)

result_text_path = (
    test_results_dir / f"evaluate_{EVALUATION_TARGET}_result.txt"
)

# 모델 출력 순서와 등급 이름
class_labels = [0, 1, 2]
class_names = ["우수", "보통", "불량"]

# ---- 추론속도 측정 설정 ----
# 워밍업: cuDNN 알고리즘 선택과 CUDA 컨텍스트 초기화가 첫 몇 번을 크게 부풀리므로
#         버리는 구간. 이걸 빼먹으면 먼저 측정한 모델만 느리게 나오는 함정에 빠진다.
WARMUP_ITERATIONS = 20

# 단일 이미지 지연 측정 반복 횟수 (중앙값과 p95를 뽑기 위해 넉넉히)
LATENCY_ITERATIONS = 100

# 배치 처리량 측정 반복 횟수
THROUGHPUT_ITERATIONS = 30


def create_resnet_model():
    """RESNET_ARCH에 맞는 ResNet 구조를 만들고 분류층을 3등급 출력으로 교체해서 반환한다.

    저장된 가중치를 덮어씌울 것이므로 ImageNet 가중치는 받지 않는다 (weights=None).
    train_resnet.py의 같은 이름 함수와 반드시 동일한 구조여야 로드가 성공한다.
    """
    if RESNET_ARCH == "resnet18":
        model = resnet18(weights=None)

    elif RESNET_ARCH == "resnet50":
        model = resnet50(weights=None)

    else:
        raise ValueError(
            'RESNET_ARCH는 "resnet18" 또는 "resnet50"이어야 합니다: '
            f"{RESNET_ARCH}"
        )

    model.fc = nn.Linear(
        model.fc.in_features,
        3
    )

    return model


def synchronize(device):
    """GPU 커널이 실제로 끝날 때까지 기다린다.

    CUDA 커널은 비동기로 실행되기 때문에, 동기화 없이 시간을 재면
    '커널을 제출하는 데 걸린 시간'만 재게 되어 숫자가 실제보다 훨씬 빠르게 나온다.
    추론속도 측정에서 이 호출을 빼먹는 것이 가장 흔한 실수다.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(sorted_values, ratio):
    """정렬된 값 목록에서 상위 ratio 지점의 값을 반환한다 (예: ratio=0.95 → p95)."""
    if not sorted_values:
        return float("nan")

    index = int(round(ratio * (len(sorted_values) - 1)))

    return sorted_values[index]


def measure_single_image_latency(model, device):
    """이미지 1장을 추론하는 데 걸리는 시간을 반복 측정해 중앙값/p95(ms)를 반환한다.

    실제 앱에서 사진 한 장을 넣었을 때 사용자가 체감하는 시간에 해당하는 값이라
    추론속도 지표 중 가장 중요하다.

    평균이 아니라 중앙값을 쓰는 이유: 클라우드 GPU는 다른 사용자의 부하 때문에
    가끔 크게 튀는 구간이 생기는데, 평균은 그 몇 번에 통째로 끌려간다.
    """
    single_image = torch.randn(
        1, 3, 224, 224,
        device=device
    )

    with torch.no_grad():
        # 워밍업 (측정에 포함하지 않음)
        for _ in range(WARMUP_ITERATIONS):
            model(single_image)

        synchronize(device)

        elapsed_milliseconds = []

        for _ in range(LATENCY_ITERATIONS):
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
    """배치 추론 시 초당 처리 장수를 측정해 반환한다.

    데이터로더를 거치지 않고 GPU에 이미 올라간 텐서로 forward만 반복하므로
    이미지 읽기/전처리 시간이 섞이지 않는다 (= 순수한 모델 속도).
    """
    image_batch = torch.randn(
        batch_size, 3, 224, 224,
        device=device
    )

    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            model(image_batch)

        synchronize(device)

        start_time = time.perf_counter()

        for _ in range(THROUGHPUT_ITERATIONS):
            model(image_batch)

        synchronize(device)

        elapsed_seconds = time.perf_counter() - start_time

    processed_image_count = batch_size * THROUGHPUT_ITERATIONS

    return processed_image_count / elapsed_seconds


def main():
    # 화면에 출력한 내용을 그대로 모아뒀다가 마지막에 파일로도 저장
    result_lines = []

    def record(text=""):
        """print와 동시에 결과 파일에 남길 내용으로 기록한다."""
        print(text)
        result_lines.append(str(text))

    if EVALUATION_TARGET not in ("baseline", "finetuned"):
        raise ValueError(
            'EVAL_TARGET은 "baseline" 또는 "finetuned"여야 합니다: '
            f"{EVALUATION_TARGET}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"평가할 모델을 찾을 수 없습니다: {model_path}\n"
            "먼저 같은 RESNET_ARCH와 RUN_NAME으로 학습을 실행하세요.\n"
            f"(예: RESNET_ARCH={RESNET_ARCH} RUN_NAME={RUN_NAME} "
            "python src/resnet/train_resnet.py)"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    record("===== 평가 대상 =====")
    record(f"모델 종류 (RESNET_ARCH) : {RESNET_ARCH}")
    record(f"실험 이름 (RUN_NAME) : {RUN_NAME}")
    record(f"평가 단계 (EVAL_TARGET) : {EVALUATION_TARGET}")
    record(f"모델 파일 : {model_path.name}")
    record(f"사용 장치 : {device}")

    if device.type == "cuda":
        record(f"GPU : {torch.cuda.get_device_name(0)}")

    record(f"torch 버전 : {torch.__version__}")

    _, _, test_loader = build_dataloaders()

    # 학습 때와 동일한 구조 생성 후 저장된 가중치 적용
    model = create_resnet_model()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    record("")
    record("===== 학습 설정 (모델 파일에 기록된 값) =====")
    record(f"baseline epoch : {checkpoint.get('epoch', '?')}")

    if "fine_tune_epoch" in checkpoint:
        record(f"파인튜닝 epoch : {checkpoint['fine_tune_epoch']}")

    saved_validation_accuracy = checkpoint.get("validation_accuracy")
    if isinstance(saved_validation_accuracy, float):
        record(f"저장 당시 Validation 정확도 : {saved_validation_accuracy:.2%}")

    record(f"클래스 가중치 : {checkpoint.get('class_weight_values', '?')}")
    record(f"동결 해제 범위 : {checkpoint.get('unfrozen_layers', '?')}")

    total_param_count = checkpoint.get("total_param_count")
    trainable_param_count = checkpoint.get("trainable_param_count")

    if total_param_count and trainable_param_count:
        record(
            f"파라미터 : 전체 {total_param_count / 1e6:.1f}M / "
            f"학습 {trainable_param_count / 1e6:.2f}M "
            f"({trainable_param_count / total_param_count:.0%})"
        )

    record(f"배치 크기 : {batch_size}")

    # ---- Test 추론 (end-to-end: dataloader 포함) ----
    all_labels = []
    all_predictions = []

    end_to_end_start_time = time.perf_counter()

    # Test에서는 모델을 수정하지 않으므로 기울기 계산 중단
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # 세 출력 중 가장 큰 위치를 예측 등급으로 선택
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    synchronize(device)

    end_to_end_seconds = time.perf_counter() - end_to_end_start_time

    # ---- 지표 계산 ----
    total_count = len(all_labels)

    correct_count = sum(
        1
        for label, prediction in zip(all_labels, all_predictions)
        if label == prediction
    )

    test_accuracy = correct_count / total_count

    # 등급별 F1 (순서: 우수, 보통, 불량)
    per_class_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=class_labels,
        average=None,
        zero_division=0
    )

    # 핵심 비교 지표: 불량을 positive로 본 F1 (프로젝트 목표 = 불량 탐지)
    test_defect_f1 = per_class_f1[2]
    test_moderate_f1 = per_class_f1[1]
    test_macro_f1 = per_class_f1.mean()

    record("")
    record("===== Test 평가 결과 =====")
    record(f"Test 이미지 수 : {total_count}")
    record(f"맞힌 이미지 수 : {correct_count}")
    record(f"Test 정확도 : {test_accuracy:.2%}")
    record(f"불량 F1 (핵심 지표) : {test_defect_f1:.4f}")
    record(f"보통 F1 : {test_moderate_f1:.4f}")
    record(f"우수 F1 : {per_class_f1[0]:.4f}")
    record(f"Macro F1 : {test_macro_f1:.4f}")

    record("")
    record("===== 등급별 상세 =====")
    record(
        classification_report(
            all_labels,
            all_predictions,
            labels=class_labels,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )

    # ---- 추론속도 ----
    # 멘토 피드백("모델의 추론속도를 모델 결과표에 추가")을 반영한 항목
    latency_median_ms, latency_p95_ms = measure_single_image_latency(
        model,
        device
    )

    throughput_images_per_second = measure_batch_throughput(model, device)

    record("===== 추론속도 =====")
    record(f"측정 장치 : {device}" + (
        f" ({torch.cuda.get_device_name(0)})"
        if device.type == "cuda"
        else ""
    ))
    record(f"입력 해상도 : 224x224")
    record("")
    record("[1] 단일 이미지 지연 (batch=1, 실서비스 체감 기준 — 가장 중요)")
    record(f"  중앙값 : {latency_median_ms:.2f}ms")
    record(f"  p95    : {latency_p95_ms:.2f}ms")
    record(f"  (워밍업 {WARMUP_ITERATIONS}회 후 {LATENCY_ITERATIONS}회 측정)")
    record("")
    record(f"[2] 배치 처리량 (batch={batch_size}, 대량 일괄 추론 기준)")
    record(f"  초당 처리 장수 : {throughput_images_per_second:.1f}장/초")
    record(f"  이미지 1장당   : {1000 / throughput_images_per_second:.2f}ms")
    record(f"  (데이터 로딩 제외, 순수 forward만)")
    record("")
    record("[3] end-to-end (데이터 로딩/전처리 포함, 기존 실험과 비교용)")
    record(f"  Test {total_count}장 전체 : {end_to_end_seconds:.1f}초")
    record(f"  초당 처리 장수 : {total_count / end_to_end_seconds:.1f}장/초")
    record(f"  이미지 1장당   : {end_to_end_seconds / total_count * 1000:.2f}ms")

    # ---- 혼동행렬 ----
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=class_labels
    )

    confusion_table = pd.DataFrame(
        matrix,
        index=["실제_우수", "실제_보통", "실제_불량"],
        columns=["예측_우수", "예측_보통", "예측_불량"]
    )

    record("")
    record("===== 혼동행렬 =====")
    record(confusion_table)

    # 보통을 불량으로 잘못 본 장수는 멘토가 지적한 병목 지점이라 따로 표시
    moderate_as_defect = int(matrix[1][2])
    moderate_total = int(matrix[1].sum())

    record("")
    record(
        f"보통 {moderate_total}장 중 {moderate_as_defect}장을 불량으로 오분류 "
        f"({moderate_as_defect / moderate_total:.1%})"
    )
    record("(기존 MobileNetV2는 620장 중 249장 = 40.2%)")

    # ---- 최종 요약 ----
    record("")
    record("==========================================")
    record(f"[비교용 최종 지표] {RESNET_ARCH} ({EVALUATION_TARGET})")
    record(f"Test 불량 F1 (핵심)   : {test_defect_f1:.4f}")
    record(f"Test Macro F1 (참고)  : {test_macro_f1:.4f}")
    record(f"Test Accuracy (참고)  : {test_accuracy:.2%}")
    record(f"단일 이미지 지연      : {latency_median_ms:.2f}ms (중앙값)")
    record(f"배치 처리량           : {throughput_images_per_second:.1f}장/초")
    record("==========================================")

    # 저장 폴더가 없으면 생성
    test_results_dir.mkdir(parents=True, exist_ok=True)

    # utf-8-sig: 윈도우 메모장/엑셀에서도 한글이 깨지지 않게
    confusion_table.to_csv(
        confusion_matrix_path,
        encoding="utf-8-sig"
    )

    result_text_path.write_text(
        "\n".join(result_lines),
        encoding="utf-8-sig"
    )

    print("")
    print("혼동행렬 저장 :", confusion_matrix_path)
    print("평가 결과 저장 :", result_text_path)


if __name__ == "__main__":
    main()
