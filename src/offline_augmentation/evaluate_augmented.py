# ============================================================
# [사용법]
# offline augmentation 실험용 test 평가 스크립트 (3단계).
# 학습된 모델을 test 세트로 평가해 등급별 F1/정확도/혼동행렬을 냅니다.
#
# 실행 (윈도우 PowerShell):
#   $env:RUN_NAME="aug3"; $env:EVAL_TARGET="finetuned"
#   python src/offline_augmentation/evaluate_augmented.py
#
# [주의: PowerShell은 리눅스식 인라인 환경변수를 못 씁니다]
# 리눅스 문법(RUN_NAME=aug3 python ...)을 PowerShell에 그대로 치면 파서 오류가
# 납니다. 반드시 위처럼 $env: 로 먼저 지정하세요.
#
# [평가할 모델 선택 (EVAL_TARGET 환경변수, 미지정 시 finetuned)]
#   "finetuned" → model/best_efficientnet_b0_aug_finetuned_<RUN_NAME>.pth 평가
#   "baseline"  → model/best_efficientnet_b0_aug_baseline_<RUN_NAME>.pth 평가
#
# [test 세트는 증강하지 않습니다]
# 증강은 train split의 '보통'에만 적용했으므로 test는 5,558장 그대로입니다.
# 따라서 기존 실험 결과와 숫자를 직접 비교할 수 있습니다. 실제로 그런지
# 확인할 수 있게 결과 파일에 test 장수와 원본/증강본 구성을 함께 기록합니다.
#
# 결과:
#   test_results/efficientnet_b0_augmented/<RUN_NAME>/evaluate_<EVAL_TARGET>_result.txt
#   test_results/efficientnet_b0_augmented/<RUN_NAME>/test_confusion_matrix_<EVAL_TARGET>.csv
#
# [기존 결과와 비교하는 법]
# 증강 유/무를 같은 코드로 돌려 비교하는 것이 목적입니다.
#   대조군 : RUN_NAME=noaug, AUG_FACTOR=1 (증강 없음)
#   실험군 : RUN_NAME=aug3,  AUG_FACTOR=3
# 두 결과 txt의 '보통' F1/recall이 이번 실험의 목표 지표이고, '불량' F1이
# 떨어지지 않았는지도 함께 확인해야 합니다.
#
# 참고 (팀 기존 수치):
#   MobileNetV2 baseline    불량 F1 0.9088 / 보통 F1 0.3814 / Macro 0.6626
#   EfficientNet-B0 finetuned 불량 F1 0.9087 / 보통 F1 0.3907 / Macro 0.6589
# 대조군이 이 수치와 비슷하게 나와야 새 코드가 기존 파이프라인을 제대로
# 재현한 것입니다. 크게 벗어나면 새 코드를 먼저 의심하세요.
# ============================================================

import os
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
from torchvision.models import efficientnet_b0

from preprocess_augmented import (
    batch_size,
    build_dataloaders
)

# 평가할 모델 선택: EVAL_TARGET 환경변수로 지정 (미지정 시 "finetuned")
EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")

# 실험 이름: 학습 때와 반드시 같은 값을 써야 그 실험의 모델을 찾음
RUN_NAME = os.environ.get("RUN_NAME", "default")

project_dir = Path(__file__).resolve().parent.parent.parent

# 평가할 모델 경로 (aug_ 가 붙어 기존 실험 모델과 구분됨)
model_path = (
    project_dir / "model"
    / f"best_efficientnet_b0_aug_{EVALUATION_TARGET}_{RUN_NAME}.pth"
)

# 결과를 저장할 폴더. 기존 test_results/efficientnet_b0/ 와 구분하기 위해
# _augmented 를 붙여 팀 기존 결과를 덮어쓰지 않게 함
test_results_dir = (
    project_dir / "test_results" / "efficientnet_b0_augmented" / RUN_NAME
)

confusion_matrix_path = (
    test_results_dir / f"test_confusion_matrix_{EVALUATION_TARGET}.csv"
)

result_text_path = (
    test_results_dir / f"evaluate_{EVALUATION_TARGET}_result.txt"
)

# 등급 표기 (0=우수, 1=보통, 2=불량)
class_labels = [0, 1, 2]
class_names = ["우수", "보통", "불량"]


def main():
    if EVALUATION_TARGET not in ("baseline", "finetuned"):
        raise ValueError(
            f"EVAL_TARGET은 'baseline' 또는 'finetuned'여야 합니다 "
            f"(받은 값: {EVALUATION_TARGET})"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {model_path}\n"
            f"먼저 같은 RUN_NAME으로 학습을 실행하세요:\n"
            f'  $env:RUN_NAME="{RUN_NAME}"\n'
            f"  python src/offline_augmentation/train_augmented.py\n"
            f"  python src/offline_augmentation/fine_tune_augmented.py"
        )

    # 화면 출력과 결과 파일에 남길 내용을 한 번에 모으기 위한 장치
    # (tee 같은 도구 없이도 실행 기록이 그대로 파일에 남음)
    result_lines = []

    def record(text=""):
        """print와 동시에 결과 파일에 남길 내용으로 기록한다."""
        print(text)
        result_lines.append(str(text))

    record("==========================================")
    record("offline augmentation 실험 - Test 평가")
    record("==========================================")
    record(f"실험 이름 (RUN_NAME) : {RUN_NAME}")
    record(f"평가 대상 (EVAL_TARGET) : {EVALUATION_TARGET}")
    record(f"모델 파일 : {model_path.name}")

    # GPU가 있으면 GPU, 없으면 CPU 사용
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    record(f"사용 장치 : {device}")

    if device.type == "cuda":
        record(f"GPU : {torch.cuda.get_device_name(0)}")

    _, _, test_loader = build_dataloaders()

    test_data = test_loader.dataset.dataframe

    record("")
    record("===== 데이터 구성 =====")
    record(f"Test 장수 : {len(test_data)}")

    # 증강이 test로 새어 들어가지 않았는지 결과 파일에서 바로 확인할 수 있게 기록
    record(
        f"Test 원본/증강본 : "
        f"{test_data['image_source'].value_counts().to_dict()}"
    )
    record(f"Test 등급별 : {test_data['class_name'].value_counts().to_dict()}")

    augmented_in_test = int((test_data["image_source"] == "augmented").sum())

    if augmented_in_test > 0:
        raise ValueError(
            f"test 세트에 증강본이 {augmented_in_test}장 섞여 있습니다. "
            "평가가 오염되었으므로 확장 CSV를 다시 만들어야 합니다."
        )

    record("→ test에 증강본 없음 (평가 오염 없음, 기존 결과와 비교 가능)")

    # 학습 때와 같은 구조를 만들고 저장된 가중치를 덮어씀
    model = efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        input_features,
        3
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    record("")
    record("===== 학습 설정 (모델 파일에 기록된 값) =====")
    record(f"baseline epoch : {checkpoint.get('epoch', '?')}")

    if "fine_tune_epoch" in checkpoint:
        record(f"파인튜닝 epoch : {checkpoint['fine_tune_epoch']}")

    record(f"Validation 정확도 : {checkpoint.get('validation_accuracy', -1):.2%}")
    record(f"클래스 가중치 : {checkpoint.get('class_weight_values', '?')}")
    record(f"가중치 결정 방식 : {checkpoint.get('class_weight_mode', '?')}")
    record(f"학습에 쓴 Train 장수 : {checkpoint.get('train_total_count', '?')}")
    record(f"학습 Train 등급별 : {checkpoint.get('train_class_counts', '?')}")
    record(f"배치 크기 : {batch_size}")

    # ---- Test 추론 ----
    model.eval()

    all_labels = []
    all_predictions = []

    inference_start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    inference_seconds = time.time() - inference_start_time

    # ---- 지표 계산 ----
    correct_count = sum(
        1
        for label, prediction in zip(all_labels, all_predictions)
        if label == prediction
    )
    test_accuracy = correct_count / len(all_labels)

    per_class_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=class_labels,
        average=None,
        zero_division=0
    )

    # 불량을 positive로 본 F1 (프로젝트 핵심 지표)
    test_defect_f1 = per_class_f1[2]

    # 보통 F1 (이번 증강 실험의 목표 지표)
    test_moderate_f1 = per_class_f1[1]

    test_macro_f1 = per_class_f1.mean()

    record("")
    record("===== Test 평가 결과 =====")
    record(f"Test 정확도 : {test_accuracy:.2%}")
    record(f"불량 F1 (핵심 지표) : {test_defect_f1:.4f}")
    record(f"보통 F1 (증강 목표 지표) : {test_moderate_f1:.4f}")
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

    # ---- 추론 속도 ----
    # 멘토 피드백("모델의 추론속도를 모델 결과표에 추가")을 반영한 항목
    images_per_second = len(all_labels) / inference_seconds

    record("===== 추론 속도 =====")
    record(f"Test {len(all_labels)}장 추론 시간 : {inference_seconds:.1f}초")
    record(f"초당 처리 장수 : {images_per_second:.1f}장/초")
    record(f"이미지 1장당 : {inference_seconds / len(all_labels) * 1000:.2f}ms")

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

    test_results_dir.mkdir(parents=True, exist_ok=True)

    # utf-8-sig: 윈도우 메모장/엑셀에서도 한글이 깨지지 않게
    confusion_table.to_csv(
        confusion_matrix_path,
        encoding="utf-8-sig"
    )

    record("")
    record(f"혼동행렬 저장 : {confusion_matrix_path}")

    result_text_path.write_text(
        "\n".join(result_lines),
        encoding="utf-8-sig"
    )

    print(f"평가 결과 저장 : {result_text_path}")


if __name__ == "__main__":
    main()
