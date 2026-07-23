# ============================================================
# [사용법]
# EfficientNet-B0 모델을 Test 데이터로 최종 평가하는 스크립트.
# 팀원의 MobileNetV2와 성능을 비교할 F1 스코어를 여기서 뽑습니다.
#
# 실행:
#   python src/efficientnet/evaluate_efficientnet.py
#
# 평가할 모델 선택:
#   아래 EVALUATION_TARGET 값을 바꾸면 됨
#   - "baseline"  → model/best_efficientnet_b0_baseline.pth 평가
#   - "finetuned" → model/best_efficientnet_b0_finetuned.pth 평가
#
# 결과:
#   - 화면에 Test 정확도, 등급별 precision/recall/F1, 혼동행렬 출력
#   - 혼동행렬 CSV 저장:
#     data/processed/test_confusion_matrix_efficientnet_<대상>.csv
#
# [팀원과 비교하는 법]
# 팀원의 evaluate_model.py도 같은 classification_report를 출력하므로
# 같은 항목끼리 비교하면 됩니다. (같은 Test set이라 공정 비교)
# - 핵심 지표: "불량" 행의 f1-score (프로젝트 목표 = 불량 탐지, 불량이 positive)
# - 불량 행의 recall(실제 불량을 놓치지 않는 비율), precision(불량 판정의 신뢰도)도
#   함께 비교하면 좋음
# - 참고: macro avg 행 (세 등급 평균)
# ============================================================

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

from preprocess_efficientnet import build_dataloaders

# ===== 평가할 모델 선택: "baseline" 또는 "finetuned" =====
EVALUATION_TARGET = "finetuned"

project_dir = Path(__file__).resolve().parent.parent.parent

# 선택에 따라 평가할 모델 경로 결정
model_path = (
    project_dir / "model"
    / f"best_efficientnet_b0_{EVALUATION_TARGET}.pth"
)

# 혼동행렬 CSV 저장 위치
confusion_matrix_path = (
    project_dir / "data" / "processed"
    / f"test_confusion_matrix_efficientnet_{EVALUATION_TARGET}.csv"
)

# 모델 출력 순서와 등급 이름
class_labels = [0, 1, 2]
class_names = ["우수", "보통", "불량"]


def main():
    if EVALUATION_TARGET not in ("baseline", "finetuned"):
        raise ValueError(
            'EVALUATION_TARGET은 "baseline" 또는 "finetuned"여야 합니다: '
            f"{EVALUATION_TARGET}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"평가할 모델을 찾을 수 없습니다: {model_path}\n"
            "train_efficientnet.py (와 fine_tune_efficientnet.py)를 먼저 실행하세요."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("사용 장치 :", device)
    print("평가 모델 :", model_path.name)

    _, _, test_loader = build_dataloaders()

    # 학습 때와 동일한 구조 생성
    # (저장한 가중치를 불러올 것이므로 사전학습 가중치는 다시 받지 않음)
    model = efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        input_features,
        3
    )

    # 저장된 체크포인트 불러오기
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("저장된 Epoch :", checkpoint.get("epoch", "정보없음"))

    if "fine_tune_epoch" in checkpoint:
        print("파인튜닝 Epoch :", checkpoint["fine_tune_epoch"])

    saved_validation_accuracy = checkpoint.get("validation_accuracy")
    if isinstance(saved_validation_accuracy, float):
        print(
            f"저장 당시 Validation 정확도: {saved_validation_accuracy:.2%}"
        )

    model = model.to(device)
    model.eval()

    # 실제 정답과 모델 예측을 전부 모음
    all_labels = []
    all_predictions = []

    correct_count = 0
    total_count = 0

    # Test에서는 모델을 수정하지 않으므로 기울기 계산 중단
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # 세 출력 중 가장 큰 위치를 예측 등급으로 선택
            predictions = outputs.argmax(dim=1)

            correct_count += (predictions == labels).sum().item()
            total_count += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

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

    # 참고 지표: 세 등급 F1의 단순 평균
    test_macro_f1 = per_class_f1.mean()

    print("\nTest 평가 결과")
    print("Test 이미지 수 :", total_count)
    print("맞힌 이미지 수 :", correct_count)
    print(f"Test 정확도 : {test_accuracy:.2%}")

    # 등급별 precision / recall / F1 상세 출력
    report = classification_report(
        all_labels,
        all_predictions,
        labels=class_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\n등급별 성능")
    print(report)

    # 혼동행렬: 실제 등급별로 어떤 등급으로 예측했는지 개수 표
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

    print("혼동행렬")
    print(confusion_table)

    confusion_table.to_csv(
        confusion_matrix_path,
        encoding="utf-8-sig"
    )
    print("\n혼동행렬 저장 위치 :", confusion_matrix_path)

    # MobileNetV2와 비교할 최종 수치
    # (팀원 evaluate_model.py 출력의 같은 항목과 비교하면 됨)
    print("\n==========================================")
    print(f"[비교용 최종 지표] EfficientNet-B0 ({EVALUATION_TARGET})")
    print(f"Test 불량 F1 (핵심)  : {test_defect_f1:.4f}")
    print(f"Test Macro F1 (참고) : {test_macro_f1:.4f}")
    print(f"Test Accuracy (참고) : {test_accuracy:.2%}")
    print("==========================================")


if __name__ == "__main__":
    main()
