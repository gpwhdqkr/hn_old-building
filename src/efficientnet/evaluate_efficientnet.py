# ============================================================
# [사용법]
# EfficientNet-B0 모델을 Test 데이터로 최종 평가하는 스크립트.
# 팀원의 MobileNetV2와 성능을 비교할 F1 스코어를 여기서 뽑습니다.
#
# 실행:
#   python src/efficientnet/evaluate_efficientnet.py
#
# 평가할 모델 선택 (EVAL_TARGET 환경변수, 미지정 시 finetuned):
#   - "finetuned" → model/best_efficientnet_b0_finetuned_<RUN_NAME>.pth 평가
#   - "baseline"  → model/best_efficientnet_b0_baseline_<RUN_NAME>.pth 평가
#   예) EVAL_TARGET=baseline python src/efficientnet/evaluate_efficientnet.py
#
# [실험 이름으로 결과 구분하기]
# train/fine_tune을 RUN_NAME을 지정해서 돌렸다면, 여기서도 같은 RUN_NAME을
# 넘겨야 그 실험의 모델을 찾아서 평가함:
#   RUN_NAME=epoch20 python src/efficientnet/evaluate_efficientnet.py
# RUN_NAME을 지정하지 않으면 기본값 "default"가 사용됨
#
# 결과:
#   - 화면에 Test 정확도, 등급별 precision/recall/F1, 혼동행렬 출력
#   - 화면에 나온 평가 결과 전문이 텍스트 파일로도 자동 저장됨 (tee 불필요):
#     test_results/efficientnet_b0/<RUN_NAME>/evaluate_<대상>_result.txt
#   - 혼동행렬 CSV 저장 (모델명/실험 이름 폴더 아래, 없으면 자동 생성):
#     test_results/efficientnet_b0/<RUN_NAME>/test_confusion_matrix_<대상>.csv
#
# [팀원과 비교하는 법]
# 팀원의 evaluate_model.py도 같은 classification_report를 출력하므로
# 같은 항목끼리 비교하면 됩니다. (같은 Test set이라 공정 비교)
# - 핵심 지표: "불량" 행의 f1-score (프로젝트 목표 = 불량 탐지, 불량이 positive)
# - 불량 행의 recall(실제 불량을 놓치지 않는 비율), precision(불량 판정의 신뢰도)도
#   함께 비교하면 좋음
# - 참고: macro avg 행 (세 등급 평균)
# ============================================================

import os
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
# EVAL_TARGET 환경변수로 지정 (미지정 시 "finetuned")
# 클라우드에서는 코드 수정 없이 실행 시 지정하면 됨:
#   EVAL_TARGET=baseline python src/efficientnet/evaluate_efficientnet.py
EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")

# 실험 이름: RUN_NAME 환경변수로 지정 (미지정 시 "default")
# train/fine_tune 실행 때와 반드시 같은 값을 써야 그 실험의 모델을 찾음
RUN_NAME = os.environ.get("RUN_NAME", "default")

project_dir = Path(__file__).resolve().parent.parent.parent

# 선택에 따라 평가할 모델 경로 결정 (실험 이름이 파일명에 포함됨)
model_path = (
    project_dir / "model"
    / f"best_efficientnet_b0_{EVALUATION_TARGET}_{RUN_NAME}.pth"
)

# 혼동행렬 CSV 저장 위치: test_results/<모델명>/<실험 이름>/ 아래
# (팀원의 test_results/Classical ML/ 과 같은 방식으로 모델별 폴더 분리,
#  실험 이름별로 하위 폴더를 나눠서 여러 번 평가해도 서로 안 겹침)
test_results_dir = (
    project_dir / "test_results" / "efficientnet_b0" / RUN_NAME
)

confusion_matrix_path = (
    test_results_dir
    / f"test_confusion_matrix_{EVALUATION_TARGET}.csv"
)

# 평가 결과 전문(화면 출력과 동일 내용)을 저장할 텍스트 파일 경로
result_text_path = (
    test_results_dir
    / f"evaluate_{EVALUATION_TARGET}_result.txt"
)

# 모델 출력 순서와 등급 이름
class_labels = [0, 1, 2]
class_names = ["우수", "보통", "불량"]


def main():
    # 화면에 출력한 내용을 그대로 모아뒀다가 마지막에 파일로도 저장
    result_lines = []

    def record(text=""):
        """print와 동시에 결과 파일에 남길 내용으로 기록한다."""
        print(text)
        result_lines.append(str(text))

    record(f"실험 이름 (RUN_NAME) : {RUN_NAME}")

    if EVALUATION_TARGET not in ("baseline", "finetuned"):
        raise ValueError(
            'EVALUATION_TARGET은 "baseline" 또는 "finetuned"여야 합니다: '
            f"{EVALUATION_TARGET}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"평가할 모델을 찾을 수 없습니다: {model_path}\n"
            "같은 RUN_NAME으로 train_efficientnet.py (와 fine_tune_efficientnet.py)"
            "를 먼저 실행하세요.\n"
            f"(예: RUN_NAME={RUN_NAME} python src/efficientnet/train_efficientnet.py)"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    record(f"사용 장치 : {device}")
    record(f"평가 모델 : {model_path.name}")

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

    record(f"저장된 Epoch : {checkpoint.get('epoch', '정보없음')}")

    if "fine_tune_epoch" in checkpoint:
        record(f"파인튜닝 Epoch : {checkpoint['fine_tune_epoch']}")

    saved_validation_accuracy = checkpoint.get("validation_accuracy")
    if isinstance(saved_validation_accuracy, float):
        record(
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

    record("\nTest 평가 결과")
    record(f"Test 이미지 수 : {total_count}")
    record(f"맞힌 이미지 수 : {correct_count}")
    record(f"Test 정확도 : {test_accuracy:.2%}")

    # 등급별 precision / recall / F1 상세 출력
    report = classification_report(
        all_labels,
        all_predictions,
        labels=class_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    record("\n등급별 성능")
    record(report)

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

    record("혼동행렬")
    record(confusion_table)

    # 저장 폴더가 없으면 생성 (test_results/efficientnet_b0/)
    test_results_dir.mkdir(parents=True, exist_ok=True)

    confusion_table.to_csv(
        confusion_matrix_path,
        encoding="utf-8-sig"
    )
    record(f"\n혼동행렬 저장 위치 : {confusion_matrix_path}")

    # MobileNetV2와 비교할 최종 수치
    # (팀원 evaluate_model.py 출력의 같은 항목과 비교하면 됨)
    record("\n==========================================")
    record(f"[비교용 최종 지표] EfficientNet-B0 ({EVALUATION_TARGET})")
    record(f"Test 불량 F1 (핵심)  : {test_defect_f1:.4f}")
    record(f"Test Macro F1 (참고) : {test_macro_f1:.4f}")
    record(f"Test Accuracy (참고) : {test_accuracy:.2%}")
    record("==========================================")

    # 화면에 출력한 평가 결과 전문을 텍스트 파일로 저장
    # (utf-8-sig: 윈도우 메모장에서도 한글이 깨지지 않게)
    result_text_path.write_text(
        "\n".join(result_lines),
        encoding="utf-8-sig"
    )
    print("평가 결과 저장 위치 :", result_text_path)


if __name__ == "__main__":
    main()
