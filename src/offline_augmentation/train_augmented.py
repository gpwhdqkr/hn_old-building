# ============================================================
# [사용법]
# offline augmentation 실험용 EfficientNet-B0 baseline 학습 스크립트 (1단계).
# ImageNet 사전학습 가중치를 불러와 특징 추출부는 전부 동결하고
# 마지막 분류층만 우수/보통/불량 3등급 분류로 학습합니다.
#
# 실행 (윈도우 PowerShell):
#   $env:RUN_NAME="aug3"; $env:NUM_EPOCHS="10"
#   python src/offline_augmentation/train_augmented.py
#
# [주의: PowerShell은 리눅스식 인라인 환경변수를 못 씁니다]
# 리눅스 문법(RUN_NAME=aug3 python ...)을 PowerShell에 그대로 치면 파서 오류가
# 납니다. 반드시 위처럼 $env: 로 먼저 지정하세요.
#
# [전제 조건]
# build_augmented_dataset.py를 같은 RUN_NAME으로 먼저 실행해 둘 것.
#
# [환경변수]
#   RUN_NAME          실험 이름 (기본 default) — 확장 CSV와 모델 파일명에 붙음
#   NUM_EPOCHS        학습 epoch 수 (기본 10)
#   BATCH_SIZE        배치 크기 (기본 128, VRAM 부족하면 낮추기)
#   CLASS_WEIGHT_MODE auto(기본) 또는 fixed
#       auto  : 학습 데이터의 실제 분포에서 가중치를 다시 계산
#       fixed : 기존 하드코딩 값 [3.79, 2.95, 0.42]를 그대로 사용
#
# [클래스 가중치를 자동 계산하는 이유]
# 기존 값 [3.79, 2.95, 0.42]는 '총 train 수 / (등급 수 x 해당 등급 수)' 공식과
# 정확히 일치합니다. 보통을 3배로 늘렸는데 2.95를 그대로 두면 데이터도 늘리고
# 손실도 더 크게 치는 이중 보정이 되므로, 실제 분포에서 다시 계산합니다.
# AUG_FACTOR=1(증강 없음)이면 이 계산이 [3.79, 2.95, 0.42]를 그대로 재현하므로
# 대조군은 기존 실험과 완전히 같은 조건이 됩니다.
# "데이터만 늘린 효과"를 따로 보고 싶으면 CLASS_WEIGHT_MODE=fixed로 돌리면 됩니다.
#
# 결과:
#   model/best_efficientnet_b0_aug_baseline_<RUN_NAME>.pth
#   (validation 정확도가 가장 높았던 시점의 모델이 저장됨 — 기존과 동일 기준)
#
# 다음 단계 (같은 RUN_NAME을 반드시 그대로 넘겨야 함):
#   python src/offline_augmentation/fine_tune_augmented.py
#
# [기존 train_efficientnet.py 와의 관계]
# - 모델 구성/동결 방식/옵티마이저(Adam, lr=0.001)/최고 모델 선택 기준
#   (validation 정확도) 전부 동일. 증강 효과만 분리해서 보려면 같아야 하므로
# - 다른 점: 확장 CSV를 읽고, 클래스 가중치를 자동 계산하고,
#   저장 파일명에 aug_ 가 붙어 기존 실험을 덮어쓰지 않음
# ============================================================

import os
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0
)

from preprocess_augmented import (
    batch_size,
    build_dataloaders,
    compute_class_weights,
    count_by_class
)

project_dir = Path(__file__).resolve().parent.parent.parent

# 실험 이름: RUN_NAME 환경변수로 지정 (미지정 시 "default")
# 이 값이 저장 파일명에 붙어서, 설정을 바꿔 여러 번 실험해도
# 이전 결과가 덮어써지지 않고 실험별로 따로 쌓임
RUN_NAME = os.environ.get("RUN_NAME", "default")

# 학습된 모델을 저장할 폴더 (없으면 생성)
model_dir = project_dir / "model"

# 가장 성능이 좋은 모델을 저장할 경로.
# 파일명에 aug_ 를 넣어 기존 EfficientNet 실험 결과와 섞이지 않게 함
best_model_path = model_dir / f"best_efficientnet_b0_aug_baseline_{RUN_NAME}.pth"

# 학습 epoch 수: NUM_EPOCHS 환경변수로 지정 (미지정 시 10)
num_epochs = int(os.environ.get("NUM_EPOCHS", 10))

# 클래스 가중치 결정 방식 (auto: 실제 분포에서 재계산, fixed: 기존 값 고정)
CLASS_WEIGHT_MODE = os.environ.get("CLASS_WEIGHT_MODE", "auto")

# CLASS_WEIGHT_MODE=fixed 일 때 쓰는 기존 하드코딩 값 (우수, 보통, 불량 순)
FIXED_CLASS_WEIGHT_VALUES = [3.79, 2.95, 0.42]

# 저장 시 함께 기록할 등급 이름
class_names = {
    0: "우수",
    1: "보통",
    2: "불량"
}


def evaluate_on_loader(model, data_loader, loss_function, device):
    """주어진 데이터로더 전체에 대해 손실/정확도/F1(참고용)을 계산한다."""
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    # macro F1 계산용: 전체 정답과 예측을 모아둠
    all_labels = []
    all_predictions = []

    # 평가에서는 모델을 수정하지 않으므로 기울기 계산 중단
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_function(outputs, labels)

            loss_sum += loss.item() * images.size(0)

            # 세 출력 중 가장 큰 위치를 예측 등급으로 선택
            predictions = outputs.argmax(dim=1)

            correct_count += (predictions == labels).sum().item()
            total_count += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = loss_sum / total_count
    accuracy = correct_count / total_count

    # 등급별 F1 (참고용 출력을 위해 계산. 순서: 우수, 보통, 불량)
    per_class_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    # 불량을 positive로 본 F1 (프로젝트 목표 지표 — 참고용 출력)
    # float() 변환 필수: numpy 타입을 체크포인트에 저장하면
    # torch.load(weights_only=True)에서 로드가 거부됨
    defect_f1 = float(per_class_f1[2])

    # 보통 F1도 함께 봄 (이번 증강 실험의 목표 지표)
    moderate_f1 = float(per_class_f1[1])

    return average_loss, accuracy, defect_f1, moderate_f1


def resolve_class_weight_values(train_data):
    """CLASS_WEIGHT_MODE에 따라 클래스 가중치를 결정한다."""
    if CLASS_WEIGHT_MODE == "fixed":
        return list(FIXED_CLASS_WEIGHT_VALUES)

    if CLASS_WEIGHT_MODE == "auto":
        return compute_class_weights(train_data)

    raise ValueError(
        f"CLASS_WEIGHT_MODE는 'auto' 또는 'fixed'여야 합니다 "
        f"(받은 값: {CLASS_WEIGHT_MODE})"
    )


def main():
    model_dir.mkdir(parents=True, exist_ok=True)

    print("실험 이름 (RUN_NAME) :", RUN_NAME)

    # GPU가 있으면 GPU, 없으면 CPU 사용
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("사용 장치 :", device)

    if device.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))

    train_loader, validation_loader, _ = build_dataloaders()

    train_data = train_loader.dataset.dataframe

    # 학습 조건을 눈으로 확인할 수 있게 출력.
    # 증강이 의도한 만큼 들어갔는지 여기서 바로 확인 가능
    class_count_by_label = count_by_class(train_data)

    print("\n==========================================")
    print("Train 장수 :", len(train_data))
    print(
        "Train 등급별 :",
        f"우수 {class_count_by_label[0]} /",
        f"보통 {class_count_by_label[1]} /",
        f"불량 {class_count_by_label[2]}"
    )
    print(
        "Train 원본/증강본 :",
        train_data["image_source"].value_counts().to_dict()
    )
    print("Validation 장수 :", len(validation_loader.dataset))
    print("배치 크기 :", batch_size)
    print("epoch 수 :", num_epochs)

    class_weight_values = resolve_class_weight_values(train_data)

    print(
        f"클래스 가중치 ({CLASS_WEIGHT_MODE}) :",
        class_weight_values
    )
    print("==========================================")

    # ImageNet으로 사전학습된 EfficientNet-B0 생성
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # 특징 추출부 전체 동결 (분류층만 학습하는 baseline)
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # 기존 마지막 분류층(1000 클래스)을 3등급 출력으로 교체
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        input_features,
        3
    )

    model = model.to(device)

    # 클래스 가중치를 적용한 손실 함수
    class_weights = torch.tensor(
        class_weight_values,
        dtype=torch.float32,
        device=device
    )
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # 분류층만 학습 (기존 baseline과 동일한 lr)
    optimizer = Adam(
        model.classifier.parameters(),
        lr=0.001
    )

    # 최고 성능 기록 (기존과 동일하게 validation 정확도 기준)
    best_validation_accuracy = -1.0

    training_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n==== Epoch {epoch}/{num_epochs} ====")

        # ---- Train ----
        model.train()

        train_loss_sum = 0.0
        train_correct_count = 0
        train_total_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 이전 배치의 기울기 초기화
            optimizer.zero_grad()

            # 예측 → 손실 → 기울기 → 가중치 수정
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            train_correct_count += (predictions == labels).sum().item()
            train_total_count += labels.size(0)

        train_loss = train_loss_sum / train_total_count
        train_accuracy = train_correct_count / train_total_count

        # ---- Validation ----
        (
            validation_loss,
            validation_accuracy,
            validation_defect_f1,
            validation_moderate_f1
        ) = evaluate_on_loader(
            model,
            validation_loader,
            loss_function,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2%}"
        )
        print(
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.2%} | "
            f"Validation 불량 F1 (참고): {validation_defect_f1:.4f} | "
            f"Validation 보통 F1 (참고): {validation_moderate_f1:.4f}"
        )

        # 기존과 동일하게 validation 정확도가 최고 기록을 넘으면 저장
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_defect_f1": validation_defect_f1,
                    "validation_moderate_f1": validation_moderate_f1,
                    "class_names": class_names,
                    "run_name": RUN_NAME,
                    # 실험 조건을 모델 파일에도 남겨 두면 나중에 결과를
                    # 비교할 때 어떤 조건으로 학습한 모델인지 확인할 수 있음
                    "class_weight_values": class_weight_values,
                    "class_weight_mode": CLASS_WEIGHT_MODE,
                    "train_class_counts": class_count_by_label,
                    "train_total_count": len(train_data)
                },
                best_model_path
            )

            print("최고 성능 모델 저장 :", best_model_path)

    training_seconds = time.time() - training_start_time

    print("\n==========================================")
    print(f"최고 Validation 정확도: {best_validation_accuracy:.2%}")
    print(f"학습 시간 : {training_seconds:.1f}초")
    print("저장된 모델 :", best_model_path)
    print("다음 단계 : python src/offline_augmentation/fine_tune_augmented.py")


if __name__ == "__main__":
    main()
