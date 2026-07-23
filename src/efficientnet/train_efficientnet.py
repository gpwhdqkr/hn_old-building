# ============================================================
# [사용법]
# EfficientNet-B0 baseline 학습 스크립트 (1단계).
# ImageNet 사전학습 가중치를 불러와 특징 추출부는 전부 동결하고
# 마지막 분류층만 우수/보통/불량 3등급 분류로 학습합니다.
#
# 실행:
#   python src/efficientnet/train_efficientnet.py
#
# [실험 이름으로 결과 구분하기]
# epoch 수 등 설정을 바꿔가며 여러 번 실험할 때, 이전 결과를 덮어쓰지 않고
# 파일명을 다르게 쌓고 싶으면 RUN_NAME 환경변수를 지정해서 실행하면 됨
# (코드 수정 없이 명령어 앞에 붙이기만 하면 됨 — 클라우드에서도 git pull만
#  받은 코드 그대로 사용 가능):
#   RUN_NAME=epoch20 python src/efficientnet/train_efficientnet.py
# RUN_NAME을 지정하지 않으면 기본값 "default"가 사용됨
#
# 결과:
#   model/best_efficientnet_b0_baseline_<RUN_NAME>.pth
#   (validation 정확도가 가장 높았던 시점의 모델이 저장됨 — 팀원과 동일 기준)
#
# 다음 단계 (같은 RUN_NAME을 반드시 그대로 넘겨야 함):
#   RUN_NAME=epoch20 python src/efficientnet/fine_tune_efficientnet.py
#
# [팀원 MobileNetV2와의 비교 조건]
# - 전처리/split/클래스 가중치/옵티마이저(Adam, lr=0.001) 동일
# - 최고 모델 선택 기준도 팀원과 동일: validation 정확도
# - 다른 점은 모델(EfficientNet-B0)과 배치 크기(128)뿐
#
# F1 스코어(불량 F1 등)는 참고용으로 매 epoch 출력되지만 모델 선택에는
# 쓰지 않음. 최종 F1 비교는 evaluate_efficientnet.py의 test 결과에서
# 팀원 evaluate_model.py의 같은 항목과 비교하면 됨.
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

from preprocess_efficientnet import build_dataloaders

project_dir = Path(__file__).resolve().parent.parent.parent

# 실험 이름: RUN_NAME 환경변수로 지정 (미지정 시 "default")
# 이 값이 저장 파일명에 붙어서, 설정을 바꿔 여러 번 실험해도
# 이전 결과가 덮어써지지 않고 실험별로 따로 쌓임
RUN_NAME = os.environ.get("RUN_NAME", "default")

# 학습된 모델을 저장할 폴더 (없으면 생성)
model_dir = project_dir / "model"

# 가장 성능이 좋은 모델을 저장할 경로 (실험 이름이 파일명에 포함됨)
best_model_path = model_dir / f"best_efficientnet_b0_baseline_{RUN_NAME}.pth"

# 학습 epoch 수
num_epochs = 10

# 클래스 불균형 보정 가중치 (팀원과 동일한 값: 우수, 보통, 불량 순)
# 데이터가 적은 등급의 손실을 크게 쳐서 불량 쪽으로 쏠리는 것을 막음
class_weight_values = [3.79, 2.95, 0.42]

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

    return average_loss, accuracy, defect_f1


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

    train_loader, validation_loader, _ = build_dataloaders()

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

    # 분류층만 학습 (팀원 baseline과 동일한 lr)
    optimizer = Adam(
        model.classifier.parameters(),
        lr=0.001
    )

    # 최고 성능 기록 (팀원과 동일하게 validation 정확도 기준)
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
        validation_loss, validation_accuracy, validation_defect_f1 = (
            evaluate_on_loader(
                model,
                validation_loader,
                loss_function,
                device
            )
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2%}"
        )
        print(
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.2%} | "
            f"Validation 불량 F1 (참고): {validation_defect_f1:.4f}"
        )

        # 팀원과 동일하게 validation 정확도가 최고 기록을 넘으면 저장
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_defect_f1": validation_defect_f1,
                    "class_names": class_names,
                    "run_name": RUN_NAME
                },
                best_model_path
            )

            print("최고 성능 모델 저장 :", best_model_path)

    training_seconds = time.time() - training_start_time

    print("\n==========================================")
    print(f"최고 Validation 정확도: {best_validation_accuracy:.2%}")
    print(f"학습 시간 : {training_seconds:.1f}초")
    print("저장된 모델 :", best_model_path)


if __name__ == "__main__":
    main()
