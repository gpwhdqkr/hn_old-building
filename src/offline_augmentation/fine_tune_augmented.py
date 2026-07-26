# ============================================================
# [사용법]
# offline augmentation 실험용 EfficientNet-B0 파인튜닝 스크립트 (2단계).
# 1단계에서 만든 baseline 모델을 불러와, 특징 추출부의 마지막 블록까지
# 함께 미세 조정합니다.
#
# 실행 (윈도우 PowerShell):
#   $env:RUN_NAME="aug3"; $env:NUM_EPOCHS="10"
#   python src/offline_augmentation/fine_tune_augmented.py
#
# [주의: PowerShell은 리눅스식 인라인 환경변수를 못 씁니다]
# 리눅스 문법(RUN_NAME=aug3 python ...)을 PowerShell에 그대로 치면 파서 오류가
# 납니다. 반드시 위처럼 $env: 로 먼저 지정하세요.
#
# [전제 조건]
# train_augmented.py를 같은 RUN_NAME으로 먼저 실행해서
# model/best_efficientnet_b0_aug_baseline_<RUN_NAME>.pth 가 있어야 함.
#
# [epoch 수와 조기 종료]
# NUM_EPOCHS=10으로 돌리지만 patience=3 조기 종료가 살아 있어서, validation
# 정확도가 3 epoch 연속 개선되지 않으면 10회를 다 채우지 않고 멈춥니다.
# 기존 코드와 같은 동작이고 대조군/실험군에 똑같이 적용되므로 비교에는 문제가
# 없지만, 두 실험의 실제 epoch 수가 다르게 나올 수 있으니 결과를 비교할 때는
# 체크포인트의 fine_tune_epoch 값을 함께 보세요.
#
# 결과:
#   model/best_efficientnet_b0_aug_finetuned_<RUN_NAME>.pth
#
# 다음 단계 (같은 RUN_NAME을 반드시 그대로 넘겨야 함):
#   $env:EVAL_TARGET="finetuned"
#   python src/offline_augmentation/evaluate_augmented.py
#
# [기존 fine_tune_efficientnet.py 와의 관계]
# - 동결 방식(features 전체 동결 후 마지막 블록만 해제), 차등 학습률,
#   BatchNorm 고정 기법, patience=3 조기 종료, baseline 정확도를 시작 기준으로
#   삼는 방식까지 전부 동일. 증강 효과만 분리해서 보려면 같아야 하므로
# - 다른 점: 확장 CSV를 읽고, 클래스 가중치는 baseline 체크포인트에 기록된
#   값을 그대로 이어 씀 (1단계와 2단계의 손실 기준이 달라지면 안 되므로)
# ============================================================

import os
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torchvision.models import efficientnet_b0

from preprocess_augmented import (
    batch_size,
    build_dataloaders,
    count_by_class
)

project_dir = Path(__file__).resolve().parent.parent.parent

# 실험 이름: train_augmented.py 실행 때와 같은 값을 써야
# 그 실험의 baseline 모델을 찾음
RUN_NAME = os.environ.get("RUN_NAME", "default")

model_dir = project_dir / "model"

# 1단계에서 학습한 baseline 모델 (불러오기)
baseline_model_path = (
    model_dir / f"best_efficientnet_b0_aug_baseline_{RUN_NAME}.pth"
)

# 파인튜닝 결과를 저장할 경로
fine_tuned_model_path = (
    model_dir / f"best_efficientnet_b0_aug_finetuned_{RUN_NAME}.pth"
)

# 파인튜닝 epoch 수: NUM_EPOCHS 환경변수로 지정 (미지정 시 10)
num_epochs = int(os.environ.get("NUM_EPOCHS", 10))

# validation 성능이 이 횟수만큼 연속으로 개선되지 않으면 조기 종료
patience = 3

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


def main():
    print("실험 이름 (RUN_NAME) :", RUN_NAME)

    if not baseline_model_path.exists():
        raise FileNotFoundError(
            f"baseline 모델을 찾을 수 없습니다: {baseline_model_path}\n"
            f"먼저 같은 RUN_NAME으로 1단계를 실행하세요:\n"
            f'  $env:RUN_NAME="{RUN_NAME}"\n'
            f"  python src/offline_augmentation/train_augmented.py"
        )

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

    class_count_by_label = count_by_class(train_data)

    # ImageNet 가중치를 다시 받지 않고 baseline 체크포인트로 덮어씀
    model = efficientnet_b0(weights=None)

    # baseline과 같은 구조가 되도록 분류층을 먼저 교체
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        input_features,
        3
    )

    # weights_only=True: 체크포인트에 임의 객체가 들어 있으면 로드를 거부함
    # (그래서 저장할 때 numpy 타입을 넣지 않도록 float()로 변환해 둠)
    checkpoint = torch.load(
        baseline_model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    baseline_validation_accuracy = checkpoint.get("validation_accuracy", -1.0)

    # 1단계에서 쓴 클래스 가중치를 그대로 이어 씀.
    # 1단계와 2단계의 손실 기준이 달라지면 baseline 정확도를 시작 기준으로
    # 삼는 아래 로직이 의미를 잃으므로 반드시 같은 값을 써야 함
    class_weight_values = checkpoint.get("class_weight_values")

    if class_weight_values is None:
        raise ValueError(
            "baseline 체크포인트에 class_weight_values가 없습니다. "
            "train_augmented.py로 만든 모델인지 확인하세요."
        )

    print("\n==========================================")
    print("불러온 baseline :", baseline_model_path.name)
    print(f"baseline Validation 정확도 : {baseline_validation_accuracy:.2%}")
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
    print("배치 크기 :", batch_size)
    print("epoch 수 :", num_epochs, f"(patience={patience} 조기 종료 있음)")
    print(
        f"클래스 가중치 ({checkpoint.get('class_weight_mode', '?')}, baseline 계승) :",
        class_weight_values
    )
    print("==========================================")

    # 특징 추출부 전체를 동결한 뒤 마지막 블록만 학습 대상으로 되돌림
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-1].parameters():
        parameter.requires_grad = True

    # 클래스 가중치를 적용한 손실 함수 (1단계와 동일한 값)
    class_weights = torch.tensor(
        class_weight_values,
        dtype=torch.float32,
        device=device
    )
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # 차등 학습률: 사전학습 특징은 조심스럽게, 분류층은 조금 더 크게
    optimizer = Adam([
        {
            "params": model.features[-1].parameters(),
            "lr": 0.00001
        },
        {
            "params": model.classifier.parameters(),
            "lr": 0.0001
        }
    ])

    # baseline의 validation 정확도를 시작 기준으로 사용 (기존과 동일)
    # (이보다 좋아질 때만 파인튜닝 모델을 갱신)
    best_validation_accuracy = baseline_validation_accuracy
    no_improvement_count = 0

    # 파인튜닝이 전혀 개선되지 않아도 baseline 상태가 finetuned 파일에
    # 남아 있도록 먼저 저장해 둠 (기존과 동일한 방식)
    torch.save(
        {
            "epoch": checkpoint.get("epoch", 0),
            "fine_tune_epoch": 0,
            "model_state_dict": model.state_dict(),
            "validation_accuracy": baseline_validation_accuracy,
            "validation_defect_f1": checkpoint.get("validation_defect_f1", -1.0),
            "validation_moderate_f1": checkpoint.get("validation_moderate_f1", -1.0),
            "class_names": class_names,
            "source_model": baseline_model_path.name,
            "run_name": RUN_NAME,
            "class_weight_values": class_weight_values,
            "class_weight_mode": checkpoint.get("class_weight_mode", "unknown"),
            "train_class_counts": class_count_by_label,
            "train_total_count": len(train_data)
        },
        fine_tuned_model_path
    )

    training_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n==== Fine-tuning Epoch {epoch}/{num_epochs} ====")

        # ---- Train ----
        model.train()

        # 동결된 앞쪽 특징 블록은 평가 상태로 유지
        # (가중치뿐 아니라 BatchNorm 통계도 바뀌지 않게 함 — 기존과 동일 기법)
        for block in model.features[:-1]:
            block.eval()

        train_loss_sum = 0.0
        train_correct_count = 0
        train_total_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

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
            no_improvement_count = 0

            torch.save(
                {
                    "epoch": checkpoint.get("epoch", 0),
                    "fine_tune_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_defect_f1": validation_defect_f1,
                    "validation_moderate_f1": validation_moderate_f1,
                    "class_names": class_names,
                    "source_model": baseline_model_path.name,
                    "run_name": RUN_NAME,
                    "class_weight_values": class_weight_values,
                    "class_weight_mode": checkpoint.get("class_weight_mode", "unknown"),
                    "train_class_counts": class_count_by_label,
                    "train_total_count": len(train_data)
                },
                fine_tuned_model_path
            )

            print("파인튜닝 최고 모델 저장 :", fine_tuned_model_path)

        else:
            no_improvement_count += 1
            print(
                f"Validation 성능 개선 없음: "
                f"{no_improvement_count}/{patience}"
            )

        # patience 만큼 연속으로 좋아지지 않으면 조기 종료
        if no_improvement_count >= patience:
            print("Validation 성능이 개선되지 않아 중단합니다.")
            break

    training_seconds = time.time() - training_start_time

    print("\n==========================================")
    print("파인튜닝 완료")
    print(f"최고 Validation 정확도: {best_validation_accuracy:.2%}")
    print(f"학습 시간 : {training_seconds:.1f}초")
    print("저장된 모델 :", fine_tuned_model_path)
    print(
        "다음 단계 : $env:EVAL_TARGET=\"finetuned\"; "
        "python src/offline_augmentation/evaluate_augmented.py"
    )


if __name__ == "__main__":
    main()
