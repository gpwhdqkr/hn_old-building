# ============================================================
# [사용법]
# EfficientNet-B0 파인튜닝 스크립트 (2단계).
# 1단계 baseline 모델을 불러와서, 마지막 특징 블록까지 동결을 풀고
# 아주 작은 학습률로 조금 더 학습합니다. (팀원의 fine_tune_model.py와 동일한 방식)
#
# 실행 (반드시 train_efficientnet.py를 먼저 실행해야 함):
#   python src/efficientnet/fine_tune_efficientnet.py
#
# 결과:
#   model/best_efficientnet_b0_finetuned.pth
#   (validation 정확도가 baseline보다 좋아진 시점의 모델이 저장됨 — 팀원과 동일 기준.
#    한 번도 안 좋아지면 baseline 상태 그대로 저장되어 있음)
#
# 다음 단계:
#   python src/efficientnet/evaluate_efficientnet.py
# ============================================================

import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torchvision.models import efficientnet_b0

from preprocess_efficientnet import build_dataloaders

project_dir = Path(__file__).resolve().parent.parent.parent

model_dir = project_dir / "model"

# 1단계에서 저장한 baseline 모델
baseline_model_path = model_dir / "best_efficientnet_b0_baseline.pth"

# 파인튜닝 결과를 저장할 경로
fine_tuned_model_path = model_dir / "best_efficientnet_b0_finetuned.pth"

# 최대 파인튜닝 epoch 수
num_epochs = 10

# 3회 연속 validation 정확도가 좋아지지 않으면 조기 종료 (팀원과 동일)
patience = 3

# 클래스 불균형 보정 가중치 (팀원과 동일)
class_weight_values = [3.79, 2.95, 0.42]

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

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss_sum += loss.item() * images.size(0)

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
    defect_f1 = per_class_f1[2]

    return average_loss, accuracy, defect_f1


def main():
    if not baseline_model_path.exists():
        raise FileNotFoundError(
            f"baseline 모델을 찾을 수 없습니다: {baseline_model_path}\n"
            "먼저 train_efficientnet.py를 실행하세요."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("사용 장치 :", device)

    train_loader, validation_loader, _ = build_dataloaders()

    # 가중치를 체크포인트에서 불러올 것이므로 ImageNet 가중치는 다시 받지 않음
    model = efficientnet_b0(weights=None)

    # baseline과 동일하게 분류층을 3등급 출력으로 교체
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        input_features,
        3
    )

    # baseline이 학습한 가중치를 불러와 적용
    checkpoint = torch.load(
        baseline_model_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    baseline_validation_accuracy = checkpoint.get("validation_accuracy", -1.0)
    print(f"baseline Validation 정확도: {baseline_validation_accuracy:.2%}")

    # 학습할 부분 정하기:
    # 특징 추출부 전체를 동결한 뒤, 마지막 특징 블록만 동결 해제
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-1].parameters():
        parameter.requires_grad = True

    model = model.to(device)

    class_weights = torch.tensor(
        class_weight_values,
        dtype=torch.float32,
        device=device
    )
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # 마지막 특징 블록은 아주 조금만, 분류층은 그보다 조금 더 크게 수정
    # (팀원의 fine_tune_model.py와 동일한 차등 학습률)
    optimizer = Adam(
        [
            {
                "params": model.features[-1].parameters(),
                "lr": 0.00001
            },
            {
                "params": model.classifier.parameters(),
                "lr": 0.0001
            }
        ]
    )

    # baseline의 validation 정확도를 시작 기준으로 사용 (팀원과 동일)
    # (이보다 좋아질 때만 파인튜닝 모델을 갱신)
    best_validation_accuracy = baseline_validation_accuracy
    no_improvement_count = 0

    # 파인튜닝이 전혀 개선되지 않아도 baseline 상태가 finetuned 파일에
    # 남아 있도록 먼저 저장해 둠 (팀원과 동일한 방식)
    torch.save(
        {
            "epoch": checkpoint.get("epoch", 0),
            "fine_tune_epoch": 0,
            "model_state_dict": model.state_dict(),
            "validation_accuracy": baseline_validation_accuracy,
            "validation_defect_f1": checkpoint.get("validation_defect_f1", -1.0),
            "class_names": class_names,
            "source_model": baseline_model_path.name
        },
        fine_tuned_model_path
    )

    training_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n==== Fine-tuning Epoch {epoch}/{num_epochs} ====")

        # ---- Train ----
        model.train()

        # 동결된 앞쪽 특징 블록은 평가 상태로 유지
        # (가중치뿐 아니라 BatchNorm 통계도 바뀌지 않게 함 — 팀원과 동일 기법)
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
            no_improvement_count = 0

            torch.save(
                {
                    "epoch": checkpoint.get("epoch", 0),
                    "fine_tune_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_defect_f1": validation_defect_f1,
                    "class_names": class_names,
                    "source_model": baseline_model_path.name
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


if __name__ == "__main__":
    main()
