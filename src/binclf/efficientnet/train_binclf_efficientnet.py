# ============================================================
# [사용법]
# 이원화(2클래스: 우수 vs 불량) EfficientNet-B0 baseline 학습 스크립트 (1단계).
# ImageNet 사전학습 가중치를 불러와 특징 추출부(features)는 전부 동결하고
# 마지막 분류층(classifier)만 우수/불량 2클래스 분류로 학습합니다.
#
# 실행 (런팟 = 리눅스):
#   RUN_NAME=bin1 NUM_EPOCHS=20 python src/binclf/efficientnet/train_binclf_efficientnet.py
#
# [주의: 윈도우 PowerShell은 리눅스식 인라인 환경변수를 못 씁니다]
#   $env:RUN_NAME="bin1"; $env:NUM_EPOCHS="20"
#   python src/binclf/efficientnet/train_binclf_efficientnet.py
#
# [환경변수]
#   RUN_NAME    : 실험 이름 (기본 "default"). 저장 파일명에 붙어서 실험별로 결과가 쌓임
#   NUM_EPOCHS  : 학습 epoch 수 (기본 10)
#   BATCH_SIZE  : 배치 크기 (기본 128)
#   DATA_DIR    : 데이터가 다른 위치에 있을 때만 지정
#
# 결과:
#   model/best_efficientnet_b0_binclf_baseline_<RUN_NAME>.pth
#   (validation macro F1이 가장 높았던 시점의 모델이 저장됨)
#
# 다음 단계 (같은 RUN_NAME을 반드시 그대로 넘겨야 함):
#   RUN_NAME=bin1 python src/binclf/efficientnet/fine_tune_binclf_efficientnet.py
#
# [기존 3클래스 실험과의 비교 조건]
# - 전처리/split/옵티마이저(Adam, lr=0.001)/동결 범위는 기존 src/efficientnet/과 동일.
#   다른 것은 ① 라벨 정의(보통→불량 병합) ② 클래스 가중치(이원화 train 분포에서
#   재계산 — 하드코딩 3클래스 값은 사용 불가) ③ 최고모델 선택 기준.
# - 선택 기준이 validation macro F1인 이유: 이원화 후 불량이 약 91%라 accuracy
#   기준으로는 "전부 불량" 예측 모델이 선택될 위험이 큼. train_cascade.py와 동일 근거.
#   기존 3클래스 baseline(val accuracy 기준)과 선택 기준이 다르다는 점을 결과표에
#   명시할 것.
# ============================================================

import os
import time

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0
)

from preprocess_binclf_efficientnet import (
    batch_size,
    build_dataloaders,
    class_names,
    compute_class_weights,
    load_split_dataframes,
    model_dir
)

# 모델 구조 이름 (파일명/결과 폴더명에 사용 — EfficientNet은 B0 고정)
MODEL_ARCH = "efficientnet_b0"

# 실험 이름: RUN_NAME 환경변수로 지정 (미지정 시 "default")
RUN_NAME = os.environ.get("RUN_NAME", "default")

# 가장 성능이 좋은 모델을 저장할 경로
# ("binclf" 토큰이 있어 기존 3클래스 실험 파일과 절대 겹치지 않음)
best_model_path = model_dir / f"best_{MODEL_ARCH}_binclf_baseline_{RUN_NAME}.pth"

# 학습 epoch 수: NUM_EPOCHS 환경변수로 지정 (미지정 시 10)
num_epochs = int(os.environ.get("NUM_EPOCHS", 10))


def create_efficientnet_model(use_pretrained_weights):
    """EfficientNet-B0를 만들고 분류층을 2클래스 출력으로 교체해서 반환한다.

    use_pretrained_weights:
        True  → ImageNet 사전학습 가중치를 받아서 시작 (1단계 학습용)
        False → 빈 구조만 생성 (저장된 체크포인트를 덮어씌울 때용)

    train/fine_tune/evaluate가 반드시 같은 구조를 만들어야 하므로
    각 스크립트에 동일한 함수를 둔다. (한쪽만 바꾸면 가중치 로드가 실패함)
    """
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT if use_pretrained_weights else None
    )

    # 기존 마지막 분류층(1000 클래스)을 우수/불량 2클래스 출력으로 교체
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        input_features,
        2
    )

    return model


def evaluate_on_loader(model, data_loader, loss_function, device):
    """주어진 데이터로더 전체에 대해 손실/정확도/F1을 계산한다.

    반환: (loss, accuracy, 우수 F1, 불량 F1, macro F1)
    macro F1이 최고 모델 선택 기준이고 나머지는 참고용 출력.
    """
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    total_count = 0

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

            # 두 출력 중 가장 큰 위치를 예측 등급으로 선택
            predictions = outputs.argmax(dim=1)

            correct_count += (predictions == labels).sum().item()
            total_count += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = loss_sum / total_count
    accuracy = correct_count / total_count

    # 클래스별 F1 (순서: 우수, 불량)
    # float() 변환 필수: numpy 타입을 체크포인트에 저장하면
    # torch.load(weights_only=True)에서 로드가 거부됨
    per_class_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=[0, 1],
        average=None,
        zero_division=0
    )

    good_f1 = float(per_class_f1[0])
    defect_f1 = float(per_class_f1[1])
    macro_f1 = float(per_class_f1.mean())

    return average_loss, accuracy, good_f1, defect_f1, macro_f1


def main():
    model_dir.mkdir(parents=True, exist_ok=True)

    print("모델 종류 :", MODEL_ARCH)
    print("실험 이름 (RUN_NAME) :", RUN_NAME)
    print("배치 크기 :", batch_size)
    print("라벨 정의 : 0=우수, 1=불량(보통 병합)")

    # GPU가 있으면 GPU, 없으면 CPU 사용
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("사용 장치 :", device)

    if device.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))

    # 이원화 train 분포에서 클래스 가중치 계산 (하드코딩 3클래스 값은 사용 불가)
    train_data, _, _ = load_split_dataframes()
    class_weight_values = compute_class_weights(train_data)

    print("Train 분포 :",
          train_data["model_label"].value_counts().sort_index().to_dict())
    print("클래스 가중치 (train 분포로 계산) :",
          [round(value, 4) for value in class_weight_values])

    train_loader, validation_loader, _ = build_dataloaders()

    # ImageNet으로 사전학습된 EfficientNet-B0 생성 (분류층은 2클래스로 교체된 상태)
    model = create_efficientnet_model(use_pretrained_weights=True)

    # 특징 추출부 전체 동결 (분류층만 학습하는 baseline)
    # EfficientNet은 특징 추출부가 model.features 컨테이너에 모여 있어
    # 그대로 전부 잠그면 됨 (기존 src/efficientnet/train_efficientnet.py와 동일)
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    model = model.to(device)

    # 실제로 학습되는 파라미터 수 (결과 리포트에 조건을 명시하기 위해 기록)
    trainable_param_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_param_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"전체 파라미터 : {total_param_count / 1e6:.1f}M / "
        f"학습 파라미터 : {trainable_param_count / 1e6:.2f}M"
    )

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

    # 최고 성능 기록 (이원화는 불균형이 심해 macro F1 기준 — 헤더 주석 참고)
    best_validation_macro_f1 = -1.0

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
            validation_good_f1,
            validation_defect_f1,
            validation_macro_f1
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
            f"Accuracy: {validation_accuracy:.2%} | "
            f"우수 F1: {validation_good_f1:.4f} | "
            f"불량 F1: {validation_defect_f1:.4f} | "
            f"Macro F1: {validation_macro_f1:.4f}"
        )

        # validation macro F1이 최고 기록을 넘으면 저장
        if validation_macro_f1 > best_validation_macro_f1:
            best_validation_macro_f1 = validation_macro_f1

            torch.save(
                {
                    "task": "binclf",
                    "label_definition": "0=우수, 1=불량(보통 병합)",
                    "metadata_file": "metadata_split_binary3.csv",
                    "selection_metric": "macro_f1",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_macro_f1": validation_macro_f1,
                    "validation_accuracy": validation_accuracy,
                    "validation_good_f1": validation_good_f1,
                    "validation_defect_f1": validation_defect_f1,
                    "class_names": class_names,
                    "class_weight_values": class_weight_values,
                    "run_name": RUN_NAME,
                    "model_arch": MODEL_ARCH,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "trainable_param_count": trainable_param_count,
                    "total_param_count": total_param_count,
                    "unfrozen_layers": "classifier"
                },
                best_model_path
            )

            print("최고 성능 모델 저장 :", best_model_path)

    training_seconds = time.time() - training_start_time

    print("\n==========================================")
    print(f"최고 Validation Macro F1: {best_validation_macro_f1:.4f}")
    print(f"학습 시간 : {training_seconds:.1f}초")
    print("저장된 모델 :", best_model_path)


if __name__ == "__main__":
    main()
