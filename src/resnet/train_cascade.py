# ============================================================
# [사용법]
# 2단계 캐스케이드 학습 스크립트 — 기존 3-클래스 코드는 일절 수정하지 않는 별도 실험.
#
# 배경: 3-클래스 argmax에서는 보통(620장)이 불량(4,437장)과 직접 경쟁해서
# 보통 F1이 0.4 근처에 갇힘 (threshold 보정 실험으로 상한 ~0.46 확인됨).
# 캐스케이드는 이 경쟁 구조 자체를 없앤다:
#   1단계 : 불량 vs 정상(우수+보통)  ← 이미 잘하는 구분 (불량 F1 0.93+)
#   2단계 : 우수 vs 보통             ← 2284 vs 2930으로 거의 균형인 쉬운 문제
#
# 이 스크립트 하나로 두 단계를 각각 학습한다 (STAGE 환경변수로 선택).
# 각 실행은 "fc만 학습(워밍업) → layer4+fc 파인튜닝" 2상을 연속으로 진행하며
# 기존 train_resnet.py / fine_tune_resnet.py와 동일한 학습률/동결 기법을 쓴다.
#
# 실행 (런팟 = 리눅스, 두 단계를 순서대로):
#   RESNET_ARCH=resnet18 RUN_NAME=cas1 STAGE=1 python src/resnet/train_cascade.py
#   RESNET_ARCH=resnet18 RUN_NAME=cas1 STAGE=2 python src/resnet/train_cascade.py
# 학습이 끝나면:
#   RESNET_ARCH=resnet18 RUN_NAME=cas1 python src/resnet/evaluate_cascade.py
#
# [환경변수]
#   STAGE        : 1 (불량 vs 정상) 또는 2 (우수 vs 보통) — 필수
#   RESNET_ARCH  : resnet18 (기본) 또는 resnet50
#   RUN_NAME     : 실험 이름 (기본 "default"). 두 단계와 평가에 같은 값을 쓸 것
#   HEAD_EPOCHS  : fc만 학습하는 워밍업 epoch 수 (기본 5)
#   FT_EPOCHS    : layer4+fc 파인튜닝 최대 epoch 수 (기본 10)
#   PATIENCE     : 파인튜닝 조기 종료 기준 (기본 3 epoch 무개선 시 중단)
#   BATCH_SIZE   : 배치 크기 (기본 128)
#   DATA_DIR     : data/ 와 model/ 이 있는 프로젝트 루트 (기본: 이 저장소)
#
# 결과:
#   model/best_<RESNET_ARCH>_cascade_stage<STAGE>_<RUN_NAME>.pth
#   (validation macro F1이 가장 높았던 시점 저장 — 두 단계 모두 이진 분류라
#    다수 클래스 쏠림에 강한 macro F1을 선택 기준으로 씀)
#
# 클래스 가중치는 고정값이 아니라 "이 단계의 실제 train 분포"로 매번 계산한다.
# (기존 실험에서 증강+고정 가중치가 겹쳐 불량이 이중 억압됐던 문제를 피하기 위함)
# ============================================================

import os
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50
)

# 전처리는 기존 학습과 완전히 동일해야 하므로 팀 공용 정의를 그대로 가져옴
from preprocess_resnet import (
    RAW_PATH_MARKER,
    evaluation_transform,
    num_workers,
    train_transform
)

project_dir = Path(__file__).resolve().parent.parent.parent

STAGE = os.environ.get("STAGE")
RESNET_ARCH = os.environ.get("RESNET_ARCH", "resnet18")
RUN_NAME = os.environ.get("RUN_NAME", "default")

head_epochs = int(os.environ.get("HEAD_EPOCHS", 5))
fine_tune_epochs = int(os.environ.get("FT_EPOCHS", 10))
patience = int(os.environ.get("PATIENCE", 3))
batch_size = int(os.environ.get("BATCH_SIZE", 128))

data_dir = Path(os.environ.get("DATA_DIR", project_dir))

metadata_path = data_dir / "data" / "processed" / "metadata_split.csv"
processed_images_dir = data_dir / "data" / "processed_images"

model_dir = data_dir / "model"

# 단계별 문제 정의
# stage_labels : 원본 3-클래스 라벨(0=우수,1=보통,2=불량) → 이 단계의 이진 라벨
STAGE_CONFIG = {
    "1": {
        "description": "1단계 : 불량(1) vs 정상=우수+보통(0)",
        "use_labels": [0, 1, 2],          # 전체 데이터 사용
        "label_map": {0: 0, 1: 0, 2: 1},  # 우수/보통 → 0, 불량 → 1
        "class_names": {0: "정상(우수+보통)", 1: "불량"},
    },
    "2": {
        "description": "2단계 : 우수(0) vs 보통(1)",
        "use_labels": [0, 1],             # 불량 제외
        "label_map": {0: 0, 1: 1},        # 우수 → 0, 보통 → 1
        "class_names": {0: "우수", 1: "보통"},
    },
}


class CascadeDataset(Dataset):
    """metadata_split.csv에서 이 단계에 필요한 행만 골라 이진 라벨로 반환한다."""

    def __init__(self, dataframe, images_dir, label_map, transform):
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        # CSV의 팀원 PC 경로에서 raw/images/ 뒤 상대 경로만 잘라 실제 위치로 변환
        posix_path = str(row["image_path"]).replace("\\", "/")
        marker_index = posix_path.find(RAW_PATH_MARKER)

        if marker_index == -1:
            raise ValueError(
                f"이미지 경로에서 '{RAW_PATH_MARKER}'를 찾을 수 없습니다: "
                f"{row['image_path']}"
            )

        relative_path = posix_path[marker_index + len(RAW_PATH_MARKER):]
        image_path = self.images_dir / relative_path

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image = self.transform(image)

        # 원본 3-클래스 라벨을 이 단계의 이진 라벨로 변환
        label = self.label_map[int(row["model_label"])]

        return image, label


def create_cascade_model(use_pretrained_weights):
    """이진 분류(출력 2개)용 ResNet을 만든다."""
    if RESNET_ARCH == "resnet18":
        weights = ResNet18_Weights.DEFAULT if use_pretrained_weights else None
        model = resnet18(weights=weights)
    elif RESNET_ARCH == "resnet50":
        weights = ResNet50_Weights.DEFAULT if use_pretrained_weights else None
        model = resnet50(weights=weights)
    else:
        raise ValueError(
            f"지원하지 않는 RESNET_ARCH: {RESNET_ARCH} (resnet18/resnet50만 가능)"
        )

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 2)

    return model


def freeze_backbone_batchnorm(model):
    """동결된 앞쪽 층의 BatchNorm 통계가 바뀌지 않게 평가 상태로 둔다.
    (기존 fine_tune_resnet.py와 동일 기법)"""
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()


def run_epoch_train(model, loader, loss_function, optimizer, device,
                    keep_backbone_bn_frozen):
    """1 epoch 학습을 수행하고 (평균 손실, 정확도)를 반환한다."""
    model.train()

    if keep_backbone_bn_frozen:
        freeze_backbone_batchnorm(model)

    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        total_count += labels.size(0)

    return loss_sum / total_count, correct_count / total_count


def run_epoch_validation(model, loader, loss_function, device):
    """validation 전체를 평가하고 (평균 손실, 정확도, macro F1)을 반환한다."""
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    total_count = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss_sum += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct_count += (predictions == labels).sum().item()
            total_count += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    macro_f1 = f1_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )

    return loss_sum / total_count, correct_count / total_count, macro_f1


def main():
    if STAGE not in STAGE_CONFIG:
        raise ValueError(
            "STAGE 환경변수를 1 또는 2로 지정하세요.\n"
            "예) STAGE=1 python src/resnet/train_cascade.py"
        )

    config = STAGE_CONFIG[STAGE]

    best_model_path = (
        model_dir
        / f"best_{RESNET_ARCH}_cascade_stage{STAGE}_{RUN_NAME}.pth"
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("사용 장치 :", device)
    print(config["description"])
    print("실험 이름 :", RUN_NAME)

    # ---------- 데이터 준비 ----------
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")

    # 이 단계에서 쓰는 원본 라벨만 남김 (2단계는 불량 제외)
    metadata = metadata[
        metadata["model_label"].isin(config["use_labels"])
    ]

    train_data = metadata[
        metadata["split"] == "train"
    ].reset_index(drop=True)

    validation_data = metadata[
        metadata["split"] == "validation"
    ].reset_index(drop=True)

    # 이진 변환 후의 train 분포로 클래스 가중치 계산
    # 공식: 전체 장수 / (클래스 수 × 해당 클래스 장수)
    binary_train_labels = train_data["model_label"].map(
        config["label_map"]
    )
    class_counts = binary_train_labels.value_counts().sort_index()

    # float()로 감싸는 이유: pandas 계산 결과(np.float64)를 체크포인트에 그대로
    # 저장하면 torch.load(weights_only=True)에서 로드가 거부됨
    class_weight_values = [
        float(len(binary_train_labels) / (2 * class_counts[class_index]))
        for class_index in [0, 1]
    ]

    print("Train 장수 :", len(train_data),
          "| 이진 분포 :", class_counts.to_dict())
    print("Validation 장수 :", len(validation_data))
    print("클래스 가중치 (train 분포로 계산) :",
          [round(value, 4) for value in class_weight_values])

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        CascadeDataset(
            train_data, processed_images_dir,
            config["label_map"], train_transform
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0)
    )

    validation_loader = DataLoader(
        CascadeDataset(
            validation_data, processed_images_dir,
            config["label_map"], evaluation_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0)
    )

    # ---------- 모델 준비 ----------
    model = create_cascade_model(use_pretrained_weights=True)
    model = model.to(device)

    class_weights = torch.tensor(
        class_weight_values,
        dtype=torch.float32,
        device=device
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    best_macro_f1 = -1.0

    def save_checkpoint(phase, epoch, validation_accuracy, macro_f1):
        torch.save(
            {
                "phase": phase,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": macro_f1,
                "stage": STAGE,
                "stage_description": config["description"],
                "class_names": config["class_names"],
                "label_map": config["label_map"],
                "class_weight_values": class_weight_values,
                "run_name": RUN_NAME,
                "resnet_arch": RESNET_ARCH,
                "batch_size": batch_size,
                "head_epochs": head_epochs,
                "fine_tune_epochs": fine_tune_epochs,
                "train_count": len(train_data),
                "validation_count": len(validation_data),
            },
            best_model_path
        )

    training_start_time = time.time()

    # ---------- 1상 : fc만 학습 (워밍업) ----------
    # 기존 train_resnet.py와 동일: 특징 추출부 전체 동결, fc만 lr 0.001
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = Adam(model.fc.parameters(), lr=0.001)

    for epoch in range(1, head_epochs + 1):
        print(f"\n==== [워밍업 {epoch}/{head_epochs}] fc만 학습 ====")

        train_loss, train_accuracy = run_epoch_train(
            model, train_loader, loss_function, optimizer, device,
            keep_backbone_bn_frozen=True
        )
        validation_loss, validation_accuracy, macro_f1 = run_epoch_validation(
            model, validation_loader, loss_function, device
        )

        print(f"Train Loss: {train_loss:.4f} | "
              f"Train Accuracy: {train_accuracy:.2%}")
        print(f"Validation Loss: {validation_loss:.4f} | "
              f"Validation Accuracy: {validation_accuracy:.2%} | "
              f"Validation Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_checkpoint("head", epoch, validation_accuracy, macro_f1)
            print("최고 성능 모델 저장 :", best_model_path.name)

    # ---------- 2상 : layer4 + fc 파인튜닝 ----------
    # 기존 fine_tune_resnet.py와 동일: layer4 lr 0.00001, fc lr 0.0001
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = Adam([
        {"params": model.layer4.parameters(), "lr": 0.00001},
        {"params": model.fc.parameters(), "lr": 0.0001},
    ])

    no_improvement_count = 0

    for epoch in range(1, fine_tune_epochs + 1):
        print(f"\n==== [파인튜닝 {epoch}/{fine_tune_epochs}] layer4+fc ====")

        train_loss, train_accuracy = run_epoch_train(
            model, train_loader, loss_function, optimizer, device,
            keep_backbone_bn_frozen=True
        )
        validation_loss, validation_accuracy, macro_f1 = run_epoch_validation(
            model, validation_loader, loss_function, device
        )

        print(f"Train Loss: {train_loss:.4f} | "
              f"Train Accuracy: {train_accuracy:.2%}")
        print(f"Validation Loss: {validation_loss:.4f} | "
              f"Validation Accuracy: {validation_accuracy:.2%} | "
              f"Validation Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            no_improvement_count = 0
            save_checkpoint("finetune", epoch, validation_accuracy, macro_f1)
            print("최고 성능 모델 저장 :", best_model_path.name)
        else:
            no_improvement_count += 1
            print(f"개선 없음 ({no_improvement_count}/{patience})")

            if no_improvement_count >= patience:
                print("조기 종료: validation macro F1이 "
                      f"{patience} epoch 동안 개선되지 않음")
                break

    training_seconds = time.time() - training_start_time

    # 학습 시간을 체크포인트에 기록 (기존 파이프라인과 동일한 방식)
    checkpoint = torch.load(
        best_model_path, map_location="cpu", weights_only=True
    )
    checkpoint["training_seconds"] = training_seconds
    torch.save(checkpoint, best_model_path)

    print("\n==========================================")
    print("최고 Validation Macro F1 :", f"{best_macro_f1:.4f}")
    print("학습 시간 :", f"{training_seconds:.1f}초")
    print("저장된 모델 :", best_model_path)

    if STAGE == "1":
        print("\n다음 단계:")
        print(f"  RESNET_ARCH={RESNET_ARCH} RUN_NAME={RUN_NAME} STAGE=2 "
              "python src/resnet/train_cascade.py")
    else:
        print("\n두 단계 학습이 끝났다면 평가:")
        print(f"  RESNET_ARCH={RESNET_ARCH} RUN_NAME={RUN_NAME} "
              "python src/resnet/evaluate_cascade.py")


if __name__ == "__main__":
    main()
