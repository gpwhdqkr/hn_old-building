# ============================================================
# [사용법]
# 이 파일은 offline augmentation 실험용 데이터셋/데이터로더 정의 파일입니다.
# 보통은 직접 실행할 필요 없이 train/fine_tune/evaluate 스크립트가 import해서 씁니다.
#
# 데이터 로딩이 정상 동작하는지 확인하고 싶으면 아래처럼 직접 실행:
#   $env:RUN_NAME="aug3"
#   python src/offline_augmentation/preprocess_augmented.py
#
# [전제 조건]
# - build_augmented_dataset.py를 같은 RUN_NAME으로 먼저 실행해서
#   data/augmented_images/metadata_split_augmented_<RUN_NAME>.csv 가 있어야 함
#
# [기존 preprocess_efficientnet.py 와의 관계]
# - transform(train/evaluation)은 기존과 완전히 동일 — 증강 효과만 분리해서
#   보려면 전처리가 같아야 하므로 한 글자도 바꾸지 않았음
# - split도 팀원의 metadata_split.csv에서 온 값을 그대로 승계 (재분리 안 함)
# - 다른 점 2가지:
#   1) 확장 CSV(원본 + 증강본)를 읽음
#   2) 경로 해석이 단순해짐. 기존에는 CSV에 팀원 PC 절대경로가 적혀 있어서
#      raw/images/ 뒤를 잘라 붙이는 변환이 필요했지만, 확장 CSV에는
#      resolved_relpath 열에 프로젝트 기준 상대경로가 이미 들어 있음
#
# [증강본에도 온라인 증강이 얹히는 것에 대해]
# 오프라인 증강본을 train_transform으로 또 한 번 변형하게 되는데 이건 의도한
# 동작입니다. 오프라인은 회전/크롭 같은 기하 다양성을 고정으로 확보하고,
# 온라인은 매 epoch 좌우반전/밝기를 미세하게 흔들어 주는 역할입니다.
# ============================================================

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# 이 파일이 src/offline_augmentation/ 안에 있으므로 세 단계 올라가면 프로젝트 폴더
project_dir = Path(__file__).resolve().parent.parent.parent

# 실험 이름: build_augmented_dataset.py 실행 때와 같은 값을 써야
# 그 실험의 확장 CSV를 찾음
RUN_NAME = os.environ.get("RUN_NAME", "default")

# 원본 + 증강본이 합쳐진 확장 CSV
augmented_metadata_path = (
    project_dir / "data" / "augmented_images"
    / f"metadata_split_augmented_{RUN_NAME}.csv"
)

# ImageNet 사전학습 모델 표준 평균/표준편차 (기존과 동일)
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

# 한 번에 모델에 전달할 이미지 수.
# 기본값 128은 기존 코드와 같은 값이지만, 기존 값은 20GB VRAM 기준으로 잡힌 것이고
# 이 실험은 16GB GPU에서도 돌리므로 환경변수로 낮출 수 있게 했음.
# CUDA out of memory가 나면 $env:BATCH_SIZE="64" 처럼 줄이면 됨
batch_size = int(os.environ.get("BATCH_SIZE", 128))

# 이미지를 미리 읽어두는 병렬 프로세스 수 (GPU가 놀지 않게 함)
# 고정값이면 실행 서버(로컬 PC vs 런팟 등)마다 적정치가 달라 코드를 매번
# 손봐야 하므로, 서버의 실제 CPU 코어 수에 맞춰 자동으로 정함.
# - 코어 절반을 쓰되 최소 4, 최대 32로 제한 (코어가 아주 많아도 워커를
#   과도하게 띄우면 프로세스 관리 오버헤드만 늘어나므로 상한을 둠)
# - 실측: 워커 4개는 128코어 런팟에서 GPU-Util 5%로 병목 발생 확인됨
_cpu_count = os.cpu_count() or 4
num_workers = min(32, max(4, _cpu_count // 2))

# 등급 개수 (0=우수, 1=보통, 2=불량)
NUM_CLASSES = 3


# Train 이미지 전처리 (기존 preprocess_efficientnet.py와 완전히 동일)
train_transform = v2.Compose([
    # PIL 이미지를 PyTorch 이미지 형태로 변환
    v2.ToImage(),
    # 이미 224x224로 변환된 이미지지만 안전장치로 유지 (크기가 다른 이미지가 섞여도 동작)
    v2.Resize(
        size=(224, 224),
        antialias=True
    ),
    # Train만 50% 확률로 좌우반전 (과대적합 방지)
    v2.RandomHorizontalFlip(p=0.5),
    # 밝기/대비를 ±10% 범위에서 무작위 변경 (과대적합 방지)
    v2.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),
    # 픽셀값을 float32로 바꾸고 0~1 범위로 스케일
    v2.ToDtype(
        torch.float32,
        scale=True
    ),
    # ImageNet 사전학습 모델 입력 형식에 맞게 정규화
    v2.Normalize(
        mean=image_mean,
        std=image_std
    )
])

# Validation/Test 이미지 전처리 (증강 없음 — 평가 결과가 흔들리면 안 되므로)
evaluation_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(
        size=(224, 224),
        antialias=True
    ),
    v2.ToDtype(
        torch.float32,
        scale=True
    ),
    v2.Normalize(
        mean=image_mean,
        std=image_std
    )
])


# 확장 CSV의 각 행과 실제 이미지 파일을 연결하는 데이터셋 클래스
class AugmentedBuildingDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    # 데이터셋에 이미지가 몇 장 있는지 반환
    def __len__(self):
        return len(self.dataframe)

    # 지정된 순서의 이미지 한 장과 정답 반환
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        # 확장 CSV에 프로젝트 기준 상대경로가 들어 있어서 그대로 붙이면 됨
        # (원본이면 data/processed_images/..., 증강본이면 data/augmented_images/...)
        image_path = project_dir / row["resolved_relpath"]

        # 이미지를 열고 RGB 형식으로 통일
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        # 위에서 정의한 전처리 적용
        image = self.transform(image)

        # 모델이 사용할 정답값 (0=우수, 1=보통, 2=불량)
        label = int(row["model_label"])

        return image, label


def load_split_dataframes():
    """확장 CSV를 읽어 train/validation/test 데이터프레임을 반환한다."""
    if not augmented_metadata_path.exists():
        raise FileNotFoundError(
            f"확장 CSV를 찾을 수 없습니다: {augmented_metadata_path}\n"
            f"먼저 같은 RUN_NAME으로 증강 데이터셋을 만들어야 합니다:\n"
            f'  $env:RUN_NAME="{RUN_NAME}"\n'
            f"  python src/offline_augmentation/build_augmented_dataset.py"
        )

    metadata = pd.read_csv(
        augmented_metadata_path,
        encoding="utf-8-sig"
    )

    train_data = metadata[
        metadata["split"] == "train"
    ].reset_index(drop=True)

    validation_data = metadata[
        metadata["split"] == "validation"
    ].reset_index(drop=True)

    test_data = metadata[
        metadata["split"] == "test"
    ].reset_index(drop=True)

    return train_data, validation_data, test_data


def compute_class_weights(train_data):
    """학습 데이터의 실제 분포에서 클래스 불균형 보정 가중치를 계산한다.

    기존 코드는 [3.79, 2.95, 0.42]를 하드코딩해 두었는데, 확인해보니 이 값은
        총 train 수 / (등급 수 x 해당 등급 수)
    공식과 정확히 일치한다. 그래서 같은 공식을 그대로 쓴다.

    offline augmentation으로 '보통'을 물리적으로 늘렸는데 가중치 2.95를 그대로
    두면 '데이터도 늘리고 손실도 더 크게 치는' 이중 보정이 된다. 실제 분포에서
    다시 계산해야 증강 효과만 깨끗하게 볼 수 있다.

    AUG_FACTOR=1(증강 없음)일 때는 이 계산이 [3.79, 2.95, 0.42]를 그대로
    재현하므로, 대조군이 기존 실험과 완전히 같은 조건이 된다.

    반환값은 순수 float 리스트다. numpy 타입을 그대로 체크포인트에 저장하면
    torch.load(weights_only=True)에서 로드가 거부되기 때문이다.
    """
    total_count = len(train_data)

    class_weight_values = []

    for model_label in range(NUM_CLASSES):
        class_count = int((train_data["model_label"] == model_label).sum())

        if class_count == 0:
            raise ValueError(
                f"학습 데이터에 등급 {model_label}이 한 장도 없습니다."
            )

        # 기존 하드코딩 값과 같은 숫자가 되도록 소수점 2자리로 반올림
        weight = round(total_count / (NUM_CLASSES * class_count), 2)

        class_weight_values.append(float(weight))

    return class_weight_values


def count_by_class(dataframe):
    """등급별 장수를 {0: n, 1: n, 2: n} 형태로 센다 (학습 조건 확인용 출력)."""
    return {
        model_label: int((dataframe["model_label"] == model_label).sum())
        for model_label in range(NUM_CLASSES)
    }


def build_dataloaders():
    """train/validation/test 데이터로더 3개를 만들어 반환한다.

    사용 예)
        from preprocess_augmented import build_dataloaders
        train_loader, validation_loader, test_loader = build_dataloaders()
    """
    train_data, validation_data, test_data = load_split_dataframes()

    train_dataset = AugmentedBuildingDataset(
        dataframe=train_data,
        transform=train_transform
    )

    validation_dataset = AugmentedBuildingDataset(
        dataframe=validation_data,
        transform=evaluation_transform
    )

    test_dataset = AugmentedBuildingDataset(
        dataframe=test_data,
        transform=evaluation_transform
    )

    # GPU 사용 시 pin_memory로 CPU→GPU 전송 속도 향상 (클라우드 학습 최적화)
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,               # 학습은 매 epoch 순서를 섞음
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0)  # epoch마다 워커 재생성 비용 절약
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,              # 평가는 순서 고정
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, validation_loader, test_loader


# 직접 실행 시: 데이터 로딩이 정상인지 확인
if __name__ == "__main__":

    print("실험 이름 (RUN_NAME) :", RUN_NAME)
    print("확장 CSV :", augmented_metadata_path)

    train_loader, validation_loader, test_loader = build_dataloaders()

    print("=========================")
    print("Train :", len(train_loader.dataset))
    print("Validation :", len(validation_loader.dataset))
    print("Test :", len(test_loader.dataset))

    train_data = train_loader.dataset.dataframe

    print("\nTrain 등급별 장수 :", count_by_class(train_data))
    print("계산된 클래스 가중치 :", compute_class_weights(train_data))

    print(
        "\nTrain 원본/증강본 :",
        train_data["image_source"].value_counts().to_dict()
    )

    # 첫 배치를 실제로 꺼내서 이미지가 잘 읽히는지 확인
    images, labels = next(iter(train_loader))

    print("==========================================")
    print("이미지 묶음 형태 :", images.shape)   # 예상: [128, 3, 224, 224]
    print("정답 묶음 형태 :", labels.shape)     # 예상: [128]
    print("이미지 자료형 :", images.dtype)      # 예상: torch.float32
    print("정답값 예시 :", labels[:10])
