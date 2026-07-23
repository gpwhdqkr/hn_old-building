# ============================================================
# [사용법]
# 이 파일은 EfficientNet 학습에 쓰는 데이터셋/데이터로더 정의 파일입니다.
# 보통은 직접 실행할 필요 없이 train/fine_tune/evaluate 스크립트가 import해서 씁니다.
#
# 데이터 로딩이 정상 동작하는지 확인하고 싶으면 아래처럼 직접 실행:
#   python src/efficientnet/preprocess_efficientnet.py
#
# [전제 조건]
# - data/processed/metadata_split.csv 가 있어야 함 (팀원이 만든 split 결과, 깃에 포함)
# - data/processed_images/ 에 224x224 변환 이미지가 있어야 함
#   (용량 문제로 깃에 없음 → 런팟에서는 구글드라이브에서 받아서 이 위치에 풀어야 함)
#
# [팀원 코드와의 관계]
# - 전처리(transform)는 팀원의 preprocess_data.py와 완전히 동일
# - split도 팀원의 metadata_split.csv를 그대로 사용 (재분리 안 함 → 공정 비교)
# - 다른 점: CSV의 이미지 경로가 팀원 PC 원본 경로(D:/hn_old-building_raw/...)라서
#   이 프로젝트의 data/processed_images/ 경로로 치환해서 읽음
# ============================================================

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# 이 파일이 src/efficientnet/ 안에 있으므로 세 단계 올라가면 프로젝트 폴더
# (로컬 D:\hn_old-building 이든 런팟 /workspace/hn_old-building 이든 동일하게 동작)
project_dir = Path(__file__).resolve().parent.parent.parent

# 팀원이 만든 split 결과 CSV
metadata_path = project_dir / "data" / "processed" / "metadata_split.csv"

# 팀원이 구글드라이브로 전달한 224x224 변환 이미지 폴더
processed_images_dir = project_dir / "data" / "processed_images"

# CSV의 경로에서 이 문자열 뒤쪽이 실제 폴더 구조와 일치함
# 예: D:/hn_old-building_raw/raw/images/TS_아파트/.../xxx.jpg
#     → raw/images/ 뒤인 TS_아파트/.../xxx.jpg 만 잘라서 씀
RAW_PATH_MARKER = "raw/images/"

# ImageNet 사전학습 모델 표준 평균/표준편차 (팀원과 동일)
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

# 한 번에 모델에 전달할 이미지 수
# 20GB VRAM 기준 EfficientNet-B0 + 224x224 에서 여유 있는 값
batch_size = 128

# 이미지를 미리 읽어두는 병렬 프로세스 수 (GPU가 놀지 않게 함)
num_workers = 4


def convert_to_local_path(csv_image_path):
    """CSV에 적힌 팀원 PC 경로를 이 프로젝트의 processed_images 경로로 변환한다.

    예) D:/hn_old-building_raw/raw/images/TS_아파트/a/b.jpg
        → <프로젝트>/data/processed_images/TS_아파트/a/b.jpg
    """
    # 윈도우 역슬래시가 섞여 있어도 처리되도록 통일
    posix_path = str(csv_image_path).replace("\\", "/")

    marker_index = posix_path.find(RAW_PATH_MARKER)

    if marker_index == -1:
        raise ValueError(
            f"이미지 경로에서 '{RAW_PATH_MARKER}'를 찾을 수 없습니다: {csv_image_path}"
        )

    # raw/images/ 뒤의 상대 경로만 추출
    relative_path = posix_path[marker_index + len(RAW_PATH_MARKER):]

    return processed_images_dir / relative_path


# Train 이미지 전처리 (팀원의 preprocess_data.py와 동일)
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


# CSV의 각 행과 실제 이미지 파일을 연결하는 데이터셋 클래스
class BuildingDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    # 데이터셋에 이미지가 몇 장 있는지 반환
    def __len__(self):
        return len(self.dataframe)

    # 지정된 순서의 이미지 한 장과 정답 반환
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        # CSV의 팀원 PC 경로를 이 프로젝트의 processed_images 경로로 치환
        image_path = convert_to_local_path(row["image_path"])

        # 이미지를 열고 RGB 형식으로 통일
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        # 위에서 정의한 전처리 적용
        image = self.transform(image)

        # 모델이 사용할 정답값 (0=우수, 1=보통, 2=불량)
        label = int(row["model_label"])

        return image, label


def load_split_dataframes():
    """metadata_split.csv를 읽어 train/validation/test 데이터프레임을 반환한다."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata_split.csv를 찾을 수 없습니다: {metadata_path}\n"
            "깃에서 data/processed/metadata_split.csv를 받았는지 확인하세요."
        )

    if not processed_images_dir.exists():
        raise FileNotFoundError(
            f"이미지 폴더를 찾을 수 없습니다: {processed_images_dir}\n"
            "구글드라이브에서 processed_images를 받아 data/ 아래에 풀었는지 확인하세요."
        )

    metadata = pd.read_csv(
        metadata_path,
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


def build_dataloaders():
    """train/validation/test 데이터로더 3개를 만들어 반환한다.

    사용 예)
        from preprocess_efficientnet import build_dataloaders
        train_loader, validation_loader, test_loader = build_dataloaders()
    """
    train_data, validation_data, test_data = load_split_dataframes()

    train_dataset = BuildingDataset(
        dataframe=train_data,
        transform=train_transform
    )

    validation_dataset = BuildingDataset(
        dataframe=validation_data,
        transform=evaluation_transform
    )

    test_dataset = BuildingDataset(
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

    train_loader, validation_loader, test_loader = build_dataloaders()

    print("=========================")
    print("Train :", len(train_loader.dataset))
    print("Validation :", len(validation_loader.dataset))
    print("Test :", len(test_loader.dataset))

    # 첫 배치를 실제로 꺼내서 이미지가 잘 읽히는지 확인
    images, labels = next(iter(train_loader))

    print("==========================================")
    print("이미지 묶음 형태 :", images.shape)   # 예상: [128, 3, 224, 224]
    print("정답 묶음 형태 :", labels.shape)     # 예상: [128]
    print("이미지 자료형 :", images.dtype)      # 예상: torch.float32
    print("정답값 예시 :", labels[:10])
