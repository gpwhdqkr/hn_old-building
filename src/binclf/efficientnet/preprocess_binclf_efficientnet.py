# ============================================================
# [사용법]
# 이원화(2클래스: 우수 vs 불량) EfficientNet-B0 학습에 쓰는 데이터셋/데이터로더 정의 파일입니다.
# 보통은 직접 실행할 필요 없이 train/fine_tune/evaluate 스크립트가 import해서 씁니다.
#
# 데이터 로딩이 정상 동작하는지 확인하고 싶으면 아래처럼 직접 실행:
#   python src/binclf/efficientnet/preprocess_binclf_efficientnet.py
#
# [전제 조건]
# - data/processed/metadata_split_binary3.csv 가 있어야 함
#   (metadata_split_binary3.xlsx를 src/binclf/convert_binary_metadata.py로 변환한
#    이원화 정본. 보통이 불량에 병합되어 model_label이 0=우수, 1=불량 두 값뿐임.
#    깃에 포함되어 있으므로 런팟에서는 git pull만 하면 됨)
# - data/processed_images/ 에 224x224 변환 이미지가 있어야 함
#   (용량 문제로 깃에 없음 → 런팟에서는 구글드라이브에서 받아서 이 위치에 풀어야 함)
#
# [기존 3클래스 코드와의 관계 — 공정 비교를 위해 전처리를 일부러 똑같이 맞춤]
# - transform(증강/정규화)은 src/efficientnet/preprocess_efficientnet.py와 완전히 동일
# - split도 기존 metadata_split.csv와 동일 (binary3는 라벨만 이원화, split은 그대로)
# → 따라서 기존 3클래스 실험과의 차이는 "라벨 정의 차이"로만 해석할 수 있음.
#   여기를 건드리면 기존 실험들과 숫자를 나란히 놓을 수 없게 되므로 수정 금지.
#
# [환경변수]
#   BATCH_SIZE      : 배치 크기 (기본 128)
#   DATA_DIR        : data/ 와 model/ 이 있는 프로젝트 루트 (기본: 이 저장소).
#                     데이터를 다른 위치에 둔 경우에만 지정
#   LIMIT_PER_SPLIT : 각 split에서 사용할 최대 장수 (기본 0 = 전체).
#                     스모크 테스트 전용 — 실험 결과에 절대 사용 금지
# ============================================================

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# 이 파일이 src/binclf/efficientnet/ 안에 있으므로 네 단계 올라가면 프로젝트 폴더
# (로컬 D:\hn_old-building 이든 런팟 /workspace/hn_old-building 이든 동일하게 동작)
project_dir = Path(__file__).resolve().parent.parent.parent.parent

# 데이터/모델 위치: 기본은 이 저장소, 다른 곳에 데이터를 뒀으면 DATA_DIR로 지정
data_dir = Path(os.environ.get("DATA_DIR", project_dir))

# 이원화 정본 CSV (0=우수, 1=불량(보통 병합))
metadata_path = data_dir / "data" / "processed" / "metadata_split_binary3.csv"

# 팀원이 구글드라이브로 전달한 224x224 변환 이미지 폴더
processed_images_dir = data_dir / "data" / "processed_images"

# 학습된 모델 저장 폴더 (train/fine_tune/evaluate가 공통으로 사용)
model_dir = data_dir / "model"

# 결과 저장 최상위 폴더 (기존 test_results/<arch>/ 와 분리)
test_results_root = data_dir / "test_results" / "binclf"

# CSV의 경로에서 이 문자열 뒤쪽이 실제 폴더 구조와 일치함
# 예: D:/hn_old-building_raw/raw/images/TS_아파트/.../xxx.jpg
#     → raw/images/ 뒤인 TS_아파트/.../xxx.jpg 만 잘라서 씀
RAW_PATH_MARKER = "raw/images/"

# 이원화 라벨 정의 (binary3 CSV의 model_label 값 그대로)
class_names = {
    0: "우수",
    1: "불량"
}

# ImageNet 사전학습 모델 표준 평균/표준편차 (기존 3클래스 실험과 동일)
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

# 한 번에 모델에 전달할 이미지 수 (기존 EfficientNet 실험과 동일한 기본값)
batch_size = int(os.environ.get("BATCH_SIZE", 128))

# 스모크 테스트용: 각 split에서 사용할 최대 장수 (0 = 전체 사용)
limit_per_split = int(os.environ.get("LIMIT_PER_SPLIT", 0))

# 이미지를 미리 읽어두는 병렬 프로세스 수 (기존과 동일한 자동 산정)
_cpu_count = os.cpu_count() or 4
num_workers = min(32, max(4, _cpu_count // 2))


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


# Train 이미지 전처리 (기존 3클래스 실험과 동일 — 수정 금지)
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

        # 모델이 사용할 정답값 (0=우수, 1=불량 — binary3에서 이미 이원화됨)
        label = int(row["model_label"])

        return image, label


def load_split_dataframes():
    """metadata_split_binary3.csv를 읽어 train/validation/test 데이터프레임을 반환한다."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata_split_binary3.csv를 찾을 수 없습니다: {metadata_path}\n"
            "깃에서 data/processed/metadata_split_binary3.csv를 받았는지 확인하세요.\n"
            "(없으면 로컬에서 python src/binclf/convert_binary_metadata.py 로 생성)"
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

    # 이원화 정본인지 방어 확인: 라벨은 0(우수)/1(불량) 두 값만 허용
    # (실수로 3클래스 CSV나 손상된 binary2.csv를 지정하면 여기서 즉시 실패)
    found_labels = set(metadata["model_label"].unique().tolist())
    if not found_labels <= {0, 1}:
        raise ValueError(
            f"model_label에 0/1 외 값이 있습니다: {sorted(found_labels)}\n"
            "이원화 CSV(metadata_split_binary3.csv)가 맞는지 확인하세요."
        )

    def take_split(split_name):
        split_data = metadata[
            metadata["split"] == split_name
        ].reset_index(drop=True)

        # 스모크 테스트: 클래스별로 절반씩 잘라 우수(소수 클래스)가 반드시 포함되게 함
        if limit_per_split > 0:
            split_data = (
                split_data
                .groupby("model_label", group_keys=False)
                .head(max(1, limit_per_split // 2))
                .reset_index(drop=True)
            )

        return split_data

    train_data = take_split("train")
    validation_data = take_split("validation")
    test_data = take_split("test")

    return train_data, validation_data, test_data


def compute_class_weights(train_dataframe):
    """train 분포에서 클래스 가중치를 계산한다. 공식: 전체 장수 / (2 × 해당 클래스 장수)

    기존 3클래스의 하드코딩 [3.79, 2.95, 0.42]는 3클래스 분포 기반이라 여기서는
    쓸 수 없다. 이원화 전체 데이터 기준 약 [5.68, 0.55]가 나온다.

    float()로 감싸는 이유: pandas 계산 결과(np.float64)를 체크포인트에 그대로
    저장하면 torch.load(weights_only=True)에서 로드가 거부됨.
    """
    class_counts = train_dataframe["model_label"].value_counts().sort_index()

    for class_index in [0, 1]:
        if class_counts.get(class_index, 0) == 0:
            raise ValueError(
                f"train에 라벨 {class_index}({class_names[class_index]}) 표본이 없습니다. "
                "LIMIT_PER_SPLIT이 너무 작거나 CSV가 잘못됐습니다."
            )

    total_count = len(train_dataframe)

    return [
        float(total_count / (2 * class_counts[class_index]))
        for class_index in [0, 1]
    ]


def build_dataloaders():
    """train/validation/test 데이터로더 3개를 만들어 반환한다.

    사용 예)
        from preprocess_binclf_efficientnet import build_dataloaders
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

    train_data, validation_data, test_data = load_split_dataframes()

    print("=========================")
    print("배치 크기 :", batch_size)
    print("데이터 로딩 워커 수 :", num_workers)

    if limit_per_split > 0:
        print(f"[스모크 모드] LIMIT_PER_SPLIT={limit_per_split} — 실험 결과에 사용 금지")

    for split_name, split_data in [
        ("Train", train_data),
        ("Validation", validation_data),
        ("Test", test_data),
    ]:
        distribution = split_data["model_label"].value_counts().sort_index().to_dict()
        print(f"{split_name} : {len(split_data)}장 | 분포(0=우수,1=불량) : {distribution}")

    print("클래스 가중치 (train 분포) :",
          [round(value, 4) for value in compute_class_weights(train_data)])

    train_loader, validation_loader, test_loader = build_dataloaders()

    # 첫 배치를 실제로 꺼내서 이미지가 잘 읽히는지 확인
    images, labels = next(iter(train_loader))

    print("==========================================")
    print("이미지 묶음 형태 :", images.shape)   # 예상: [배치, 3, 224, 224]
    print("정답 묶음 형태 :", labels.shape)
    print("이미지 자료형 :", images.dtype)      # 예상: torch.float32
    print("정답값 예시 :", labels[:10])
