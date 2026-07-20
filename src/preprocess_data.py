from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

project_dir = Path(__file__).resolve().parent.parent

metadata_path = project_dir / "data" / "processed" / "metadata_split.csv"

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

# MobileNetV2 사전학습 모델에 맞춘 평균과 표준편차
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

# Train이미지 전처리
train_transform = v2.Compose([
    #PIL이미지를 PyTorch 이미지 형태로 변환
    v2.ToImage(),

    v2.Resize(
        size=(224, 224),
        antialias=True
    ),
    #train만 50%확률로 좌우반전
    v2.RandomHorizontalFlip(p=0.5),
    #밝기 대비 약간 변경
    v2.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),
    #이미지 숫자를 float32 형식과 0~1 범위로 변환
    v2.ToDtype(
        torch.float32,
        scale=True
    ),
    #MobileNetV2입력 형식에 맞게 정규화
    v2.Normalize(
        mean=image_mean,
        std=image_std
    )
])

#Validation과 Test 이미지 전처리
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
        mean =image_mean,
        std=image_std
    )
])

#csv와 실제 이미지를 연결하는 데이터셋 클래스
class BuildingDataset(Dataset):
    
    def __init__(
        self,
        dataframe,
        project_directory,
        transform
    ):
        self.dataframe = dataframe
        self.project_directory = project_directory
        self.transform = transform

    # 데이터셋에 이미지가 몇 장 있는지 반환
    def __len__(self):
        return len(self.dataframe)
    
    # 지정된 순서의 이미지 한 장과 정답 반환
    def __getitem__(self, index):

        #CSV에서 index번째 행을 가져옴
        row = self.dataframe.iloc[index]
        #CSV 상대경로를 실제 전체 경로로변환
        image_path = (
            self.project_directory
            / row["image_path"]
        )

        # 이미지를 열고 RGB형식으로 통일
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        
        # 이미지 전처리
        image = self.transform(image)

        # 모델이 사용할 정답값
        label = int(row["model_label"])
        
        return image, label

# Train 데이터셋
train_dataset = BuildingDataset(
    dataframe=train_data,
    project_directory=project_dir,
    transform=train_transform
)

# Validation 데이터셋
validation_dataset = BuildingDataset(
    dataframe=validation_data,
    project_directory=project_dir,
    transform=evaluation_transform
)

# Test 데이터셋
test_dataset = BuildingDataset(
    dataframe = test_data,
    project_directory = project_dir,
    transform = evaluation_transform
)

# 한번에 모델에 전달할 이미지 수
batch_size = 16

#Train 데이터로더
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 0
)

#validation데이터 로더
validation_loader = DataLoader(
    validation_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 0
)

# Test데이터로더
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 0
)

# Train 데이터에서 첫 번째 묶음 가져오기
if __name__ =="__main__":

    images, labels = next(iter(train_loader))

    print("=========================")
    print("Train : ", len(train_dataset))
    print("validation : ", len(validation_dataset))
    print("Test : ", len(test_dataset))

    print("==========================================")
    print("이미지 묶음 형태 : ", images.shape)
    print("정답 묶음 형태 : ", labels.shape)
    print("이미지 자료형 : ", images.dtype)
    print("정답값 : ", labels)
