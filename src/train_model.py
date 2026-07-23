import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torchvision.models import(
    MobileNet_V2_Weights,
    mobilenet_v2
)

from preprocess_data import (
    train_loader,
    validation_loader
)

project_dir = Path(__file__).resolve().parent.parent

#학습된 모델 저장할 폴더
model_dir = project_dir / "model"

# model없으면 생성
model_dir.mkdir(
    parents=True,
    exist_ok=True
)

#가장 성능이 좋은 모델을 저장할 경로
best_model_path = model_dir / "best_mobilenet_v2_baseline.pth"

#GPU를 사용할 수 있으면 GPU,
# 사용할 수 없으면 CPU 선택
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("사용 장치 : ", device)

# ImageNet으로 미리 학습된 MobileNetV2 가중치
weights = MobileNet_V2_Weights.DEFAULT

# 사전학습된 MobileNetV2 생성
model = mobilenet_v2(
    weights = weights
)

# 기존 특징 추출 부분의 가중치를 고정
for parameter in model.features.parameters():
    parameter.requires_grad = False

# for parameter in model.features[-1].parameters():
#     parameter.requires_gred = True

# 기존 마지막 분류 계층이 받는 입력 개수
input_features = model.classifier[1].in_features

#기존 1000개 분휴 출력을
# 우수 보통 불량 3개 출력으로 변경
model.classifier[1] = nn.Linear(
    input_features,
    3
)

# 모델을 cpu 또는 GPU로 이동
model = model.to(device)

class_weights = torch.tensor(
    [3.79,2.95,0.42],
    dtype=torch.float32,
    device=device
)

loss_function = nn.CrossEntropyLoss(
    weight = class_weights
)

# 마지막 분류 계층만 학습하도록 설정
optimizer = Adam(
    model.classifier.parameters(),
    lr=0.001
)

#처음에 3회로 코드정상작동 확인 
num_epochs = 1

# 가장 높았던 Validation정확도
best_validation_accuracy = -1.0

# 전체 학습 시간 측정 시작
training_start_time = time.time()

# 전체 Train데이터를 num_epochs만큼 반복
for epoch in range(1, num_epochs + 1):
    print(f"\n==== Epoch {epoch}/{num_epochs} ====")

    #train단계
    # 모델을 학습상태로 변경
    model.train()

    train_loss_sum = 0.0
    train_correct_count = 0
    train_total_count = 0

    #train 이미지를 배치 단위로 꺼냄
    for images, labels in train_loader:
        
        #이미지와 정답을 같은 장치로 이동
        images = images.to(
            device,
            non_blocking=True
        )
        labels = labels.to(
            device,
            non_blocking=True
        )

        # 이전 배치에서 계산된 기울기를 초기화
        optimizer.zero_grad()
        #이미지를 모델에 넣어 예측값 계산
        outputs = model(images)
        # 모델 예측과 실제 정답의 차이 계산
        loss = loss_function(
            outputs,
            labels
        )

        # 손실을 기준으로 기울기 계산
        loss.backward()

        # 계산한 기울기를 이용해 모델 수정
        optimizer.step()

        # 현재 배치의 손실을 누적
        train_loss_sum += (
            loss.item()
            * images.size(0)
        )

        #세 출력 중 값이 가장 큰 위치를 예측 등급으로 선택
        predictions = outputs.argmax(
            dim = 1
        )

        #맞힌개수 누적
        train_correct_count += (
            predictions == labels
        ).sum().item()

        # 처리한 전체 이미지 수 누적
        train_total_count += labels.size(0)
    
    #Train 전체의 평균 손실
    train_loss = (
        train_loss_sum
        / train_total_count
      )

    # train 전체 정확도
    train_accuracy = (
        train_correct_count
        / train_total_count
    )

    #Validation 단계
    # 모델을 평가 상태로 변경
    model.eval()

    validation_loss_sum = 0.0
    
    validation_correct_count= 0
    
    validation_total_count = 0

    #Validation에서는 모델을 수정하지 않으므로
    # 기울기 계산을 중단
    with torch.no_grad():
        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            # 예측만 수행
            outputs = model(images)

            #손실계간
            loss = loss_function(
                outputs,
                labels
            )

            validation_loss_sum += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim = 1
            )

            validation_correct_count +=(
                predictions == labels
            ).sum().item()

            validation_total_count += (
                labels.size(0)
            )
    #Validation 퍙균 손실
    validation_loss = (
        validation_loss_sum
        / validation_total_count
    )

    #validation 정확도
    validation_accuracy = (
        validation_correct_count
        / validation_total_count
    )

    #한 Epoch의 결과 출력
    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%}"    
    )

    print(
        f"Validation Loss: {validation_loss:.4f} |"
        f"Validation Accuracy: {validation_accuracy:.2%}" 
    )

    # 현재 Validation 정확도가 이전 최고 기록보다 노ㅠ다면 저장
    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        #모델과 필요한 정보를 함께 저장
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_accuracy": validation_accuracy,
                "class_names": {
                    0: "우수",
                    1: "보통",
                    2: "불량"
                }
            },
            best_model_path
        )

        print(
            "최고 성능 모델 저장:",
            best_model_path
        )
# 전체 학습 시간 계산
training_seconds = (
    time.time() - training_start_time
)

print("==========================================")
print(
    "최고 Validation 정확도: ",
    f"{best_validation_accuracy:.2%}"
)
print(
    "학습 시간: ",
    f"{training_seconds:.1f}초"
)
print(
    "저장된 모델:",
    best_model_path
)