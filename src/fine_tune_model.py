import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torchvision.models import mobilenet_v2

from preprocess_data import (
    train_loader,
    validation_loader
)

project_dir = Path(__file__).resolve().parent.parent

model_dir = project_dir / "model"

baseline_model_path = model_dir / "best_mobilenet_v2_baseline.pth"

fine_tuned_model_path = (
    model_dir / "best_mobilenet_v2_finetuned.pth"
)

if not baseline_model_path.exists():
    raise FileNotFoundError(
        f"기본 모델을 찾을 수 없습니다: {baseline_model_path}"
    )

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("사용 장치:", device)

#가중치를 불러올 것이으모 ImageNet 가중치를 다시 다운로드 하지 않음
model = mobilenet_v2(weights=None)

#MobileNetV2 마지막 분류 계층의 입력 크기
input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    3
)

#기본모델 가중치 불러오기
checkpoint= torch.load(
    baseline_model_path,
    map_location=device,
    weights_only=True
)

#기본 모델이 학습한 가중치를 현재 모델에 넣는다
model.load_state_dict(
    checkpoint["model_state_dict"]
)

baseline_validation_accuracy = checkpoint.get(
    "validation_accuracy",
    checkpoint.get(
        "calidation_accuracy",
        -1.0
    )
)

print(
    "기본 모델 Validation 정확도:",
    f"{baseline_validation_accuracy:.2%}"
)

#학습할 부분 정하기
for parameter in model.features.parameters():
    parameter.requires_grad = False

for parameter in model.features[-1].parameters():
    parameter.requires_grad = True

model = model.to(device)

class_weights = torch.tensor(
    [3.79,2.95,0.42],
    dtype=torch.float32,
    device=device
)

loss_function = nn.CrossEntropyLoss(
    weight = class_weights
)

# 특징 추출 블록은 아주 조금 수정하고 분류기는 그보다 조금 더 크게 수정한다
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

# 최대 미세 조정 횟수
num_epochs = 10

# 3회 연속 Validation 성능이 좋아지지 않으면 중단
patience = 3
no_improvement_count = 0

# 기본 모델의 Validation 정확도를 시작 기준으로 사용
best_validation_accuracy = baseline_validation_accuracy

# 미세 조정이 전혀 개선되지 않더라고 기본 모델 상태가 finetuned 파일에 남도록 먼저 저장
torch.save(
    {
        "epoch": checkpoint.get("eqoch", 0),
        "fine_tune_epoch": 0,
        "model_state_dict": model.state_dict(),
        "validation_ccuracy": baseline_validation_accuracy,
        "class_names": {
            0: "우수",
            1: "보통",
            2: "불량"
        },
        "source_model": baseline_model_path.name
    },
    fine_tuned_model_path
)

training_start_time = time.time()

#미세 조정 학습
for epoch in range(1, num_epochs + 1):
    print(f"\n==== Fine_tuning Epoch {epoch}/{num_epochs}")

    #train
    model.train()

    # 고정된 앞쪽 특징 블록은 평가 상태로 유지한다
    # 가중치뿐 아니라 BatchNorm 상태도 바뀌지 않게 한다
    for block in model.features[:-1]:
        block.eval()

    train_loss_sum = 0.0
    train_correct_count = 0
    train_total_count = 0

    for images, labels in train_loader:
        
        images = images.to(device)
        labels = labels.to(device)

        #이전 배치 기울기 제거
        optimizer.zero_grad()
        #모델 예측
        outputs = model(images)

        # 예측과 실제 정답 차이 계산
        loss = loss_function(
            outputs,
            labels
        )

        #수정 방향 계산
        loss.backward()

        # 마지막 특징 블록과 분류가 수정
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)
        
        predictions = outputs.argmax(dim=1)

        train_correct_count += (predictions == labels).sum().item()

        train_total_count += labels.size(0)
    
    train_loss = train_loss_sum / train_total_count

    train_accuracy = train_correct_count / train_total_count

    #Validation

    model.eval()

    validation_loss_sum = 0.0
    validation_correct_count = 0
    validation_total_count = 0

    with torch.no_grad():
        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_function(
                outputs,
                labels
            )

            validation_loss_sum += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            validation_correct_count += (predictions == labels).sum().item()

            validation_total_count += labels.size(0)
        
    validation_loss = validation_loss_sum / validation_total_count

    validation_accuracy = validation_correct_count / validation_total_count

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%}"
    )

    print(
        f"Validation Loss: {validation_loss:.4f} | "
        f"Validation Accuracy: "
        f"{validation_accuracy:.2%}"
    )

    #최고 성능 모델 저장
    if validation_accuracy  > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        no_improvement_count = 0

        torch.save(
            {
                "epoch": epoch,
                "fine_tune_epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_accuracy":validation_accuracy ,
                "class_names":{
                    0: "우수",
                    1: "보통",
                    2: "불량"
                },
                "source_model": baseline_model_path.name
            },
            fine_tuned_model_path
        )

        print("미세 조정 최고 모델 저장 : ", fine_tuned_model_path)

    else:
        no_improvement_count += 1
        print(f"Validation 성능 개선 없음: {no_improvement_count} / {patience}")
    
    #3회 연속 좋아지지 않으면 조기 종료
    if no_improvement_count >= patience:
        print("validation 성능이 개선되지 않아 중단합니다")
        break

training_seconds = (
    time.time()-training_start_time
)

print("\n=========================================")
print("미세조정완료")
print(f"최고 Validation 정확도: {best_validation_accuracy:.2%}")
print(f"학습시간 : {training_seconds:.1f}초")
print("저장된 모델 : ", fine_tuned_model_path)