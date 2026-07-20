from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from torch import nn
from torchvision.models import mobilenet_v2

from preprocess_data import test_loader

project_dir = Path(__file__).resolve().parent.parent

# 학습 때 저장한 최고 모델의 위치
best_model_path = project_dir / "model" / "best_mobilenet_v2_baseline.pth"

# 혼동 행렬 csv를 저장할 위치
confusion_matrix_path = project_dir / "data" / "processed" / "test_confusion_matrix_b..csv"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("사용 장치 : ", device)
print("평가 모델 : ", best_model_path.name)

# 학습 때와 동일한 MobileNetV2 구조새성
# 저장한 가중치를 불러올 것이므로 사전학습 가중치는 다시 받지 않음
model = mobilenet_v2(weights=None)

# 기존 마지막 계층의 입력 개수
input_features = model.classifier[1].in_features

# 학습 때와 동일하게 출력 클래스를 3개로 변경
model.classifier[1] = nn.Linear(
    input_features,
    3
)

# 저장한 체크포인트 불러오기
checkpoint = torch.load(
    best_model_path,
    map_location = device,
    weights_only = True
)

saved_epoch = checkpoint.get(
    "fine_tune_epoch",
    checkpoint.get(
       "eqoch", "정보없음"
    )
)

saved_validation_accuracy = checkpoint.get(
    "validation_accuracy",
    checkpoint.get(
        "validation_ccuract",
        "정보없음"
    )
)

print("저장된 Epoch : ", saved_epoch)

if isinstance(saved_validation_accuracy, float):
    print(
        f"저장 당시 Validation 정확도 : {saved_validation_accuracy:.2%}"
    )
else:
    print("저장 당시 Validation장확도 : ", saved_validation_accuracy)

# 체크포인트에 들어 있는 학습된 가중치를 모델에 적용
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

# 실제 정답과 모델 예측을 저장할 리스트
all_labels = []
all_predictions = []

total_count = 0
correct_count = 0

# Test에서는 모델을 수정하지 않으므로 기울기를 계산하지않음
with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 모델에 이미지를 입력해 예측 점수 계산
        outputs = model(images)

        # 우수, 보통, 불량 점수 중 가장 높은 위치를 선택
        predictions = outputs.argmax(dim=1)

        correct_count += (
            predictions == labels
        ).sum().item()

        # 전체 이미지 개수 누적
        total_count += labels.size(0)

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )
# 전체 Text 정확도
test_accuracy = correct_count / total_count

# 모델에서 사용하는 클래스 순서
class_labels = [0,1,2]

class_names = [
    "우수",
    "보통",
    "불량"
]

print("\nTest 평가 결과")
print("Test이미지 수 : ", total_count)
print("맞힌 이미지 수 : ", correct_count)
print("Test 정확도 : ",f"{test_accuracy:.2%}")

# 등급별 precision, recall, F1-score 출력
report = classification_report(
    all_labels,
    all_predictions,
    labels=class_labels,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n등급별 성능")
print(report)

#혼동행렬계산
matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=class_labels
)

# 혼동 행렬을 사람이 읽기쉰운 표로 변환
confusion_table = pd.DataFrame(
    matrix,
    index=[
        "실제_우수",
        "실제_보통",
        "실제_불량"
    ],
    columns=[
        "예측_우수",
        "예측_보통",
        "예측_불량"
    ]
)

print("\n혼동행렬")
print(confusion_table)

#혼동행렬csv로 저장
confusion_table.to_csv(
    confusion_matrix_path,
    encoding="utf-8-sig"
)

print(
    "\n혼동행렬 저장위치",
    confusion_matrix_path
)