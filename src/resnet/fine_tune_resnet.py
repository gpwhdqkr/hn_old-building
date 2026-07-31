# ============================================================
# [사용법]
# ResNet 파인튜닝 스크립트 (2단계).
# 1단계 baseline 모델을 불러와서, 마지막 특징 블록(layer4)까지 동결을 풀고
# 아주 작은 학습률로 조금 더 학습합니다.
#
# 실행 (반드시 train_resnet.py를 먼저 실행해야 함. 런팟 = 리눅스):
#   RESNET_ARCH=resnet18 RUN_NAME=r18_e20 NUM_EPOCHS=20 python src/resnet/fine_tune_resnet.py
#
# [주의: 윈도우 PowerShell은 리눅스식 인라인 환경변수를 못 씁니다]
#   $env:RESNET_ARCH="resnet18"; $env:RUN_NAME="r18_e20"; $env:NUM_EPOCHS="20"
#   python src/resnet/fine_tune_resnet.py
#
# [환경변수] — train_resnet.py 실행 때와 반드시 같은 값을 써야 그 baseline을 찾습니다
#   RESNET_ARCH : resnet18 (기본) 또는 resnet50
#   RUN_NAME    : 실험 이름 (기본 "default")
#   NUM_EPOCHS  : 최대 파인튜닝 epoch 수 (기본 10). patience 조기 종료가 있어서
#                 20으로 지정해도 개선이 멈추면 먼저 끝남
#   BATCH_SIZE  : 배치 크기 (기본 128)
#
# 결과:
#   model/best_<RESNET_ARCH>_finetuned_<RUN_NAME>.pth
#   (validation 정확도가 baseline보다 좋아진 시점의 모델이 저장됨.
#    한 번도 안 좋아지면 baseline 상태 그대로 저장되어 있음 — 팀원과 동일 방식)
#
# 다음 단계 (같은 RESNET_ARCH와 RUN_NAME을 반드시 그대로 넘겨야 함):
#   RESNET_ARCH=resnet18 RUN_NAME=r18_e20 python src/resnet/evaluate_resnet.py
#
# [EfficientNet 실험과의 조건 차이 — 결과 해석 시 반드시 감안할 것]
# "마지막 특징 블록만 푼다"는 방침은 같지만, 푸는 규모가 모델마다 다릅니다.
#   EfficientNet-B0  features[-1] =  0.41M (전체의  8%)
#   ResNet18         layer4       =  8.4M  (전체의 72%)
#   ResNet50         layer4       = 15.0M  (전체의 59%)
# ResNet의 layer4는 위치상으로는 대응되지만 사실상 모델 대부분을 푸는 셈이라
# EfficientNet보다 공격적인 파인튜닝입니다. lr이 1e-5로 낮고 train이 26k장이라
# 학습 자체는 안정적이지만, 결과표에 조건 차이를 적어야 공정한 비교가 됩니다.
# (그래서 체크포인트에 trainable_param_count를 기록하고 evaluate가 출력합니다)
# ============================================================

import os
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torchvision.models import resnet18, resnet50

from preprocess_resnet import batch_size, build_dataloaders

project_dir = Path(__file__).resolve().parent.parent.parent

# 학습할 ResNet 종류: train_resnet.py 실행 때와 반드시 같은 값
RESNET_ARCH = os.environ.get("RESNET_ARCH", "resnet18")

# 실험 이름: train_resnet.py 실행 때와 반드시 같은 값을 써야 baseline을 찾음
RUN_NAME = os.environ.get("RUN_NAME", "default")

model_dir = project_dir / "model"

# 1단계에서 저장한 baseline 모델
baseline_model_path = model_dir / f"best_{RESNET_ARCH}_baseline_{RUN_NAME}.pth"

# 파인튜닝 결과를 저장할 경로
fine_tuned_model_path = model_dir / f"best_{RESNET_ARCH}_finetuned_{RUN_NAME}.pth"

# 최대 파인튜닝 epoch 수 (기본 10)
num_epochs = int(os.environ.get("NUM_EPOCHS", 10))

# 3회 연속 validation 정확도가 좋아지지 않으면 조기 종료 (팀원과 동일)
patience = 3

# 클래스 불균형 보정 가중치 (팀원과 동일)
class_weight_values = [3.79, 2.95, 0.42]

class_names = {
    0: "우수",
    1: "보통",
    2: "불량"
}


def create_resnet_model():
    """RESNET_ARCH에 맞는 ResNet 구조를 만들고 분류층을 3등급 출력으로 교체해서 반환한다.

    baseline 체크포인트의 가중치를 덮어씌울 것이므로 ImageNet 가중치는 받지 않는다
    (weights=None). train_resnet.py의 같은 이름 함수와 반드시 동일한 구조를 만들어야
    한다. 한쪽만 바꾸면 baseline 가중치 로드가 실패한다.
    """
    if RESNET_ARCH == "resnet18":
        model = resnet18(weights=None)

    elif RESNET_ARCH == "resnet50":
        model = resnet50(weights=None)

    else:
        raise ValueError(
            'RESNET_ARCH는 "resnet18" 또는 "resnet50"이어야 합니다: '
            f"{RESNET_ARCH}"
        )

    model.fc = nn.Linear(
        model.fc.in_features,
        3
    )

    return model


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
    # float() 변환 필수: numpy 타입을 체크포인트에 저장하면
    # torch.load(weights_only=True)에서 로드가 거부됨
    defect_f1 = float(per_class_f1[2])

    return average_loss, accuracy, defect_f1


def main():
    print("모델 종류 (RESNET_ARCH) :", RESNET_ARCH)
    print("실험 이름 (RUN_NAME) :", RUN_NAME)
    print("배치 크기 :", batch_size)

    if not baseline_model_path.exists():
        raise FileNotFoundError(
            f"baseline 모델을 찾을 수 없습니다: {baseline_model_path}\n"
            "먼저 같은 RESNET_ARCH와 RUN_NAME으로 train_resnet.py를 실행하세요.\n"
            f"(예: RESNET_ARCH={RESNET_ARCH} RUN_NAME={RUN_NAME} "
            "python src/resnet/train_resnet.py)"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("사용 장치 :", device)

    if device.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))

    train_loader, validation_loader, _ = build_dataloaders()

    # 가중치를 체크포인트에서 불러올 것이므로 ImageNet 가중치는 다시 받지 않음
    model = create_resnet_model()

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
    # 전체를 동결한 뒤, 마지막 특징 블록(layer4)과 분류층(fc)만 동결 해제
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    model = model.to(device)

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
        f"학습 파라미터 : {trainable_param_count / 1e6:.2f}M "
        f"({trainable_param_count / total_param_count:.0%})"
    )

    class_weights = torch.tensor(
        class_weight_values,
        dtype=torch.float32,
        device=device
    )
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # 마지막 특징 블록은 아주 조금만, 분류층은 그보다 조금 더 크게 수정
    # (팀원의 fine_tune_model.py / EfficientNet과 동일한 차등 학습률)
    optimizer = Adam(
        [
            {
                "params": model.layer4.parameters(),
                "lr": 0.00001
            },
            {
                "params": model.fc.parameters(),
                "lr": 0.0001
            }
        ]
    )

    # baseline의 validation 정확도를 시작 기준으로 사용 (팀원과 동일)
    # (이보다 좋아질 때만 파인튜닝 모델을 갱신)
    best_validation_accuracy = baseline_validation_accuracy
    no_improvement_count = 0

    # 체크포인트에 매번 넣을 공통 정보
    def build_checkpoint(fine_tune_epoch, validation_accuracy, validation_defect_f1):
        return {
            "epoch": checkpoint.get("epoch", 0),
            "fine_tune_epoch": fine_tune_epoch,
            "model_state_dict": model.state_dict(),
            "validation_accuracy": validation_accuracy,
            "validation_defect_f1": validation_defect_f1,
            "class_names": class_names,
            "class_weight_values": class_weight_values,
            "source_model": baseline_model_path.name,
            "run_name": RUN_NAME,
            "resnet_arch": RESNET_ARCH,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "trainable_param_count": trainable_param_count,
            "total_param_count": total_param_count,
            "unfrozen_layers": "layer4 + fc"
        }

    # 파인튜닝이 전혀 개선되지 않아도 baseline 상태가 finetuned 파일에
    # 남아 있도록 먼저 저장해 둠 (팀원과 동일한 방식)
    torch.save(
        build_checkpoint(
            fine_tune_epoch=0,
            validation_accuracy=baseline_validation_accuracy,
            validation_defect_f1=checkpoint.get("validation_defect_f1", -1.0)
        ),
        fine_tuned_model_path
    )

    training_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n==== Fine-tuning Epoch {epoch}/{num_epochs} ====")

        # ---- Train ----
        model.train()

        # 동결된 앞쪽 층은 평가 상태로 유지
        # (가중치뿐 아니라 BatchNorm 통계도 바뀌지 않게 함 — 팀원/EfficientNet과 동일 기법)
        # ResNet은 컨테이너가 없어서 앞쪽 모듈을 하나씩 지정해야 함
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()

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
                build_checkpoint(
                    fine_tune_epoch=epoch,
                    validation_accuracy=validation_accuracy,
                    validation_defect_f1=validation_defect_f1
                ),
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
