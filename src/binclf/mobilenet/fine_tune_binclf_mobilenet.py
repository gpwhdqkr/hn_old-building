# ============================================================
# [사용법]
# 이원화(2클래스: 우수 vs 불량) MobileNetV2 파인튜닝 스크립트 (2단계).
# 1단계 baseline 모델을 불러와서, 마지막 특징 블록(features[-1])까지 동결을 풀고
# 아주 작은 학습률로 조금 더 학습합니다.
#
# 기존 src/train_model.py와 학습 조건(동결 범위/lr/batch 50)은 동일하나
# 코드 구조는 binclf 표준(resnet 스타일)을 따름.
# (파인튜닝 조건은 기존 src/fine_tune_model.py의 features[-1]+classifier 해제,
#  차등 학습률 1e-5/1e-4, patience 3을 그대로 이식)
#
# 실행 (반드시 train_binclf_mobilenet.py를 먼저 실행해야 함. 런팟 = 리눅스):
#   RUN_NAME=bin1 NUM_EPOCHS=20 python src/binclf/mobilenet/fine_tune_binclf_mobilenet.py
#
# [주의: 윈도우 PowerShell은 리눅스식 인라인 환경변수를 못 씁니다]
#   $env:RUN_NAME="bin1"; $env:NUM_EPOCHS="20"
#   python src/binclf/mobilenet/fine_tune_binclf_mobilenet.py
#
# [환경변수] — train_binclf_mobilenet.py 실행 때와 반드시 같은 값을 써야 그 baseline을 찾습니다
#   RUN_NAME    : 실험 이름 (기본 "default")
#   NUM_EPOCHS  : 최대 파인튜닝 epoch 수 (기본 10). patience 조기 종료가 있어서
#                 20으로 지정해도 개선이 멈추면 먼저 끝남
#   BATCH_SIZE  : 배치 크기 (기본 50)
#   DATA_DIR    : 데이터가 다른 위치에 있을 때만 지정
#
# 결과:
#   model/best_mobilenet_v2_binclf_finetuned_<RUN_NAME>.pth
#   (validation macro F1이 baseline보다 좋아진 시점의 모델이 저장됨.
#    한 번도 안 좋아지면 baseline 상태 그대로 저장되어 있음 — 기존 방식과 동일)
#
# 다음 단계 (같은 RUN_NAME을 반드시 그대로 넘겨야 함):
#   RUN_NAME=bin1 python src/binclf/mobilenet/evaluate_binclf_mobilenet.py
#
# [기존 3클래스 실험과의 차이]
# 동결 해제 범위(features[-1]+classifier)/차등 학습률(1e-5/1e-4)/patience(3)는
# 기존 src/fine_tune_model.py와 동일. 다른 것은 라벨 정의(이원화),
# 클래스 가중치(분포 재계산), 선택 기준(macro F1)뿐.
# ============================================================

import os
import time

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torchvision.models import mobilenet_v2

from preprocess_binclf_mobilenet import (
    batch_size,
    build_dataloaders,
    class_names,
    compute_class_weights,
    load_split_dataframes,
    model_dir
)

# 모델 아키텍처: MobileNetV2 고정 (resnet 버전과 달리 환경변수 분기 없음)
MODEL_ARCH = "mobilenet_v2"

# 실험 이름: train_binclf_mobilenet.py 실행 때와 반드시 같은 값을 써야 baseline을 찾음
RUN_NAME = os.environ.get("RUN_NAME", "default")

# 1단계에서 저장한 baseline 모델
baseline_model_path = model_dir / f"best_{MODEL_ARCH}_binclf_baseline_{RUN_NAME}.pth"

# 파인튜닝 결과를 저장할 경로
fine_tuned_model_path = model_dir / f"best_{MODEL_ARCH}_binclf_finetuned_{RUN_NAME}.pth"

# 최대 파인튜닝 epoch 수 (기본 10)
num_epochs = int(os.environ.get("NUM_EPOCHS", 10))

# 3회 연속 validation macro F1이 좋아지지 않으면 조기 종료 (기존과 동일한 patience)
patience = 3


def create_mobilenet_model():
    """MobileNetV2 구조를 만들고 분류층을 2클래스 출력으로 교체해서 반환한다.

    baseline 체크포인트의 가중치를 덮어씌울 것이므로 ImageNet 가중치는 받지 않는다
    (weights=None). train_binclf_mobilenet.py의 같은 이름 함수와 반드시 동일한 구조를
    만들어야 한다. 한쪽만 바꾸면 baseline 가중치 로드가 실패한다.
    """
    model = mobilenet_v2(weights=None)

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        2
    )

    return model


def evaluate_on_loader(model, data_loader, loss_function, device):
    """주어진 데이터로더 전체에 대해 손실/정확도/F1을 계산한다.

    반환: (loss, accuracy, 우수 F1, 불량 F1, macro F1)
    """
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
    print("모델 종류 (MODEL_ARCH) :", MODEL_ARCH)
    print("실험 이름 (RUN_NAME) :", RUN_NAME)
    print("배치 크기 :", batch_size)
    print("라벨 정의 : 0=우수, 1=불량(보통 병합)")

    if not baseline_model_path.exists():
        raise FileNotFoundError(
            f"baseline 모델을 찾을 수 없습니다: {baseline_model_path}\n"
            "먼저 같은 RUN_NAME으로 train_binclf_mobilenet.py를 실행하세요.\n"
            f"(예: RUN_NAME={RUN_NAME} "
            "python src/binclf/mobilenet/train_binclf_mobilenet.py)"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("사용 장치 :", device)

    if device.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))

    # 이원화 train 분포에서 클래스 가중치 계산 (baseline 학습 때와 동일한 방식)
    train_data, _, _ = load_split_dataframes()
    class_weight_values = compute_class_weights(train_data)

    print("클래스 가중치 (train 분포로 계산) :",
          [round(value, 4) for value in class_weight_values])

    train_loader, validation_loader, _ = build_dataloaders()

    # 가중치를 체크포인트에서 불러올 것이므로 ImageNet 가중치는 다시 받지 않음
    model = create_mobilenet_model()

    # baseline이 학습한 가중치를 불러와 적용
    checkpoint = torch.load(
        baseline_model_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    baseline_validation_macro_f1 = checkpoint.get("validation_macro_f1", -1.0)
    print(f"baseline Validation Macro F1: {baseline_validation_macro_f1:.4f}")

    # 학습할 부분 정하기:
    # 전체를 동결한 뒤, 마지막 특징 블록(features[-1])과 분류층(classifier)만 동결 해제
    # (기존 src/fine_tune_model.py와 동일한 범위)
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-1].parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
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
    # (기존 src/fine_tune_model.py와 동일한 차등 학습률)
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

    # baseline의 validation macro F1을 시작 기준으로 사용
    # (이보다 좋아질 때만 파인튜닝 모델을 갱신)
    best_validation_macro_f1 = baseline_validation_macro_f1
    no_improvement_count = 0

    # 체크포인트에 매번 넣을 공통 정보
    def build_checkpoint(
        fine_tune_epoch,
        validation_macro_f1,
        validation_accuracy,
        validation_good_f1,
        validation_defect_f1
    ):
        return {
            "task": "binclf",
            "label_definition": "0=우수, 1=불량(보통 병합)",
            "metadata_file": "metadata_split_binary3.csv",
            "selection_metric": "macro_f1",
            "epoch": checkpoint.get("epoch", 0),
            "fine_tune_epoch": fine_tune_epoch,
            "model_state_dict": model.state_dict(),
            "validation_macro_f1": validation_macro_f1,
            "validation_accuracy": validation_accuracy,
            "validation_good_f1": validation_good_f1,
            "validation_defect_f1": validation_defect_f1,
            "class_names": class_names,
            "class_weight_values": class_weight_values,
            "source_model": baseline_model_path.name,
            "run_name": RUN_NAME,
            "model_arch": MODEL_ARCH,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "trainable_param_count": trainable_param_count,
            "total_param_count": total_param_count,
            "unfrozen_layers": "features[-1] + classifier"
        }

    # 파인튜닝이 전혀 개선되지 않아도 baseline 상태가 finetuned 파일에
    # 남아 있도록 먼저 저장해 둠 (기존 방식과 동일)
    torch.save(
        build_checkpoint(
            fine_tune_epoch=0,
            validation_macro_f1=baseline_validation_macro_f1,
            validation_accuracy=checkpoint.get("validation_accuracy", -1.0),
            validation_good_f1=checkpoint.get("validation_good_f1", -1.0),
            validation_defect_f1=checkpoint.get("validation_defect_f1", -1.0)
        ),
        fine_tuned_model_path
    )

    training_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n==== Fine-tuning Epoch {epoch}/{num_epochs} ====")

        # ---- Train ----
        model.train()

        # 동결된 앞쪽 특징 블록은 평가 상태로 유지
        # (가중치뿐 아니라 BatchNorm 통계도 바뀌지 않게 함 — 기존과 동일 기법)
        # MobileNetV2는 features가 Sequential 컨테이너라 슬라이스로 한 번에 지정 가능
        for block in model.features[:-1]:
            block.eval()

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
            no_improvement_count = 0

            torch.save(
                build_checkpoint(
                    fine_tune_epoch=epoch,
                    validation_macro_f1=validation_macro_f1,
                    validation_accuracy=validation_accuracy,
                    validation_good_f1=validation_good_f1,
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
    print(f"최고 Validation Macro F1: {best_validation_macro_f1:.4f}")
    print(f"학습 시간 : {training_seconds:.1f}초")
    print("저장된 모델 :", fine_tuned_model_path)


if __name__ == "__main__":
    main()
