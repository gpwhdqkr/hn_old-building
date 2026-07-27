# ============================================================
# [사용법]
# 이원화(2클래스: 우수 vs 불량) 결정 임계값(threshold) 보정 스크립트
# — 재학습 없이 기존 체크포인트만 사용.
#
# 학습된 이원화 EfficientNet-B0 체크포인트를 읽어 validation 셋에서 "불량 확률이
# 얼마 이상이면 불량으로 판정할지"(threshold)를 탐색하고, 확정된 threshold
# 하나로 test를 1회 평가합니다.
# threshold 탐색은 validation에서만 수행 (test 누출 없음).
#
# [3클래스 bias sweep과 다른 점 — 핵심 로직 재설계]
# 3클래스 tune_threshold_resnet.py는 "보통(1) logit에 bias를 더하는" 방식인데,
# 이원화에서는 보통 클래스 자체가 없으므로 그 방식이 무의미합니다. 2클래스는
# softmax 불량 확률 하나로 결정 규칙이 완전히 표현되므로, 불량 확률의 임계값을
# 직접 탐색합니다. (threshold=0.5가 기존 argmax와 동일한 결정)
#
# 기존 코드/체크포인트/평가 결과는 일절 수정하지 않습니다.
# 결과는 test_results/binclf/efficientnet_b0/<RUN_NAME>_threshold/ 아래에만 새로 저장됩니다.
#
# 실행 (런팟 = 리눅스):
#   RUN_NAME=bin1 python src/binclf/efficientnet/tune_threshold_binclf_efficientnet.py
#
# 실행 (윈도우 PowerShell):
#   $env:RUN_NAME="bin1"
#   python src/binclf/efficientnet/tune_threshold_binclf_efficientnet.py
#
# [환경변수]
#   RUN_NAME     : 학습 때 쓴 실험 이름 (기본 "default")
#   EVAL_TARGET  : baseline 또는 finetuned (기본 "finetuned")
#   RECALL_FLOOR : 불량 recall 하한 (기본 0.85).
#                  "불량 recall이 이 값 이상"인 threshold 중에서 macro F1 최대를
#                  선택 (동률이면 우수 F1 최대)
#   NUM_WORKERS  : 데이터 로딩 워커 수 (기본 4)
#   BATCH_SIZE   : 배치 크기 (기본 64 — CPU에서도 무난한 값)
#   DATA_DIR     : data/ 와 model/ 이 있는 프로젝트 루트 (기본: 이 저장소).
#                  워크트리처럼 데이터가 없는 곳에서 돌릴 때 원본 저장소를 지정
#                  (preprocess_binclf_efficientnet가 읽어서 경로에 반영함)
#
# GPU가 없어도 됩니다. 학습이 아니라 추론 2회(validation 1회 + test 1회)뿐이라
# CPU로도 동작하며, threshold 탐색은 저장해 둔 확률에 대한 배열 연산이라 수 초면 끝납니다.
# ============================================================

import os

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0

# 전처리/경로/라벨 검증은 이원화 공용 정의를 그대로 가져옴
# (load_split_dataframes가 binary3 CSV의 0/1 라벨 방어 검증까지 수행)
from preprocess_binclf_efficientnet import (
    RAW_PATH_MARKER,
    data_dir,
    evaluation_transform,
    load_split_dataframes,
    model_dir,
    processed_images_dir,
    test_results_root
)

# 모델 구조 이름 (파일명/결과 폴더명에 사용 — EfficientNet은 B0 고정)
MODEL_ARCH = "efficientnet_b0"

RUN_NAME = os.environ.get("RUN_NAME", "default")
EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")

# 불량 recall 하한: 이 값 밑으로 떨어뜨리는 threshold는 후보에서 제외
RECALL_FLOOR = float(os.environ.get("RECALL_FLOOR", 0.85))

# 추론 2회뿐이라 preprocess의 기본값(128)까지 필요 없음 — CPU에서도 무난한 64
batch_size = int(os.environ.get("BATCH_SIZE", 64))
num_workers = int(os.environ.get("NUM_WORKERS", 4))

# 보정할 체크포인트 ("binclf" 토큰이 있어 기존 3클래스 모델과 절대 겹치지 않음)
best_model_path = (
    model_dir
    / f"best_{MODEL_ARCH}_binclf_{EVALUATION_TARGET}_{RUN_NAME}.pth"
)

# 결과는 기존 평가 결과와 다른 폴더에 저장 (덮어쓰기 없음)
results_dir = test_results_root / MODEL_ARCH / f"{RUN_NAME}_threshold"

class_names = ["우수", "불량"]

# 탐색할 threshold 후보: 0.02 ~ 0.98을 0.01 간격으로 (97개)
# 양 끝(0.00/1.00 근처)은 전부 한 클래스로 찍는 퇴화 구간이라 제외
threshold_candidates = [
    round(0.02 + step * 0.01, 2) for step in range(97)
]


class ThresholdDataset(Dataset):
    """binary3 CSV 한 split의 이미지와 정답(0=우수, 1=불량)을 반환하는 데이터셋.

    preprocess_binclf_efficientnet.BuildingDataset과 동일한 동작이지만, 이미지 폴더
    위치를 모듈 전역이 아니라 생성자 인자로 받는다. (3클래스 tune_threshold와
    같은 구조 유지 — 기존 코드는 수정하지 않는다.)
    """

    def __init__(self, dataframe, images_dir, transform):
        self.dataframe = dataframe
        self.images_dir = images_dir
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
        label = int(row["model_label"])

        return image, label


def create_efficientnet_model():
    """저장된 체크포인트를 덮어씌울 빈 EfficientNet-B0 구조를 만든다
    (evaluate_binclf_efficientnet.py와 동일)."""
    model = efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    return model


def collect_logits(model, loader, device):
    """한 split 전체를 추론해 (logits, labels) 텐서를 반환한다. 모델 수정 없음."""
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            all_logits.append(outputs.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def metrics_with_threshold(probabilities, labels, threshold):
    """불량 확률이 threshold 이상이면 불량(1)으로 판정했을 때의 주요 지표를 계산한다."""
    predictions = (probabilities >= threshold).long().numpy()

    labels_numpy = labels.numpy()

    return {
        "threshold": threshold,
        "predictions": predictions,
        "accuracy": (predictions == labels_numpy).mean(),
        "good_f1": f1_score(
            labels_numpy, predictions, labels=[0], average=None,
            zero_division=0
        )[0],
        "good_recall": recall_score(
            labels_numpy, predictions, labels=[0], average=None,
            zero_division=0
        )[0],
        "defect_recall": recall_score(
            labels_numpy, predictions, labels=[1], average=None,
            zero_division=0
        )[0],
        "defect_f1": f1_score(
            labels_numpy, predictions, labels=[1], average=None,
            zero_division=0
        )[0],
        "macro_f1": f1_score(
            labels_numpy, predictions, average="macro", zero_division=0
        ),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("사용 장치 :", device)
    print("평가 모델 :", best_model_path.name)
    print("데이터 위치 :", data_dir)
    print("라벨 정의 : 0=우수, 1=불량(보통 병합)")
    print("불량 recall 하한 :", RECALL_FLOOR)

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {best_model_path}\n"
            "RUN_NAME / EVAL_TARGET / DATA_DIR 값을 확인하세요."
        )

    # binary3 CSV 로드 + 0/1 라벨 방어 검증 (preprocess 공용 함수)
    _, validation_data, test_data = load_split_dataframes()

    print("Validation 장수 :", len(validation_data))
    print("Test 장수 :", len(test_data))

    validation_loader = DataLoader(
        ThresholdDataset(
            validation_data, processed_images_dir, evaluation_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        ThresholdDataset(
            test_data, processed_images_dir, evaluation_transform
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = create_efficientnet_model()

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ---------- 1단계: validation 1회 추론 후 threshold 탐색 ----------
    print("\nValidation 추론 중 (1회만 실행됨)...")
    validation_logits, validation_labels = collect_logits(
        model, validation_loader, device
    )

    # softmax로 불량(1) 확률만 뽑는다 — 2클래스는 이 값 하나로 결정이 완전히 표현됨
    validation_probabilities = torch.softmax(validation_logits, dim=1)[:, 1]

    sweep_rows = []
    for threshold in threshold_candidates:
        result = metrics_with_threshold(
            validation_probabilities, validation_labels, threshold
        )
        sweep_rows.append({
            "threshold": result["threshold"],
            "우수_F1": round(result["good_f1"], 4),
            "우수_recall": round(result["good_recall"], 4),
            "불량_recall": round(result["defect_recall"], 4),
            "불량_F1": round(result["defect_f1"], 4),
            "macro_F1": round(result["macro_f1"], 4),
            "accuracy": round(result["accuracy"], 4),
        })

    sweep_table = pd.DataFrame(sweep_rows)

    # 불량 recall 하한을 지키는 후보 중 macro F1이 최대인 threshold 선택
    # (동률이면 우수 F1이 큰 쪽 — 이원화의 목적이 우수 구분력 확보이므로)
    eligible = sweep_table[sweep_table["불량_recall"] >= RECALL_FLOOR]

    if eligible.empty:
        raise RuntimeError(
            f"불량 recall {RECALL_FLOOR} 이상을 만족하는 threshold가 없습니다. "
            "RECALL_FLOOR를 낮춰서 다시 실행하세요."
        )

    best_macro_f1 = eligible["macro_F1"].max()
    tied_candidates = eligible[eligible["macro_F1"] == best_macro_f1]
    best_row = tied_candidates.loc[tied_candidates["우수_F1"].idxmax()]
    chosen_threshold = float(best_row["threshold"])

    print("\n[validation 탐색 결과]")
    print(f"선택된 threshold : {chosen_threshold}")
    print(f"validation macro F1 : {best_row['macro_F1']:.4f}")
    print(f"validation 우수 F1 : {best_row['우수_F1']:.4f}")
    print(f"validation 불량 recall : {best_row['불량_recall']:.4f}")

    # ---------- 2단계: 확정된 threshold로 test 1회 평가 ----------
    print("\nTest 추론 중 (1회만 실행됨)...")
    test_logits, test_labels = collect_logits(model, test_loader, device)

    test_probabilities = torch.softmax(test_logits, dim=1)[:, 1]

    # 기준선 threshold=0.5: 2클래스 softmax에서 불량 확률 0.5 이상 = argmax와
    # 동일한 결정 (= 기존 evaluate_binclf_efficientnet.py 결과와 같아야 정상)
    baseline = metrics_with_threshold(test_probabilities, test_labels, 0.5)
    adjusted = metrics_with_threshold(
        test_probabilities, test_labels, chosen_threshold
    )

    labels_numpy = test_labels.numpy()

    baseline_report = classification_report(
        labels_numpy, baseline["predictions"],
        labels=[0, 1], target_names=class_names,
        digits=4, zero_division=0
    )

    adjusted_report = classification_report(
        labels_numpy, adjusted["predictions"],
        labels=[0, 1], target_names=class_names,
        digits=4, zero_division=0
    )

    matrix_index = ["실제_우수", "실제_불량"]
    matrix_columns = ["예측_우수", "예측_불량"]

    baseline_matrix = pd.DataFrame(
        confusion_matrix(
            labels_numpy, baseline["predictions"], labels=[0, 1]
        ),
        index=matrix_index,
        columns=matrix_columns
    )

    adjusted_matrix = pd.DataFrame(
        confusion_matrix(
            labels_numpy, adjusted["predictions"], labels=[0, 1]
        ),
        index=matrix_index,
        columns=matrix_columns
    )

    # ---------- 결과 저장 (기존 결과와 다른 새 폴더) ----------
    results_dir.mkdir(parents=True, exist_ok=True)

    sweep_table.to_csv(
        results_dir / "validation_threshold_sweep.csv",
        index=False,
        encoding="utf-8-sig"
    )

    baseline_matrix.to_csv(
        results_dir / "test_confusion_matrix_default05.csv",
        encoding="utf-8-sig"
    )

    adjusted_matrix.to_csv(
        results_dir / "test_confusion_matrix_threshold.csv",
        encoding="utf-8-sig"
    )

    lines = []
    record = lines.append

    record("==========================================")
    record("이원화 결정 임계값(threshold) 보정 실험 결과")
    record("==========================================")
    record(f"모델 파일 : {best_model_path.name}")
    record(f"모델 종류 : {MODEL_ARCH} ({EVALUATION_TARGET})")
    record(f"실험 이름 : {RUN_NAME}")
    record("라벨 정의 : 0=우수, 1=불량(보통 병합)")
    record(f"사용 장치 : {device}")
    record("")
    record("[threshold 결정 절차]")
    record("threshold 탐색은 validation에서만 수행 (test 누출 없음)")
    record(
        f"탐색 범위 : 0.02 ~ 0.98 "
        f"(0.01 간격, {len(threshold_candidates)}개, 불량 확률 기준)"
    )
    record(f"제약 조건 : 불량 recall >= {RECALL_FLOOR}")
    record("선택 기준 : macro F1 최대 (동률이면 우수 F1 최대)")
    record(f"선택된 threshold : {chosen_threshold} (불량 확률이 이 값 이상이면 불량)")
    record(f"validation macro F1 : {best_row['macro_F1']:.4f}")
    record(f"validation 우수 F1 : {best_row['우수_F1']:.4f}")
    record(f"validation 불량 recall : {best_row['불량_recall']:.4f}")
    record("")
    record("===== Test 결과 비교 (같은 체크포인트, 결정 규칙만 다름) =====")
    record("")
    record(f"{'지표':<14}{'기본 0.5(=argmax)':>18}{'threshold 보정':>16}")
    record(
        f"{'우수 F1':<14}"
        f"{baseline['good_f1']:>18.4f}"
        f"{adjusted['good_f1']:>16.4f}"
    )
    record(
        f"{'우수 recall':<14}"
        f"{baseline['good_recall']:>18.4f}"
        f"{adjusted['good_recall']:>16.4f}"
    )
    record(
        f"{'불량 recall':<14}"
        f"{baseline['defect_recall']:>18.4f}"
        f"{adjusted['defect_recall']:>16.4f}"
    )
    record(
        f"{'불량 F1':<14}"
        f"{baseline['defect_f1']:>18.4f}"
        f"{adjusted['defect_f1']:>16.4f}"
    )
    record(
        f"{'Macro F1':<14}"
        f"{baseline['macro_f1']:>18.4f}"
        f"{adjusted['macro_f1']:>16.4f}"
    )
    record(
        f"{'Accuracy':<14}"
        f"{baseline['accuracy']:>18.4f}"
        f"{adjusted['accuracy']:>16.4f}"
    )
    record("")
    record("===== 기본 0.5(=argmax) 클래스별 상세 =====")
    record(baseline_report)
    record("")
    record("===== threshold 보정 클래스별 상세 =====")
    record(adjusted_report)
    record("")
    record("===== 기본 0.5(=argmax) 혼동행렬 =====")
    record(baseline_matrix.to_string())
    record("")
    record("===== threshold 보정 혼동행렬 =====")
    record(adjusted_matrix.to_string())

    report_text = "\n".join(lines)
    print("\n" + report_text)

    # utf-8-sig: 저장소 관례 통일 (윈도우 메모장/엑셀에서도 한글이 깨지지 않게)
    report_path = results_dir / "threshold_tuning_result.txt"
    report_path.write_text(report_text, encoding="utf-8-sig")

    print("\n결과 저장 위치 :", results_dir)


if __name__ == "__main__":
    main()
