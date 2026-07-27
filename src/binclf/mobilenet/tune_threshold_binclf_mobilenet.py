# ============================================================
# [사용법]
# 이원화(2클래스: 우수 vs 불량) MobileNetV2 결정 임계값(threshold) 보정 스크립트
# — 재학습 없이 기존 체크포인트만 사용.
#
# 기존 src/train_model.py와 학습 조건(동결 범위/lr/batch 50)은 동일하나
# 코드 구조는 binclf 표준(resnet 스타일)을 따름.
#
# 이미 학습된 MobileNetV2 binclf 체크포인트를 읽어 validation 셋에서
# "불량 확률 임계값"을 탐색하고, 확정된 threshold 하나로 test를 1회 평가합니다.
# (3클래스의 "보통 logit bias sweep"을 이원화에 맞게 확률 임계값 sweep으로 재설계.
#  softmax(logits)의 불량 확률이 threshold 이상이면 불량으로 판정)
# threshold 탐색은 validation에서만 하므로 test 정보 누출(leakage)이 없습니다.
#
# 기존 코드/체크포인트/평가 결과는 일절 수정하지 않습니다.
# 결과는 test_results/binclf/mobilenet_v2/<RUN_NAME>_threshold/ 아래에만 새로 저장됩니다.
#
# 실행 (런팟 = 리눅스):
#   RUN_NAME=bin1 python src/binclf/mobilenet/tune_threshold_binclf_mobilenet.py
#
# 실행 (윈도우 PowerShell):
#   $env:RUN_NAME="bin1"; $env:EVAL_TARGET="finetuned"
#   python src/binclf/mobilenet/tune_threshold_binclf_mobilenet.py
#
# [환경변수]
#   RUN_NAME     : 학습 때 쓴 실험 이름 (기본 "default")
#   EVAL_TARGET  : baseline 또는 finetuned (기본 "finetuned")
#   RECALL_FLOOR : 불량 recall 하한 (기본 0.85).
#                  "불량 recall이 이 값 이상"인 threshold 중에서 macro F1 최대를 선택
#                  (동률이면 우수 F1이 큰 쪽)
#   BATCH_SIZE   : 배치 크기 (기본 50)
#   DATA_DIR     : data/ 와 model/ 이 있는 프로젝트 루트 (기본: 이 저장소).
#                  워크트리처럼 데이터가 없는 곳에서 돌릴 때 원본 저장소를 지정
#
# GPU가 없어도 됩니다. 학습이 아니라 추론 2회(validation 1회 + test 1회)뿐이라
# CPU로도 동작하며, threshold 탐색은 저장해 둔 확률에 대한 배열 연산이라 수 초면 끝납니다.
# ============================================================

import os

import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)
from torch import nn
from torchvision.models import mobilenet_v2

# 전처리/split/경로는 학습·평가와 완전히 동일해야 하므로 binclf 공용 정의를 그대로 사용
from preprocess_binclf_mobilenet import (
    build_dataloaders,
    model_dir,
    test_results_root
)

# 모델 아키텍처: MobileNetV2 고정 (resnet 버전과 달리 환경변수 분기 없음)
MODEL_ARCH = "mobilenet_v2"

RUN_NAME = os.environ.get("RUN_NAME", "default")
EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")

# 불량 recall 하한: 이 값 밑으로 떨어뜨리는 threshold는 후보에서 제외
RECALL_FLOOR = float(os.environ.get("RECALL_FLOOR", 0.85))

best_model_path = (
    model_dir / f"best_{MODEL_ARCH}_binclf_{EVALUATION_TARGET}_{RUN_NAME}.pth"
)

# 결과는 기존 평가 결과와 다른 폴더에 저장 (덮어쓰기 없음)
results_dir = test_results_root / MODEL_ARCH / f"{RUN_NAME}_threshold"

class_names = ["우수", "불량"]

# 탐색할 threshold 후보: 0.02 ~ 0.98을 0.01 간격으로
# (불량 확률 P(불량)이 threshold 이상이면 불량 판정.
#  0.50 = 기존 argmax와 동일한 결정 규칙이 후보에 포함됨)
threshold_candidates = [round(0.02 + step * 0.01, 2) for step in range(97)]

# 기준선: threshold 0.5 (두 클래스 softmax에서 argmax와 동일한 결정 규칙)
DEFAULT_THRESHOLD = 0.5


def create_mobilenet_model():
    """저장된 체크포인트를 덮어씌울 빈 MobileNetV2 구조를 만든다.

    evaluate_binclf_mobilenet.py의 create_mobilenet_model과 동일 —
    구조가 다르면 가중치 로드가 실패한다.
    """
    model = mobilenet_v2(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)

    return model


def collect_defect_probabilities(model, loader, device):
    """한 split 전체를 추론해 (불량 확률, 정답) 텐서를 반환한다. 모델 수정 없음.

    불량 확률 = softmax(logits)[:, 1]. 이 값 하나만 있으면 어떤 threshold든
    재추론 없이 배열 연산으로 예측을 다시 만들 수 있다.
    """
    all_defect_probabilities = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            defect_probabilities = torch.softmax(outputs, dim=1)[:, 1]

            all_defect_probabilities.append(defect_probabilities.cpu())
            all_labels.append(labels)

    return torch.cat(all_defect_probabilities), torch.cat(all_labels)


def metrics_with_threshold(defect_probabilities, labels, threshold):
    """불량 확률이 threshold 이상이면 불량(1)으로 판정했을 때의 주요 지표를 계산한다."""
    predictions = (
        defect_probabilities >= threshold
    ).long().numpy()

    labels_numpy = labels.numpy()

    per_class_f1 = f1_score(
        labels_numpy, predictions, labels=[0, 1], average=None,
        zero_division=0
    )

    return {
        "threshold": threshold,
        "predictions": predictions,
        "accuracy": float((predictions == labels_numpy).mean()),
        "good_f1": float(per_class_f1[0]),
        "defect_f1": float(per_class_f1[1]),
        "defect_recall": float(recall_score(
            labels_numpy, predictions, labels=[1], average=None,
            zero_division=0
        )[0]),
        "macro_f1": float(per_class_f1.mean()),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("사용 장치 :", device)
    print("평가 모델 :", best_model_path.name)
    print("라벨 정의 : 0=우수, 1=불량(보통 병합)")
    print("불량 recall 하한 :", RECALL_FLOOR)

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {best_model_path}\n"
            "RUN_NAME / EVAL_TARGET / DATA_DIR 값을 확인하세요."
        )

    # validation/test 로더는 binclf 공용 정의를 그대로 사용
    # (metadata_split_binary3.csv 검증, LIMIT_PER_SPLIT, DATA_DIR 오버라이드 모두 포함)
    _, validation_loader, test_loader = build_dataloaders()

    model = create_mobilenet_model()

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
    validation_probabilities, validation_labels = collect_defect_probabilities(
        model, validation_loader, device
    )

    print("Validation 장수 :", len(validation_labels))

    sweep_rows = []
    for threshold in threshold_candidates:
        result = metrics_with_threshold(
            validation_probabilities, validation_labels, threshold
        )
        sweep_rows.append({
            "threshold": result["threshold"],
            "우수_F1": result["good_f1"],
            "불량_recall": result["defect_recall"],
            "불량_F1": result["defect_f1"],
            "macro_F1": result["macro_f1"],
            "accuracy": result["accuracy"],
        })

    # 선택은 반올림 전의 원래 값으로 수행 (CSV 저장 시에만 반올림)
    sweep_table = pd.DataFrame(sweep_rows)

    # 불량 recall 하한을 지키는 후보 중 macro F1이 최대인 threshold 선택
    # (동률이면 우수 F1이 큰 쪽 — 이원화의 병목은 소수 클래스인 우수 검출이므로)
    eligible = sweep_table[sweep_table["불량_recall"] >= RECALL_FLOOR]

    if eligible.empty:
        raise RuntimeError(
            f"불량 recall {RECALL_FLOOR} 이상을 만족하는 threshold가 없습니다. "
            "RECALL_FLOOR를 낮춰서 다시 실행하세요."
        )

    best_row = eligible.sort_values(
        ["macro_F1", "우수_F1"],
        ascending=False
    ).iloc[0]

    chosen_threshold = float(best_row["threshold"])

    print("\n[validation 탐색 결과]")
    print(f"선택된 threshold : {chosen_threshold}")
    print(f"validation macro F1 : {best_row['macro_F1']:.4f}")
    print(f"validation 우수 F1 : {best_row['우수_F1']:.4f}")
    print(f"validation 불량 recall : {best_row['불량_recall']:.4f}")

    # ---------- 2단계: 확정된 threshold로 test 1회 평가 ----------
    print("\nTest 추론 중 (1회만 실행됨)...")
    test_probabilities, test_labels = collect_defect_probabilities(
        model, test_loader, device
    )

    print("Test 장수 :", len(test_labels))

    baseline = metrics_with_threshold(
        test_probabilities, test_labels, DEFAULT_THRESHOLD
    )
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

    # utf-8-sig: 윈도우 메모장/엑셀에서도 한글이 깨지지 않게
    sweep_table.round(4).to_csv(
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
    record("결정 임계값(threshold) 보정 실험 결과 — 이원화(binclf)")
    record("==========================================")
    record(f"모델 파일 : {best_model_path.name}")
    record(f"모델 종류 : {MODEL_ARCH} ({EVALUATION_TARGET})")
    record(f"실험 이름 : {RUN_NAME}")
    record("라벨 정의 : 0=우수, 1=불량(보통 병합)")
    record(f"사용 장치 : {device}")
    record("")
    record("[threshold 결정 절차]")
    record("판정 규칙 : softmax(logits)의 불량 확률 >= threshold 이면 불량")
    record("threshold 탐색은 validation에서만 수행 (test 누출 없음)")
    record(
        f"탐색 범위 : 0.02 ~ 0.98 (0.01 간격, {len(threshold_candidates)}개)"
    )
    record(f"제약 조건 : 불량 recall >= {RECALL_FLOOR}")
    record("선택 기준 : 제약을 만족하는 후보 중 macro F1 최대 (동률 시 우수 F1 최대)")
    record(f"선택된 threshold : {chosen_threshold}")
    record(f"validation macro F1 : {best_row['macro_F1']:.4f}")
    record(f"validation 우수 F1 : {best_row['우수_F1']:.4f}")
    record(f"validation 불량 recall : {best_row['불량_recall']:.4f}")
    record("")
    record("===== Test 결과 비교 (같은 체크포인트, 결정 규칙만 다름) =====")
    record(f"기준선 threshold = {DEFAULT_THRESHOLD} (2클래스 argmax와 동일한 규칙)")
    record("")
    record(f"{'지표':<14}{'기준 0.5':>12}{'threshold 보정':>16}")
    record(
        f"{'우수 F1':<14}"
        f"{baseline['good_f1']:>12.4f}"
        f"{adjusted['good_f1']:>16.4f}"
    )
    record(
        f"{'불량 recall':<14}"
        f"{baseline['defect_recall']:>12.4f}"
        f"{adjusted['defect_recall']:>16.4f}"
    )
    record(
        f"{'불량 F1':<14}"
        f"{baseline['defect_f1']:>12.4f}"
        f"{adjusted['defect_f1']:>16.4f}"
    )
    record(
        f"{'Macro F1':<14}"
        f"{baseline['macro_f1']:>12.4f}"
        f"{adjusted['macro_f1']:>16.4f}"
    )
    record(
        f"{'Accuracy':<14}"
        f"{baseline['accuracy']:>12.4f}"
        f"{adjusted['accuracy']:>16.4f}"
    )
    record("")
    record("===== 기준(threshold 0.5) 클래스별 상세 =====")
    record(baseline_report)
    record("")
    record("===== threshold 보정 클래스별 상세 =====")
    record(adjusted_report)
    record("")
    record("===== 기준(threshold 0.5) 혼동행렬 =====")
    record(baseline_matrix.to_string())
    record("")
    record("===== threshold 보정 혼동행렬 =====")
    record(adjusted_matrix.to_string())

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path = results_dir / "threshold_tuning_result.txt"
    report_path.write_text(report_text, encoding="utf-8-sig")

    print("\n결과 저장 위치 :", results_dir)


if __name__ == "__main__":
    main()
