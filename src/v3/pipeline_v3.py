# -*- coding: utf-8 -*-
# ============================================================
# v3 학습/평가 엔진 — 이원화(binclf) 단일 과제, middle_id 완전 제거.
# v2 엔진(pipeline_v2)에서 midclf(7클래스) 분기를 걷어낸 버전이며,
# 과제별 스크립트(train_binclf_v3.py 등)가 이 엔진을 task 인자로 호출합니다.
# 이미지 캐시/annotation은 v2 산출물을 공유하고 CSV만 metadata_split_v3.csv를 씁니다.
#
# 기존 binclf 대비 v2 신규:
#   - 시드 고정 (preprocess_v3.set_seed)
#   - epoch별 학습 히스토리 CSV 저장 (train_history_*.csv)
#   - test 샘플별 예측 CSV 저장 (predictions_*.csv)
#   - cosine LR 스케줄러(USE_SCHEDULER) / AMP(USE_AMP, CUDA에서만) — fine_tune 단계
#   - 체크포인트에 use_clahe/eval_resize_mode 저장 + evaluate에서 불일치 경고
# ============================================================

import contextlib
import csv
import os
import statistics
import time

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from model_factory_v3 import (
    build_finetune_param_groups,
    count_parameters,
    create_model,
    create_optimizer,
    freeze_backbone,
    head_parameters,
    set_frozen_modules_eval,
    unfreeze_for_finetune,
)
from preprocess_v3 import (
    INPUT_SIZE,
    SEED,
    TASK_CLASS_NAMES,
    TASK_NUM_CLASSES,
    TASK_TOKENS,
    batch_size,
    build_dataloaders,
    compute_class_weights,
    eval_resize_mode,
    limit_per_split,
    load_split_dataframes,
    model_dir,
    results_dir,
    set_seed,
    use_clahe,
)

# ---- 공통 환경변수 ----
ARCH = os.environ.get("ARCH", "resnet50")
RUN_NAME = os.environ.get("RUN_NAME", "default")
LR_HEAD = float(os.environ.get("LR_HEAD", 1e-3))
LR_BACKBONE = float(os.environ.get("LR_BACKBONE", 1e-5))
PATIENCE = int(os.environ.get("PATIENCE", 3))
USE_SCHEDULER = os.environ.get("USE_SCHEDULER", "1") == "1"
USE_AMP = os.environ.get("USE_AMP", "1") == "1"

# 파인튜닝 2단계에서 head에 적용할 학습률 (관례: 백본의 10배)
LR_HEAD_FINETUNE = float(os.environ.get("LR_HEAD_FINETUNE", 1e-4))

WARMUP_ITERATIONS = 20
LATENCY_ITERATIONS = 100
THROUGHPUT_ITERATIONS = 30

TASK_LABEL_DEFINITIONS = {
    "binclf": "0=우수, 1=불량(보통 병합)",
}


def checkpoint_path(task, stage, arch=ARCH, run_name=RUN_NAME):
    """예: model/best_resnet50_binclf_v2_baseline_default.pth"""
    return model_dir / f"best_{arch}_{TASK_TOKENS[task]}_{stage}_{run_name}.pth"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_amp_tools(device):
    """(GradScaler 또는 None, autocast 컨텍스트 팩토리)를 반환한다. CPU면 AMP 비활성."""
    if not (USE_AMP and device.type == "cuda"):
        return None, contextlib.nullcontext
    try:
        scaler = torch.amp.GradScaler("cuda")
        return scaler, lambda: torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
        return scaler, torch.cuda.amp.autocast


def synchronize(device):
    """CUDA 커널이 끝날 때까지 대기 (비동기 실행 때문에 시간 측정 전 필수)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def evaluate_on_loader(model, data_loader, loss_function, device, num_classes):
    """loss/accuracy/클래스별 F1/macro F1을 계산한다. 전부 파이썬 기본 타입으로 반환."""
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

    per_class_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    return (
        loss_sum / total_count,
        correct_count / total_count,
        [float(value) for value in per_class_f1],
        float(per_class_f1.mean()),
    )


class HistoryWriter:
    """epoch별 지표를 CSV로 즉시 기록한다 (중단 대비 append+flush).

    focus F1: 우수(소수 클래스) F1 —
    macro F1이 가려버리는 최약 지점을 따로 추적하기 위한 열.
    """

    def __init__(self, task, stage):
        directory = results_dir(task, ARCH, RUN_NAME)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"train_history_{stage}.csv"

        focus_column = "val_f1_우수" if task == "binclf" else "val_f1_최저클래스"
        self.fields = [
            "stage", "epoch", "train_loss", "val_loss", "val_macro_f1",
            "val_accuracy", focus_column, "lr_head", "lr_backbone",
            "epoch_time_sec", "is_best",
        ]
        self.stage = stage
        self.task = task

        # 같은 RUN_NAME 재실행 시 이전 히스토리는 새로 시작 (실험 섞임 방지)
        self.file = self.path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.fields)
        self.file.flush()

    def write(self, epoch, train_loss, val_loss, val_macro_f1, val_accuracy,
              per_class_f1, lr_head, lr_backbone, epoch_time_sec, is_best):
        focus_f1 = per_class_f1[0] if self.task == "binclf" else min(per_class_f1)
        self.writer.writerow([
            self.stage, epoch,
            f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_macro_f1:.6f}",
            f"{val_accuracy:.6f}", f"{focus_f1:.6f}",
            f"{lr_head:.2e}", f"{lr_backbone:.2e}" if lr_backbone else "",
            f"{epoch_time_sec:.1f}", int(is_best),
        ])
        self.file.flush()

    def close(self):
        self.file.close()


def build_checkpoint_dict(task, stage, model, epoch, val_metrics, class_weight_values,
                          trainable_count, total_count, unfrozen, extra=None):
    """체크포인트 dict. 전부 파이썬 기본 타입 — torch.load(weights_only=True) 호환."""
    val_loss, val_accuracy, per_class_f1, val_macro_f1 = val_metrics
    payload = {
        "task": TASK_TOKENS[task],
        "label_definition": TASK_LABEL_DEFINITIONS[task],
        "metadata_file": "metadata_split_v2.csv",
        "selection_metric": "macro_f1",
        "arch": ARCH,
        "run_name": RUN_NAME,
        "seed": SEED,
        "input_size": INPUT_SIZE,
        "use_clahe": use_clahe,
        "eval_resize_mode": eval_resize_mode,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "validation_macro_f1": float(val_macro_f1),
        "validation_per_class_f1": per_class_f1,
        "class_names": list(TASK_CLASS_NAMES[task]),
        "class_weight_values": class_weight_values,
        "batch_size": batch_size,
        "trainable_param_count": trainable_count,
        "total_param_count": total_count,
        "unfrozen_layers": unfrozen,
        "use_scheduler": USE_SCHEDULER,
        "use_amp": USE_AMP,
    }
    if task == "binclf":
        payload["validation_good_f1"] = per_class_f1[0]
        payload["validation_defect_f1"] = per_class_f1[1]
    if extra:
        payload.update(extra)
    return payload


def _print_run_header(task, stage_description):
    print(f"과제 : {TASK_TOKENS[task]} ({TASK_LABEL_DEFINITIONS[task]})")
    print(f"단계 : {stage_description}")
    print("모델 (ARCH) :", ARCH)
    print("실험 이름 (RUN_NAME) :", RUN_NAME)
    print("배치 크기 :", batch_size)
    print("입력 크기 :", f"{INPUT_SIZE}x{INPUT_SIZE}")
    print("시드 :", SEED)
    print("CLAHE :", "켜짐" if use_clahe else "꺼짐")
    print("평가 리사이즈 모드 :", eval_resize_mode)
    if limit_per_split > 0:
        print(f"[스모크 모드] LIMIT_PER_SPLIT={limit_per_split} — 실험 결과에 사용 금지")


def _train_one_epoch(model, train_loader, loss_function, optimizer, device,
                     scaler, autocast_factory):
    """1 epoch 학습. (train_loss, train_accuracy)를 반환한다."""
    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast_factory():
                outputs = model(images)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        total_count += labels.size(0)

    return loss_sum / total_count, correct_count / total_count


# ============================================================
# 1단계: baseline 학습 (head만)
# ============================================================

def run_train(task):
    num_epochs = int(os.environ.get("NUM_EPOCHS", 5))
    num_classes = TASK_NUM_CLASSES[task]

    _print_run_header(task, f"1단계 baseline (head만, 최대 {num_epochs} epoch)")

    device = get_device()
    print("사용 장치 :", device)
    if device.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))

    generator = set_seed()
    train_loader, validation_loader, _ = build_dataloaders(task, generator)

    train_data, _, _ = load_split_dataframes(task)
    class_weight_values = compute_class_weights(train_data, task)
    print("클래스 가중치 (train 분포) :",
          [round(value, 4) for value in class_weight_values])

    model = create_model(ARCH, num_classes, use_pretrained_weights=True)
    freeze_backbone(model, ARCH)
    model = model.to(device)

    total_count, trainable_count = count_parameters(model)
    print(f"파라미터 : 전체 {total_count / 1e6:.1f}M / "
          f"학습 {trainable_count / 1e6:.2f}M ({trainable_count / total_count:.0%})")

    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weight_values, dtype=torch.float32, device=device)
    )
    optimizer = create_optimizer(ARCH, head_parameters(model, ARCH), LR_HEAD)
    scaler, autocast_factory = make_amp_tools(device)
    print("AMP :", "켜짐" if scaler is not None else "꺼짐")

    best_path = checkpoint_path(task, "baseline")
    model_dir.mkdir(parents=True, exist_ok=True)
    history = HistoryWriter(task, "baseline")

    best_macro_f1 = -1.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        print(f"\n==== Epoch {epoch}/{num_epochs} ====")

        model.train()
        # baseline은 head만 학습하므로 백본(BatchNorm 포함)은 평가 상태 유지
        set_frozen_modules_eval(model, ARCH)
        if ARCH == "resnet50":
            model.layer4.eval()  # baseline에서는 layer4도 동결 상태

        train_loss, train_accuracy = _train_one_epoch(
            model, train_loader, loss_function, optimizer, device,
            scaler, autocast_factory,
        )

        val_metrics = evaluate_on_loader(
            model, validation_loader, loss_function, device, num_classes
        )
        val_loss, val_accuracy, per_class_f1, val_macro_f1 = val_metrics

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2%}")
        print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.2%} | "
              f"Macro F1: {val_macro_f1:.4f}")
        print("클래스별 F1 :", [round(value, 4) for value in per_class_f1])

        is_best = val_macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = val_macro_f1
            torch.save(
                build_checkpoint_dict(
                    task, "baseline", model, epoch, val_metrics,
                    class_weight_values, trainable_count, total_count,
                    unfrozen="head",
                    extra={"num_epochs": num_epochs},
                ),
                best_path,
            )
            print("최고 모델 저장 :", best_path)

        history.write(epoch, train_loss, val_loss, val_macro_f1, val_accuracy,
                      per_class_f1, LR_HEAD, None, time.time() - epoch_start, is_best)

    history.close()

    print("\n==========================================")
    print("baseline 학습 완료")
    print(f"최고 Validation Macro F1: {best_macro_f1:.4f}")
    print(f"학습 시간 : {time.time() - start_time:.1f}초")
    print("저장된 모델 :", best_path)
    print("히스토리 :", history.path)


# ============================================================
# 2단계: 파인튜닝 (부분 unfreeze + 차등 lr + cosine + AMP)
# ============================================================

def run_fine_tune(task):
    num_epochs = int(os.environ.get("NUM_EPOCHS", 15))
    num_classes = TASK_NUM_CLASSES[task]

    _print_run_header(
        task,
        f"2단계 파인튜닝 (부분 unfreeze, 최대 {num_epochs} epoch, patience {PATIENCE})",
    )

    baseline_path = checkpoint_path(task, "baseline")
    finetuned_path = checkpoint_path(task, "finetuned")

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"baseline 모델을 찾을 수 없습니다: {baseline_path}\n"
            f"먼저 같은 ARCH/RUN_NAME으로 train_{task}_v3.py를 실행하세요."
        )

    device = get_device()
    print("사용 장치 :", device)
    if device.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))

    generator = set_seed()
    train_loader, validation_loader, _ = build_dataloaders(task, generator)

    train_data, _, _ = load_split_dataframes(task)
    class_weight_values = compute_class_weights(train_data, task)

    model = create_model(ARCH, num_classes, use_pretrained_weights=False)
    baseline_checkpoint = torch.load(
        baseline_path, map_location=device, weights_only=True
    )
    model.load_state_dict(baseline_checkpoint["model_state_dict"])

    baseline_macro_f1 = baseline_checkpoint.get("validation_macro_f1", -1.0)
    print(f"baseline Validation Macro F1: {baseline_macro_f1:.4f}")

    unfrozen = unfreeze_for_finetune(model, ARCH)
    model = model.to(device)

    total_count, trainable_count = count_parameters(model)
    print(f"동결 해제 : {unfrozen}")
    print(f"파라미터 : 전체 {total_count / 1e6:.1f}M / "
          f"학습 {trainable_count / 1e6:.2f}M ({trainable_count / total_count:.0%})")

    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weight_values, dtype=torch.float32, device=device)
    )
    optimizer = create_optimizer(
        ARCH,
        build_finetune_param_groups(model, ARCH, LR_BACKBONE, LR_HEAD_FINETUNE),
        LR_BACKBONE,
    )

    scheduler = None
    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    print("LR 스케줄러 :", "cosine" if scheduler else "없음")

    scaler, autocast_factory = make_amp_tools(device)
    print("AMP :", "켜짐" if scaler is not None else "꺼짐")

    history = HistoryWriter(task, "finetuned")

    def save_finetuned(epoch, val_metrics):
        torch.save(
            build_checkpoint_dict(
                task, "finetuned", model, baseline_checkpoint.get("epoch", 0),
                val_metrics, class_weight_values, trainable_count, total_count,
                unfrozen=unfrozen,
                extra={
                    "num_epochs": num_epochs,
                    "fine_tune_epoch": epoch,
                    "source_model": baseline_path.name,
                },
            ),
            finetuned_path,
        )

    # 개선이 전혀 없어도 baseline 상태가 finetuned 파일로 존재하도록 선저장 (관례)
    save_finetuned(
        0,
        (
            baseline_checkpoint.get("validation_loss", -1.0),
            baseline_checkpoint.get("validation_accuracy", -1.0),
            baseline_checkpoint.get(
                "validation_per_class_f1", [-1.0] * num_classes
            ),
            baseline_macro_f1,
        ),
    )

    best_macro_f1 = baseline_macro_f1
    no_improvement_count = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        print(f"\n==== Fine-tuning Epoch {epoch}/{num_epochs} ====")

        model.train()
        set_frozen_modules_eval(model, ARCH)

        train_loss, train_accuracy = _train_one_epoch(
            model, train_loader, loss_function, optimizer, device,
            scaler, autocast_factory,
        )

        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[-1]["lr"]

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate_on_loader(
            model, validation_loader, loss_function, device, num_classes
        )
        val_loss, val_accuracy, per_class_f1, val_macro_f1 = val_metrics

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2%}")
        print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.2%} | "
              f"Macro F1: {val_macro_f1:.4f}")
        print("클래스별 F1 :", [round(value, 4) for value in per_class_f1])

        is_best = val_macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = val_macro_f1
            no_improvement_count = 0
            save_finetuned(epoch, val_metrics)
            print("파인튜닝 최고 모델 저장 :", finetuned_path)
        else:
            no_improvement_count += 1
            print(f"Validation 성능 개선 없음: {no_improvement_count}/{PATIENCE}")

        history.write(epoch, train_loss, val_loss, val_macro_f1, val_accuracy,
                      per_class_f1, current_lr_head, current_lr_backbone,
                      time.time() - epoch_start, is_best)

        if no_improvement_count >= PATIENCE:
            print("Validation 성능이 개선되지 않아 중단합니다.")
            break

    history.close()

    print("\n==========================================")
    print("파인튜닝 완료")
    print(f"최고 Validation Macro F1: {best_macro_f1:.4f}")
    print(f"학습 시간 : {time.time() - start_time:.1f}초")
    print("저장된 모델 :", finetuned_path)
    print("히스토리 :", history.path)


# ============================================================
# 3단계: test 평가 (+ 추론속도 3층 + 예측 CSV)
# ============================================================

def _measure_single_image_latency(model, device):
    """단일 이미지 추론 시간의 (중앙값 ms, p95 ms). 실서비스 체감 기준."""
    single_image = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            model(single_image)
        synchronize(device)

        elapsed_milliseconds = []
        for _ in range(LATENCY_ITERATIONS):
            start_time = time.perf_counter()
            model(single_image)
            synchronize(device)
            elapsed_milliseconds.append((time.perf_counter() - start_time) * 1000)

    elapsed_milliseconds.sort()
    p95_index = int(round(0.95 * (len(elapsed_milliseconds) - 1)))
    return statistics.median(elapsed_milliseconds), elapsed_milliseconds[p95_index]


def _measure_batch_throughput(model, device):
    """배치 forward만 반복해 초당 처리 장수를 측정 (데이터 로딩 제외)."""
    image_batch = torch.randn(batch_size, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            model(image_batch)
        synchronize(device)

        start_time = time.perf_counter()
        for _ in range(THROUGHPUT_ITERATIONS):
            model(image_batch)
        synchronize(device)
        elapsed_seconds = time.perf_counter() - start_time

    return batch_size * THROUGHPUT_ITERATIONS / elapsed_seconds


def run_evaluate(task):
    evaluation_target = os.environ.get("EVAL_TARGET", "finetuned")
    if evaluation_target not in ("baseline", "finetuned"):
        raise ValueError(f'EVAL_TARGET은 "baseline"/"finetuned"여야 합니다: {evaluation_target}')

    num_classes = TASK_NUM_CLASSES[task]
    class_names = TASK_CLASS_NAMES[task]
    class_labels = list(range(num_classes))

    model_path = checkpoint_path(task, evaluation_target)
    output_dir = results_dir(task, ARCH, RUN_NAME)
    result_text_path = output_dir / f"evaluate_{evaluation_target}_result.txt"
    confusion_matrix_path = output_dir / f"test_confusion_matrix_{evaluation_target}.csv"
    predictions_path = output_dir / f"predictions_{evaluation_target}.csv"

    result_lines = []

    def record(text=""):
        print(text)
        result_lines.append(str(text))

    if not model_path.exists():
        raise FileNotFoundError(
            f"평가할 모델을 찾을 수 없습니다: {model_path}\n"
            f"먼저 같은 ARCH/RUN_NAME으로 학습을 실행하세요."
        )

    device = get_device()

    record("===== 평가 대상 =====")
    record(f"과제 : {TASK_TOKENS[task]}")
    record(f"모델 종류 (ARCH) : {ARCH}")
    record(f"실험 이름 (RUN_NAME) : {RUN_NAME}")
    record(f"평가 단계 (EVAL_TARGET) : {evaluation_target}")
    record(f"모델 파일 : {model_path.name}")
    record(f"라벨 정의 : {TASK_LABEL_DEFINITIONS[task]}")
    record(f"사용 장치 : {device}")
    if device.type == "cuda":
        record(f"GPU : {torch.cuda.get_device_name(0)}")
    record(f"torch 버전 : {torch.__version__}")

    set_seed()
    _, _, test_loader = build_dataloaders(task)
    _, _, test_data = load_split_dataframes(task)

    model = create_model(ARCH, num_classes, use_pretrained_weights=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    record("")
    record("===== 학습 설정 (모델 파일에 기록된 값) =====")
    record(f"baseline epoch : {checkpoint.get('epoch', '?')}")
    if "fine_tune_epoch" in checkpoint:
        record(f"파인튜닝 epoch : {checkpoint['fine_tune_epoch']}")

    saved_macro_f1 = checkpoint.get("validation_macro_f1")
    if isinstance(saved_macro_f1, float):
        record(f"저장 당시 Validation Macro F1 : {saved_macro_f1:.4f}")

    record(f"클래스 가중치 : {checkpoint.get('class_weight_values', '?')}")
    record(f"동결 해제 범위 : {checkpoint.get('unfrozen_layers', '?')}")
    record(f"입력 크기 : {checkpoint.get('input_size', '?')}")
    record(f"시드 : {checkpoint.get('seed', '?')}")
    record(f"배치 크기 : {batch_size}")

    # 학습 시점 전처리 설정과 현재 env가 다르면 결과 해석이 왜곡되므로 경고
    # (의도적 ablation일 수 있어 중단은 하지 않음)
    saved_use_clahe = checkpoint.get("use_clahe")
    if saved_use_clahe is not None and saved_use_clahe != use_clahe:
        record(f"[경고] 학습 시 USE_CLAHE={int(saved_use_clahe)} vs "
               f"현재 {int(use_clahe)} — 전처리 불일치 상태로 평가 중")

    saved_resize_mode = checkpoint.get("eval_resize_mode")
    if saved_resize_mode is not None and saved_resize_mode != eval_resize_mode:
        record(f"[경고] 학습 시 EVAL_RESIZE_MODE={saved_resize_mode} vs "
               f"현재 {eval_resize_mode} — 전처리 불일치 상태로 평가 중")

    # ---- Test 추론 (end-to-end, 확률까지 수집) ----
    all_labels = []
    all_predictions = []
    all_probabilities = []

    end_to_end_start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    synchronize(device)
    end_to_end_seconds = time.perf_counter() - end_to_end_start_time

    # ---- 지표 ----
    total_count = len(all_labels)
    correct_count = sum(
        1 for label, prediction in zip(all_labels, all_predictions)
        if label == prediction
    )
    test_accuracy = correct_count / total_count

    per_class_f1 = f1_score(
        all_labels, all_predictions,
        labels=class_labels, average=None, zero_division=0,
    )
    test_macro_f1 = float(per_class_f1.mean())

    record("")
    record("===== Test 평가 결과 =====")
    record(f"Test 이미지 수 : {total_count}")
    record(f"맞힌 이미지 수 : {correct_count}")
    record(f"Test 정확도 : {test_accuracy:.2%}")

    if task == "binclf":
        record(f"불량 F1 (핵심 지표) : {per_class_f1[1]:.4f}")
        record(f"우수 F1 : {per_class_f1[0]:.4f}")
    else:
        worst_index = int(per_class_f1.argmin())
        record(f"최저 클래스 F1 : {per_class_f1[worst_index]:.4f} "
               f"({class_names[worst_index]})")

    record(f"Macro F1 : {test_macro_f1:.4f}")

    record("")
    record("===== 클래스별 상세 =====")
    record(classification_report(
        all_labels, all_predictions,
        labels=class_labels, target_names=class_names,
        digits=4, zero_division=0,
    ))

    # ---- 추론속도 (기존 관례의 3층 측정) ----
    latency_median_ms, latency_p95_ms = _measure_single_image_latency(model, device)
    throughput_images_per_second = _measure_batch_throughput(model, device)

    record("===== 추론속도 =====")
    record(f"측정 장치 : {device}" + (
        f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    ))
    record(f"입력 해상도 : {INPUT_SIZE}x{INPUT_SIZE}")
    record("")
    record("[1] 단일 이미지 지연 (batch=1, 실서비스 체감 기준 — 가장 중요)")
    record(f"  중앙값 : {latency_median_ms:.2f}ms")
    record(f"  p95    : {latency_p95_ms:.2f}ms")
    record(f"  (워밍업 {WARMUP_ITERATIONS}회 후 {LATENCY_ITERATIONS}회 측정)")
    record("")
    record(f"[2] 배치 처리량 (batch={batch_size}, 대량 일괄 추론 기준)")
    record(f"  초당 처리 장수 : {throughput_images_per_second:.1f}장/초")
    record(f"  이미지 1장당   : {1000 / throughput_images_per_second:.2f}ms")
    record("  (데이터 로딩 제외, 순수 forward만)")
    record("")
    record("[3] end-to-end (데이터 로딩/전처리 포함, 기존 실험과 비교용)")
    record(f"  Test {total_count}장 전체 : {end_to_end_seconds:.1f}초")
    record(f"  초당 처리 장수 : {total_count / end_to_end_seconds:.1f}장/초")
    record(f"  이미지 1장당   : {end_to_end_seconds / total_count * 1000:.2f}ms")

    # ---- 혼동행렬 ----
    matrix = confusion_matrix(all_labels, all_predictions, labels=class_labels)
    confusion_table = pd.DataFrame(
        matrix,
        index=[f"실제_{name}" for name in class_names],
        columns=[f"예측_{name}" for name in class_names],
    )

    record("")
    record("===== 혼동행렬 =====")
    record(confusion_table)

    if task == "binclf":
        good_as_defect = int(matrix[0][1])
        good_total = int(matrix[0].sum())
        record("")
        record(f"우수 {good_total}장 중 {good_as_defect}장을 불량으로 오분류 "
               f"({good_as_defect / good_total:.1%})")

    # ---- 최종 요약 ----
    record("")
    record("==========================================")
    record(f"[비교용 최종 지표] {ARCH} {TASK_TOKENS[task]} ({evaluation_target})")
    if task == "binclf":
        record(f"Test 불량 F1 (핵심)   : {per_class_f1[1]:.4f}")
        record(f"Test 우수 F1          : {per_class_f1[0]:.4f}")
    else:
        record(f"Test 최저 클래스 F1   : {float(per_class_f1.min()):.4f}")
    record(f"Test Macro F1         : {test_macro_f1:.4f}")
    record(f"Test Accuracy (참고)  : {test_accuracy:.2%}")
    record(f"단일 이미지 지연      : {latency_median_ms:.2f}ms (중앙값)")
    record(f"배치 처리량           : {throughput_images_per_second:.1f}장/초")
    record("==========================================")

    # ---- 저장 ----
    output_dir.mkdir(parents=True, exist_ok=True)

    confusion_table.to_csv(confusion_matrix_path, encoding="utf-8-sig")
    result_text_path.write_text("\n".join(result_lines), encoding="utf-8-sig")

    # 샘플별 예측 CSV (v2 신규) — 오답 분석과 이원화×7클래스 교차 분석의 원천
    # test_loader는 shuffle=False라서 test_data 행 순서와 예측 순서가 1:1로 일치
    if len(test_data) != total_count:
        raise RuntimeError(
            f"예측 수({total_count})와 test 행 수({len(test_data)})가 다릅니다"
        )

    predictions_frame = pd.DataFrame({
        "source_data_id": test_data["source_data_id"].tolist(),
        "image_relpath": test_data["image_relpath"].tolist(),
    })

    predictions_frame["true_label"] = all_labels
    predictions_frame["pred_label"] = all_predictions
    predictions_frame["prob_불량"] = [
        round(probability[1], 6) for probability in all_probabilities
    ]

    predictions_frame["correct"] = [
        int(label == prediction)
        for label, prediction in zip(all_labels, all_predictions)
    ]
    predictions_frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print("")
    print("혼동행렬 저장 :", confusion_matrix_path)
    print("평가 결과 저장 :", result_text_path)
    print("예측 CSV 저장 :", predictions_path)
