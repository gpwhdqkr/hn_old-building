# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# v3 학습 파이프라인(이원화 binclf_v3 단일 과제 — middle_id 완전 제거)의
# 공용 데이터 정의 파일입니다. 보통은 train/fine_tune/evaluate 스크립트가
# import해서 쓰고, 데이터 로딩 점검용으로 직접 실행할 수 있습니다:
#
#   (리눅스)     DATA_DIR=/workspace/hn_old-building python src/v3/preprocess_v3.py
#   (PowerShell) $env:DATA_DIR="D:/hn_old-building"; python src/v3/preprocess_v3.py
#
# [v2와 다른 점]
# - midclf(결함 종류 7클래스) 없음 — 이원화(우수 vs 불량)만
# - CSV에 middle_id/middle_name 컬럼 자체가 없음 (make_metadata_split_v3.py로 생성,
#   split 배정은 v2와 동일하게 보존 — 그룹 누수 방지 검증 재사용)
# - 이미지 캐시/annotation/spec은 v2 산출물을 그대로 공유 (재전처리 불필요)
#
# [전제 조건]
# - data/processed/metadata_split_v3.csv   (v2에서 middle 컬럼 제거본, 깃 포함)
# - data/processed/annotations_v2.jsonl    (bbox 항상 존재 — defect-aware crop용)
# - data/processed/image_spec_v2.csv       (실측 해상도/배율, 깃 포함)
# - data/processed_images_v2/              (짧은 변 768 캐시 14GB, 깃 미포함 — 반입 필요)
#
# [기존 binclf와 다른 점]
# - 이미지: processed_images(224 정방형) → processed_images_v2(종횡비 유지 768)
# - 경로: CSV의 image_relpath를 그대로 사용 (raw/images 마커 잘라내기 불필요)
# - 입력 448, train은 RandomResizedCrop + defect-aware crop(불량은 폴리곤 bbox 중심)
# - 시드 고정(SEED), CLAHE 옵션(USE_CLAHE), 평가 리사이즈 모드(EVAL_RESIZE_MODE)
#
# [환경변수]
#   DATA_DIR         : data/, model/, test_results/ 상위 (기본: 이 저장소 루트)
#   BATCH_SIZE       : 배치 크기 (기본 64). 448 입력 기준 — OOM 시 32
#   NUM_WORKERS      : 데이터 로딩 워커 수 (기본 자동). 윈도우 스모크는 2 권장, 디버깅은 0
#   LIMIT_PER_SPLIT  : split당 최대 장수 (기본 0=전체). 스모크 전용 — 실험 사용 금지
#   SEED             : 난수 시드 (기본 42)
#   USE_CLAHE        : 1이면 CLAHE(음영 완화, cv2 필요) — train/val/test 동일 적용
#   EVAL_RESIZE_MODE : val/test 리사이즈 방식. "centercrop"(기본) 또는 "letterbox"
#   DEFECT_CROP_P    : 결함 이미지에 defect-aware crop을 적용할 확률 (기본 1.0)
#
# [주의]
# - 이 파일은 src/v3/ 안에 있으므로 프로젝트 루트는 두 단계 위입니다
#   (기존 src/binclf/<model>/ 은 세 단계 — 복붙 시 최다 빈발 버그).
# - 시드 고정은 "재현 가능한 셔플/크롭/초기화" 수준이며 cudnn.benchmark를 유지하므로
#   완전한 비트 단위 결정성은 아닙니다.
# ============================================================

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomResizedCrop as _RandomResizedCropV1
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

# src/v3/xxx.py → 두 단계 위가 프로젝트 루트
project_dir = Path(__file__).resolve().parent.parent.parent

data_dir = Path(os.environ.get("DATA_DIR", project_dir))

metadata_path = data_dir / "data" / "processed" / "metadata_split_v3.csv"
annotations_path = data_dir / "data" / "processed" / "annotations_v2.jsonl"
image_spec_path = data_dir / "data" / "processed" / "image_spec_v2.csv"
processed_images_dir = data_dir / "data" / "processed_images_v2"

model_dir = data_dir / "model"
test_results_root = data_dir / "test_results"

# ---- 과제 정의 ----
# v3는 이원화 단일 과제: binclf (0=우수, 1=불량(보통 병합)) — 극불균형 8.8%/91.2%
# middle_id(결함 종류 7종)는 v3에서 완전히 제거됨 — CSV에도 컬럼 없음
BIN_CLASS_NAMES = ["우수", "불량"]

TASK_TOKENS = {"binclf": "binclf_v3"}
TASK_NUM_CLASSES = {"binclf": 2}
TASK_CLASS_NAMES = {"binclf": BIN_CLASS_NAMES}

# ---- 입력 규격 ----
INPUT_SIZE = 448          # 최종 모델 입력 (224 대비 미세 균열 보존)
EVAL_RESIZE_SHORT = 512   # 평가 centercrop 모드에서 Resize 짧은 변

# ImageNet 사전학습 표준 (기존 파이프라인과 동일)
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

# letterbox 패딩색 = ImageNet 평균 (정규화 후 0 근처가 되도록)
_LETTERBOX_FILL = tuple(int(round(value * 255)) for value in image_mean)

# ---- 환경변수 ----
batch_size = int(os.environ.get("BATCH_SIZE", 64))
limit_per_split = int(os.environ.get("LIMIT_PER_SPLIT", 0))
SEED = int(os.environ.get("SEED", 42))
use_clahe = os.environ.get("USE_CLAHE", "0") == "1"
eval_resize_mode = os.environ.get("EVAL_RESIZE_MODE", "centercrop")
defect_crop_probability = float(os.environ.get("DEFECT_CROP_P", 1.0))

_cpu_count = os.cpu_count() or 4
num_workers = int(os.environ.get("NUM_WORKERS", min(32, max(4, _cpu_count // 2))))

if eval_resize_mode not in ("centercrop", "letterbox"):
    raise ValueError(
        f'EVAL_RESIZE_MODE는 "centercrop" 또는 "letterbox"여야 합니다: {eval_resize_mode}'
    )


def results_dir(task, arch, run_name):
    """결과 저장 폴더: test_results/binclf_v3/<ARCH>/<RUN_NAME>/"""
    return test_results_root / TASK_TOKENS[task] / arch / run_name


# ============================================================
# 시드 고정 (v2 신규 — 기존 binclf에는 없던 부분)
# ============================================================

def set_seed(seed=SEED):
    """random/numpy/torch 시드를 고정하고 DataLoader용 generator를 반환한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _worker_init_fn(worker_id):
    """DataLoader 워커별 시드 파생 + cv2 내부 스레드 억제 (워커 수와 곱해져 CPU 경합 방지)."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass


# ============================================================
# 폴리곤 bbox 인덱스 (defect-aware crop용)
# ============================================================

def load_defect_bboxes():
    """annotations_v2.jsonl에서 결함(Class_ID 2·3) annotation의 bbox를
    캐시 좌표계(원본 좌표 × scale)로 변환해 dict로 반환한다.

    annotations_v2.jsonl의 bbox 필드는 모든 annotation에 항상 존재한다
    ([x0,y0,x1,y1] 코너, 폴리곤형은 전처리에서 min/max로 유도해 채움) —
    따라서 여기서는 bbox만 읽으면 된다. polygon 필드는 정밀 윤곽 보존용.

    반환: (dict[source_data_id, list[(x0,y0,x1,y1)]], 제외된 퇴화/빈 annotation 수)
    피클 가능한 기본 타입만 담아 Windows spawn 워커에도 안전하게 복사된다.
    """
    if not annotations_path.exists():
        raise FileNotFoundError(f"annotations_v2.jsonl이 없습니다: {annotations_path}")
    if not image_spec_path.exists():
        raise FileNotFoundError(f"image_spec_v2.csv가 없습니다: {image_spec_path}")

    spec = pd.read_csv(image_spec_path, encoding="utf-8-sig")
    spec_by_id = {
        row.source_data_id: (row.cache_width, row.cache_height, row.scale)
        for row in spec.itertuples(index=False)
    }

    defect_bboxes = {}
    degenerate_count = 0

    with annotations_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            record = json.loads(line)

            # 우수(1) annotation은 결함이 아니므로 크롭 유도 대상에서 제외
            if record["class_id"] not in ("2", "3"):
                continue

            source_id = record["source_data_id"]
            cache_width, cache_height, scale = spec_by_id[source_id]

            bbox = record.get("bbox") or []
            if len(bbox) != 4:
                degenerate_count += 1
                continue
            raw_x0, raw_y0, raw_x1, raw_y1 = bbox

            x0 = max(0.0, raw_x0 * scale)
            y0 = max(0.0, raw_y0 * scale)
            x1 = min(float(cache_width), raw_x1 * scale)
            y1 = min(float(cache_height), raw_y1 * scale)

            # 한 변이 2px 미만인 퇴화 bbox는 크롭 중심으로 쓰기에 부적합
            if (x1 - x0) < 2 or (y1 - y0) < 2:
                degenerate_count += 1
                continue

            defect_bboxes.setdefault(source_id, []).append((x0, y0, x1, y1))

    return defect_bboxes, degenerate_count


# ============================================================
# CLAHE (음영↔균열 혼동 완화 옵션 — 캐시에 굽지 않고 여기서 적용)
# ============================================================

def apply_clahe(pil_image, clahe_object):
    """LAB 색공간의 L(밝기) 채널에만 CLAHE를 적용한다. 색(녹 등) 신호는 보존."""
    import cv2
    array = np.asarray(pil_image)
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe_object.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


# ============================================================
# transform 파이프라인 (크롭/리사이즈 뒤에 적용되는 공통 부분)
# ============================================================

# train: 좌우반전은 Dataset에서 직접, ColorJitter는 여기서
# hue/saturation을 건드리지 않는 이유: 철근 노출의 녹 색 같은 색 신호 보존
_train_post_transform = v2.Compose([
    v2.ToImage(),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=image_mean, std=image_std),
])

_eval_post_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=image_mean, std=image_std),
])


def _letterbox(pil_image):
    """종횡비 유지로 448 박스에 맞춰 축소 후 ImageNet 평균색으로 패딩한다.

    centercrop과 달리 결함이 절대 잘리지 않는 대신 조금 더 축소된다.
    (measure_centercrop_miss_v2.py의 실측 결과에 따라 선택)
    """
    width, height = pil_image.size
    scale = INPUT_SIZE / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), _LETTERBOX_FILL)
    canvas.paste(resized, ((INPUT_SIZE - new_width) // 2, (INPUT_SIZE - new_height) // 2))
    return canvas


# ============================================================
# Dataset
# ============================================================

class BuildingDatasetV2(Dataset):
    """v2 캐시 이미지 + defect-aware crop을 지원하는 Dataset.

    task       : "binclf" (v3 단일 과제)
    split_kind : "train"(무작위 크롭+증강) | "eval"(결정적 리사이즈)
    defect_bboxes : load_defect_bboxes() 결과 (train에서만 필요)
    """

    def __init__(self, dataframe, task, split_kind, defect_bboxes=None):
        if task not in TASK_TOKENS:
            raise ValueError(f'v3의 task는 "binclf"만 지원합니다: {task}')
        if split_kind not in ("train", "eval"):
            raise ValueError(f'split_kind는 "train" 또는 "eval"이어야 합니다: {split_kind}')

        self.dataframe = dataframe
        self.task = task
        self.split_kind = split_kind
        self.defect_bboxes = defect_bboxes or {}

        # cv2.CLAHE 객체는 피클 불가 → 워커 프로세스에서 첫 호출 시 lazy 생성
        self._clahe = None

        # defect-aware crop이 5회 재시도 후 일반 크롭으로 넘어간 횟수
        # (워커 프로세스별 복사본이라 합산은 근사치 — NUM_WORKERS=0 디버깅에서 정확)
        self.defect_crop_fallback_count = 0

    def __len__(self):
        return len(self.dataframe)

    def _sample_defect_crop(self, image_width, image_height, bboxes):
        """결함 bbox 중심 근방에서 크롭 창을 뽑는다. 실패 시 None (일반 크롭으로 fallback).

        RandomResizedCrop과 같은 분포로 크기를 뽑고, 중심만 bbox 쪽으로 유도한다.
        중심 지터 ±25%는 "결함이 항상 정중앙"이라는 위치 편향을 막기 위한 값.
        보장 조건은 "bbox 중심이 크롭 창 안"까지다 — bbox 전체 포함은
        큰 결함(창보다 큰 bbox)에서 불가능하므로 요구하지 않는다.
        """
        area = image_width * image_height

        for _ in range(5):
            target_area = random.uniform(0.6, 1.0) * area
            log_ratio = random.uniform(math.log(3 / 4), math.log(4 / 3))
            aspect = math.exp(log_ratio)

            crop_width = min(image_width, int(round(math.sqrt(target_area * aspect))))
            crop_height = min(image_height, int(round(math.sqrt(target_area / aspect))))

            x0, y0, x1, y1 = random.choice(bboxes)
            bbox_center_x = (x0 + x1) / 2
            bbox_center_y = (y0 + y1) / 2

            center_x = bbox_center_x + random.uniform(-0.25, 0.25) * crop_width
            center_y = bbox_center_y + random.uniform(-0.25, 0.25) * crop_height

            left = int(round(center_x - crop_width / 2))
            top = int(round(center_y - crop_height / 2))
            left = max(0, min(left, image_width - crop_width))
            top = max(0, min(top, image_height - crop_height))

            # clamp 후에도 bbox 중심이 창 안에 있는지 확인
            if (left <= bbox_center_x <= left + crop_width
                    and top <= bbox_center_y <= top + crop_height):
                return top, left, crop_height, crop_width

        self.defect_crop_fallback_count += 1
        return None

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = processed_images_dir / row["image_relpath"]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if use_clahe:
            if self._clahe is None:
                import cv2
                self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = apply_clahe(image, self._clahe)

        if self.split_kind == "train":
            width, height = image.size
            crop_params = None

            # 결함 이미지(불량)는 폴리곤 bbox 중심으로 크롭을 유도해
            # "결함 없는 조각에 불량 라벨" (라벨 노이즈)을 방지한다
            bboxes = self.defect_bboxes.get(row["source_data_id"])
            if bboxes and random.random() < defect_crop_probability:
                crop_params = self._sample_defect_crop(width, height, bboxes)

            if crop_params is None:
                crop_params = _RandomResizedCropV1.get_params(
                    image, scale=[0.6, 1.0], ratio=[3 / 4, 4 / 3]
                )

            top, left, crop_height, crop_width = crop_params
            image = TF.resized_crop(
                image, top, left, crop_height, crop_width,
                [INPUT_SIZE, INPUT_SIZE], antialias=True,
            )

            if random.random() < 0.5:
                image = TF.hflip(image)

            image = _train_post_transform(image)

        else:
            if eval_resize_mode == "centercrop":
                image = TF.resize(image, EVAL_RESIZE_SHORT, antialias=True)
                image = TF.center_crop(image, [INPUT_SIZE, INPUT_SIZE])
            else:
                image = _letterbox(image)

            image = _eval_post_transform(image)

        label = int(row["model_label"])

        return image, label


# ============================================================
# 데이터프레임 로딩 / 클래스 가중치 / 데이터로더
# ============================================================

def load_split_dataframes(task):
    """metadata_split_v2.csv를 읽어 (train, validation, test) 데이터프레임을 반환한다."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata_split_v2.csv를 찾을 수 없습니다: {metadata_path}\n"
            "전처리 v2(라벨) 산출물이 있는지, DATA_DIR이 맞는지 확인하세요."
        )
    if not processed_images_dir.exists():
        raise FileNotFoundError(
            f"이미지 캐시 폴더를 찾을 수 없습니다: {processed_images_dir}\n"
            "processed_images_v2(14GB)를 반입해 data/ 아래에 뒀는지 확인하세요."
        )

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")

    # 방어 검증: 잘못된 CSV를 지정하면 여기서 즉시 실패
    found_labels = set(metadata["model_label"].unique().tolist())
    if not found_labels <= {0, 1}:
        raise ValueError(f"model_label에 0/1 외 값이 있습니다: {sorted(found_labels)}")

    if "middle_id" in metadata.columns:
        raise ValueError(
            "CSV에 middle_id 컬럼이 있습니다 — v3 CSV(metadata_split_v3.csv)가 아니라 "
            "v2 CSV를 지정한 것 같습니다. make_metadata_split_v3.py로 생성하세요."
        )

    found_splits = set(metadata["split"].unique().tolist())
    if found_splits != {"train", "validation", "test"}:
        raise ValueError(f"split 값이 이상합니다: {sorted(found_splits)}")

    # 이미지 캐시 표본 존재 확인 (전수는 validate_image_cache_v2.py 몫)
    for relpath in metadata["image_relpath"].head(5):
        if not (processed_images_dir / relpath).exists():
            raise FileNotFoundError(
                f"캐시 이미지가 없습니다: {processed_images_dir / relpath}\n"
                "processed_images_v2 반입이 불완전한 것 같습니다."
            )

    # 스모크 절단 기준 컬럼: 라벨 분포를 보존해야 함
    stratify_column = "model_label"
    per_class = max(1, limit_per_split // TASK_NUM_CLASSES[task])

    def take_split(split_name):
        split_data = metadata[metadata["split"] == split_name].reset_index(drop=True)
        if limit_per_split > 0:
            split_data = (
                split_data
                .groupby(stratify_column, group_keys=False)
                .head(per_class)
                .reset_index(drop=True)
            )
        return split_data

    return take_split("train"), take_split("validation"), take_split("test")


def compute_class_weights(train_dataframe, task):
    """train 분포에서 클래스 가중치를 계산한다. 공식: 전체 / (클래스 수 × 클래스 장수)

    float() 캐스팅 필수: np.float64를 체크포인트에 넣으면
    torch.load(weights_only=True)에서 로드가 거부됨 (기존 binclf 관례).
    """
    n_classes = TASK_NUM_CLASSES[task]

    counts = train_dataframe["model_label"].value_counts().sort_index()
    keys = list(range(n_classes))

    for key in keys:
        if counts.get(key, 0) == 0:
            raise ValueError(
                f"train에 클래스 {key} 표본이 없습니다. "
                "LIMIT_PER_SPLIT이 너무 작거나 CSV가 잘못됐습니다."
            )

    total_count = len(train_dataframe)
    return [float(total_count / (n_classes * counts[key])) for key in keys]


def build_dataloaders(task, generator=None):
    """(train_loader, validation_loader, test_loader)를 반환한다.

    generator: set_seed()가 돌려준 torch.Generator (셔플 재현성).
    미지정 시 내부에서 시드로 새로 만든다.
    """
    train_data, validation_data, test_data = load_split_dataframes(task)

    defect_bboxes, degenerate_count = load_defect_bboxes()

    # 교차 검증: bbox 수 + 퇴화 제외 수 == metadata의 n_defect_annotations 합
    metadata_defect_total = int(
        pd.concat([train_data, validation_data, test_data])["n_defect_annotations"].sum()
    ) if limit_per_split > 0 else None
    bbox_total = sum(len(boxes) for boxes in defect_bboxes.values())
    if limit_per_split == 0:
        full_defect_total = int(
            pd.read_csv(metadata_path, encoding="utf-8-sig")["n_defect_annotations"].sum()
        )
        if bbox_total + degenerate_count != full_defect_total:
            raise ValueError(
                f"bbox 인덱스({bbox_total}) + 퇴화 제외({degenerate_count}) != "
                f"metadata 결함 annotation 합({full_defect_total})"
            )

    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(SEED)

    train_dataset = BuildingDatasetV2(train_data, task, "train", defect_bboxes)
    validation_dataset = BuildingDatasetV2(validation_data, task, "eval")
    test_dataset = BuildingDatasetV2(test_data, task, "eval")

    use_pin_memory = torch.cuda.is_available()
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common_kwargs
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)

    return train_loader, validation_loader, test_loader


# ============================================================
# 직접 실행 시: 데이터 로딩 자체 점검 (관례)
# ============================================================

if __name__ == "__main__":
    task = os.environ.get("TASK", "binclf")

    print("=========================")
    print("과제 (TASK) :", task)
    print("배치 크기 :", batch_size)
    print("데이터 로딩 워커 수 :", num_workers)
    print("시드 :", SEED)
    print("CLAHE :", "켜짐" if use_clahe else "꺼짐")
    print("평가 리사이즈 모드 :", eval_resize_mode)
    print("defect-aware crop 확률 :", defect_crop_probability)

    if limit_per_split > 0:
        print(f"[스모크 모드] LIMIT_PER_SPLIT={limit_per_split} — 실험 결과에 사용 금지")

    train_data, validation_data, test_data = load_split_dataframes(task)

    label_column = "model_label"
    for split_name, split_data in [
        ("Train", train_data),
        ("Validation", validation_data),
        ("Test", test_data),
    ]:
        distribution = split_data[label_column].value_counts().sort_index().to_dict()
        print(f"{split_name} : {len(split_data)}장 | 분포 : {distribution}")

    print("클래스 가중치 (train 분포) :",
          [round(value, 4) for value in compute_class_weights(train_data, task)])

    defect_bboxes, degenerate_count = load_defect_bboxes()
    print(f"결함 bbox 인덱스 : {sum(len(b) for b in defect_bboxes.values())}개 "
          f"({len(defect_bboxes)}개 이미지, 퇴화 제외 {degenerate_count}개)")

    generator = set_seed()
    train_loader, validation_loader, test_loader = build_dataloaders(task, generator)

    images, labels = next(iter(train_loader))
    print("==========================================")
    print("이미지 묶음 형태 :", images.shape)   # 예상: [배치, 3, 448, 448]
    print("정답 묶음 형태 :", labels.shape)
    print("이미지 자료형 :", images.dtype)      # 예상: torch.float32
    print("정답값 예시 :", labels[:10])
