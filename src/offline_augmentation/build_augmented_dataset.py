# ============================================================
# [사용법]
# offline augmentation: '보통' 등급 train 이미지를 미리 변형해 디스크에 저장하고,
# 원본 + 증강본이 합쳐진 확장 CSV를 만드는 스크립트입니다. (실험당 1회 실행)
#
# 기존 파이프라인의 증강은 학습 중 매번 새로 만드는 온라인 증강 2가지뿐입니다
# (좌우반전 50%, 밝기/대비 ±10% — preprocess_efficientnet.py). 이 스크립트는
# 그와 겹치지 않는 변형(회전, 이동/확대축소, 랜덤크롭, 채도/색조)을 써서
# '보통' 등급의 학습 이미지 자체를 물리적으로 늘립니다.
#
# 실행 (윈도우 PowerShell):
#   $env:RUN_NAME="aug3"; $env:AUG_FACTOR="3"
#   python src/offline_augmentation/build_augmented_dataset.py
#
# [주의: PowerShell은 리눅스식 인라인 환경변수를 못 씁니다]
# 기존 문서에 적힌 리눅스 문법(RUN_NAME=aug3 python ...)을 PowerShell에
# 그대로 치면 파서 오류가 납니다. 반드시 위처럼 $env: 로 먼저 지정하세요.
# 런팟(리눅스)에서 돌릴 때는 기존 문법을 그대로 쓰면 됩니다:
#   RUN_NAME=aug3 AUG_FACTOR=3 python src/offline_augmentation/build_augmented_dataset.py
#
# [환경변수]
#   RUN_NAME    실험 이름 (기본 default) — 확장 CSV 파일명에 붙음
#   AUG_FACTOR  '보통' 등급을 몇 배로 늘릴지 (기본 3)
#               1이면 증강본을 만들지 않음 → 증강 없는 대조군용 CSV가 만들어짐
#   AUG_SEED    증강 난수 시드 (기본 42)
#   AUG_QUALITY 저장 JPEG 품질 (기본 95)
#
# [전제 조건]
# - data/processed/metadata_split.csv (팀원이 만든 split 결과, 깃에 포함)
# - data/processed_images/ 에 224x224 변환 이미지 37,285장
#   (용량 문제로 깃에 없음 → 구글드라이브에서 받아 이 위치에 풀어야 함)
# 시작할 때 이미지가 전부 있는지 먼저 전수 검사합니다. 학습을 몇 분 돌린 뒤에
# 파일이 없어서 죽는 일을 막기 위한 것입니다.
#
# 결과:
#   data/augmented_images/<원본과 같은 폴더구조>/<원본이름>__aug{n}.jpg
#   data/augmented_images/metadata_split_augmented_<RUN_NAME>.csv
#
# 다음 단계 (같은 RUN_NAME을 그대로 넘겨야 함):
#   python src/offline_augmentation/train_augmented.py
#
# [기존 코드와의 관계]
# - 기존 파일은 하나도 수정하지 않음. 이 폴더 안에서 자급자족으로 동작함
# - split은 팀원의 metadata_split.csv를 그대로 사용. 증강본은 원본의 group_id와
#   split을 그대로 물려받으므로 train/validation/test 누수가 생기지 않음
# - 증강은 train split의 '보통'(model_label=1)에만 적용. validation/test는
#   한 장도 건드리지 않음 (평가 기준이 흔들리면 비교가 무의미해지므로)
# ============================================================

import os
import zlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, v2

# 이 파일이 src/offline_augmentation/ 안에 있으므로 세 단계 올라가면 프로젝트 폴더
project_dir = Path(__file__).resolve().parent.parent.parent

# 실험 이름: 확장 CSV 파일명에 붙어서 실험별로 CSV가 따로 쌓임
# (대조군 CSV와 실험군 CSV가 공존하므로 실험 순서에 제약이 없음)
RUN_NAME = os.environ.get("RUN_NAME", "default")

# '보통' 등급을 몇 배로 늘릴지 (원본 1장당 증강본 AUG_FACTOR-1장)
AUG_FACTOR = int(os.environ.get("AUG_FACTOR", 3))

# 증강 난수 시드 (이미지별 시드 계산의 기준값)
AUG_SEED = int(os.environ.get("AUG_SEED", 42))

# 저장 JPEG 품질. 이미 JPEG인 원본을 다시 인코딩하므로 이중 압축 손실이
# 생기는데, 얇은 균열이 핵심 특징인 데이터라 품질을 높게 잡음
AUG_QUALITY = int(os.environ.get("AUG_QUALITY", 95))

# 팀원이 만든 split 결과 CSV (읽기만 함)
metadata_path = project_dir / "data" / "processed" / "metadata_split.csv"

# 224x224 변환 이미지 폴더 (읽기만 함)
processed_images_dir = project_dir / "data" / "processed_images"

# 증강 결과를 저장할 폴더 (원본 폴더 구조를 그대로 미러링)
augmented_images_dir = project_dir / "data" / "augmented_images"

# 원본 + 증강본이 합쳐진 확장 CSV
augmented_metadata_path = (
    augmented_images_dir / f"metadata_split_augmented_{RUN_NAME}.csv"
)

# CSV의 경로에서 이 문자열 뒤쪽이 실제 폴더 구조와 일치함
# 예: D:/hn_old-building_raw/raw/images/TS_아파트/.../xxx.jpg
#     → raw/images/ 뒤인 TS_아파트/.../xxx.jpg 만 잘라서 씀
RAW_PATH_MARKER = "raw/images/"

# 증강 대상 등급 (0=우수, 1=보통, 2=불량)
TARGET_MODEL_LABEL = 1

# 증강 파이프라인 버전 이름. 확장 CSV의 augment_recipe에 시드와 함께 기록해서,
# 나중에 파이프라인을 바꿔도 어떤 규칙으로 만든 행인지 구분할 수 있게 함
AUGMENT_RECIPE_NAME = "geo_color_v1"


# 오프라인 증강 파이프라인
#
# 기존 온라인 증강(좌우반전, 밝기/대비 ±10%)과 겹치지 않는 변형을 중심으로 구성.
# ToImage로 텐서(uint8)로 바꿔 작업한 뒤 ToPILImage로 되돌려 저장한다
# (정규화는 하지 않음 — 저장하는 건 어디까지나 눈에 보이는 이미지여야 하므로).
augment_transform = v2.Compose([
    # PIL 이미지를 PyTorch 이미지 형태로 변환 (uint8 유지)
    v2.ToImage(),
    # 좌우반전
    v2.RandomHorizontalFlip(
        p=0.5
    ),
    # 회전 전에 테두리를 반사(거울) 방식으로 덧붙임.
    # 224 이미지를 그냥 회전하면 네 귀퉁이에 검은 삼각형이 남고, 모델이 그걸
    # '보통 등급의 특징'으로 학습해버린다. 실제 이미지를 반사해 채운 뒤
    # 아래 CenterCrop으로 되돌려 자르면 검은 여백이 한 픽셀도 남지 않는다
    # (224+32*2=288을 15도 돌려도 중앙 224 영역은 전부 실제 화소로 채워짐).
    v2.Pad(
        padding=32,
        padding_mode="reflect"
    ),
    # 회전 ±15도. 균열은 방향 자체가 의미 있는 특징이라 크게 돌리면
    # 오히려 등급 판단 근거가 흐려지므로 좁게 제한
    v2.RandomRotation(
        degrees=15,
        interpolation=InterpolationMode.BILINEAR
    ),
    # 위에서 덧붙인 테두리를 제거해 원래 크기로 복귀
    v2.CenterCrop(
        224
    ),
    # 일부만 잘라 확대 (촬영 거리가 달라진 상황을 흉내냄)
    v2.RandomResizedCrop(
        size=(224, 224),
        scale=(0.8, 1.0),
        ratio=(0.9, 1.11),
        antialias=True
    ),
    # 색상 변형. 채도(saturation)와 색조(hue)는 온라인 증강에 없는 항목이라
    # 여기서 넣는 실익이 크고, 밝기/대비는 온라인(±10%)보다 조금 넓게 줌
    v2.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.20,
        hue=0.03
    ),
    # 저장할 수 있도록 PIL 이미지로 되돌림
    v2.ToPILImage()
])


def convert_to_relative_path(csv_image_path):
    """CSV에 적힌 팀원 PC 경로에서 raw/images/ 뒤쪽 상대 경로만 잘라낸다.

    예) D:/hn_old-building_raw/raw/images/TS_아파트/a/b.jpg
        → TS_아파트/a/b.jpg
    """
    # 윈도우 역슬래시가 섞여 있어도 처리되도록 통일
    posix_path = str(csv_image_path).replace("\\", "/")

    marker_index = posix_path.find(RAW_PATH_MARKER)

    if marker_index == -1:
        raise ValueError(
            f"이미지 경로에서 '{RAW_PATH_MARKER}'를 찾을 수 없습니다: {csv_image_path}"
        )

    return posix_path[marker_index + len(RAW_PATH_MARKER):]


def make_augment_seed(relative_path, augment_index):
    """이미지 하나하나에 고정된 시드를 만든다.

    전역 시드를 한 번만 걸고 순서대로 돌리는 방식은, 처리 순서가 바뀌거나
    중간에 끊겨서 다시 실행하면 결과가 달라진다. 경로와 증강 번호로 시드를
    계산하면 순서와 무관하게 항상 같은 결과가 나오고, AUG_FACTOR를 3에서 5로
    늘려도 기존 __aug1, __aug2는 그대로 유지된다.

    파이썬 내장 hash()는 문자열에 대해 실행할 때마다 값이 바뀌므로
    (PYTHONHASHSEED 무작위화) 실행 간에 일정한 zlib.crc32를 쓴다.
    """
    path_hash = zlib.crc32(relative_path.encode("utf-8"))

    return AUG_SEED + path_hash + augment_index * 1_000_003


def verify_original_images(relative_paths):
    """확장 CSV에 들어갈 원본 이미지가 전부 디스크에 있는지 먼저 확인한다.

    구글드라이브에서 받아 푸는 과정에서 일부 폴더가 빠지는 일이 흔한데,
    학습을 몇 분 돌린 뒤에 FileNotFoundError로 죽으면 시간이 아깝다.
    """
    missing_paths = [
        relative_path
        for relative_path in relative_paths
        if not (processed_images_dir / relative_path).exists()
    ]

    if not missing_paths:
        print(f"원본 이미지 확인 : {len(relative_paths)}장 전부 있음")
        return

    # 어느 폴더에서 몇 장이 빠졌는지 알려주면 복사를 다시 해야 할 범위가 보임
    missing_count_by_folder = {}

    for relative_path in missing_paths:
        folder = str(Path(relative_path).parent)
        missing_count_by_folder[folder] = missing_count_by_folder.get(folder, 0) + 1

    message_lines = [
        f"원본 이미지 {len(missing_paths)}장이 없습니다 "
        f"(전체 {len(relative_paths)}장 중).",
        f"확인한 폴더: {processed_images_dir}",
        "",
        "빠진 장수가 많은 폴더 상위 10개:"
    ]

    sorted_folders = sorted(
        missing_count_by_folder.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for folder, count in sorted_folders[:10]:
        message_lines.append(f"  {count:6d}장  {folder}")

    message_lines.append("")
    message_lines.append(
        "구글드라이브에서 받은 processed_images를 data/ 아래에 "
        "빠짐없이 풀었는지 확인하세요."
    )

    raise FileNotFoundError("\n".join(message_lines))


def build_original_rows(metadata):
    """원본 행에 image_source / resolved_relpath / augment_recipe를 붙인다."""
    original_rows = metadata.copy()

    original_rows["image_source"] = "original"

    original_rows["resolved_relpath"] = [
        f"data/processed_images/{convert_to_relative_path(image_path)}"
        for image_path in original_rows["image_path"]
    ]

    original_rows["augment_recipe"] = ""

    return original_rows


def build_augmented_rows(metadata):
    """'보통' 등급 train 행을 증강해 이미지를 저장하고 새 행 목록을 만든다."""
    target_rows = metadata[
        (metadata["split"] == "train")
        & (metadata["model_label"] == TARGET_MODEL_LABEL)
    ]

    # 처리 순서를 고정해 실행할 때마다 진행 로그가 같게 나오도록 함
    # (시드는 경로 기반이라 순서와 무관하지만, 로그를 비교하기 쉬워짐)
    target_rows = target_rows.sort_values("image_path").reset_index(drop=True)

    print(
        f"증강 대상 : 보통(model_label={TARGET_MODEL_LABEL}) train "
        f"{len(target_rows)}장 → 원본 1장당 {AUG_FACTOR - 1}장 생성"
    )

    augmented_rows = []
    created_count = 0
    skipped_count = 0

    for row_index in range(len(target_rows)):
        row = target_rows.iloc[row_index]

        relative_path = convert_to_relative_path(row["image_path"])
        source_path = processed_images_dir / relative_path

        relative_path_object = Path(relative_path)

        for augment_index in range(1, AUG_FACTOR):
            augmented_name = (
                f"{relative_path_object.stem}__aug{augment_index}"
                f"{relative_path_object.suffix}"
            )
            augmented_relative_path = relative_path_object.parent / augmented_name
            augmented_path = augmented_images_dir / augmented_relative_path

            # 이미 만들어 둔 파일은 다시 만들지 않음 (중간에 끊겨도 이어서 진행)
            if augmented_path.exists():
                skipped_count += 1

            else:
                augmented_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                seed = make_augment_seed(relative_path, augment_index)

                # 이 시드가 아래 변형에 쓰이는 모든 난수를 결정함
                torch.manual_seed(seed)

                with Image.open(source_path) as image_file:
                    image = image_file.convert("RGB")

                augmented_image = augment_transform(image)

                augmented_image.save(
                    augmented_path,
                    quality=AUG_QUALITY,
                    subsampling=0
                )

                created_count += 1

            # 증강본은 원본의 등급/그룹/split을 그대로 물려받는다.
            # group_id를 그대로 쓰는 것이 중요 — split이 group_id 단위로
            # 나뉘어 있으므로, 다른 group으로 새어 나가면 train/test 누수가 된다
            augmented_rows.append({
                "image_path": str(row["image_path"]),
                "source_data_id": f"{row['source_data_id']}__aug{augment_index}",
                "class_id": row["class_id"],
                "model_label": row["model_label"],
                "class_name": row["class_name"],
                "group_id": row["group_id"],
                "split": row["split"],
                "image_source": "augmented",
                "resolved_relpath": (
                    f"data/augmented_images/"
                    f"{augmented_relative_path.as_posix()}"
                ),
                "augment_recipe": (
                    f"{AUGMENT_RECIPE_NAME}:seed="
                    f"{make_augment_seed(relative_path, augment_index)}"
                )
            })

        if (row_index + 1) % 500 == 0:
            print(
                f"  진행 {row_index + 1}/{len(target_rows)} "
                f"(새로 만듦 {created_count}, 건너뜀 {skipped_count})"
            )

    print(f"증강 이미지 : 새로 만듦 {created_count}장, 이미 있어서 건너뜀 {skipped_count}장")

    return pd.DataFrame(augmented_rows)


def report_dataset_summary(augmented_metadata):
    """확장 CSV가 의도대로 만들어졌는지 자체 점검한 결과를 출력한다."""
    print("\n==========================================")
    print("확장 데이터셋 점검")

    print(f"전체 행 수 : {len(augmented_metadata)}")

    print("\nsplit x 등급별 장수")
    print(
        pd.crosstab(
            augmented_metadata["split"],
            augmented_metadata["class_name"]
        ).to_string()
    )

    print("\nsplit x 원본/증강본")
    print(
        pd.crosstab(
            augmented_metadata["split"],
            augmented_metadata["image_source"]
        ).to_string()
    )

    # 증강본이 train의 '보통'에만 들어갔는지 확인
    augmented_only = augmented_metadata[
        augmented_metadata["image_source"] == "augmented"
    ]

    if len(augmented_only) > 0:
        wrong_split = augmented_only[augmented_only["split"] != "train"]
        wrong_label = augmented_only[
            augmented_only["model_label"] != TARGET_MODEL_LABEL
        ]

        if len(wrong_split) > 0 or len(wrong_label) > 0:
            raise ValueError(
                f"증강본이 잘못된 곳에 들어갔습니다: "
                f"train이 아닌 행 {len(wrong_split)}개, "
                f"보통이 아닌 행 {len(wrong_label)}개"
            )

        print("\n증강본 위치 확인 : 전부 train 의 보통 등급 (정상)")

    # group_id 누수 확인 — split끼리 group이 겹치면 안 됨
    group_by_split = {
        split_name: set(
            augmented_metadata[augmented_metadata["split"] == split_name]["group_id"]
        )
        for split_name in ["train", "validation", "test"]
    }

    leak_pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test")
    ]

    for first_split, second_split in leak_pairs:
        overlap = group_by_split[first_split] & group_by_split[second_split]

        if overlap:
            raise ValueError(
                f"group_id 누수 발견: {first_split} 와 {second_split} 가 "
                f"{len(overlap)}개 그룹을 공유합니다"
            )

    print("group_id 누수 확인 : train/validation/test 겹침 없음 (정상)")

    # 확장 CSV의 모든 경로가 실제로 존재하는지 전수 확인
    missing_count = sum(
        1
        for relative_path in augmented_metadata["resolved_relpath"]
        if not (project_dir / relative_path).exists()
    )

    if missing_count > 0:
        raise FileNotFoundError(
            f"확장 CSV에 적힌 이미지 {missing_count}장이 실제로 없습니다"
        )

    print("이미지 실존 확인 : 확장 CSV의 모든 경로가 디스크에 있음 (정상)")


def main():
    print("실험 이름 (RUN_NAME) :", RUN_NAME)
    print("증강 배수 (AUG_FACTOR) :", AUG_FACTOR)
    print("증강 시드 (AUG_SEED) :", AUG_SEED)

    if AUG_FACTOR < 1:
        raise ValueError(
            f"AUG_FACTOR는 1 이상이어야 합니다 (받은 값: {AUG_FACTOR}). "
            "1이면 증강 없는 대조군이 됩니다."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata_split.csv를 찾을 수 없습니다: {metadata_path}\n"
            "깃에서 data/processed/metadata_split.csv를 받았는지 확인하세요."
        )

    if not processed_images_dir.exists():
        raise FileNotFoundError(
            f"이미지 폴더를 찾을 수 없습니다: {processed_images_dir}\n"
            "구글드라이브에서 processed_images를 받아 data/ 아래에 풀었는지 확인하세요."
        )

    metadata = pd.read_csv(
        metadata_path,
        encoding="utf-8-sig"
    )

    print(f"\n원본 CSV : {len(metadata)}행")

    # 원본 이미지가 전부 있는지 먼저 확인 (없으면 여기서 중단)
    original_relative_paths = [
        convert_to_relative_path(image_path)
        for image_path in metadata["image_path"]
    ]

    verify_original_images(original_relative_paths)

    original_rows = build_original_rows(metadata)

    if AUG_FACTOR == 1:
        print("\nAUG_FACTOR=1 이므로 증강본을 만들지 않습니다 (대조군).")
        augmented_metadata = original_rows

    else:
        print()
        augmented_rows = build_augmented_rows(metadata)
        augmented_metadata = pd.concat(
            [original_rows, augmented_rows],
            ignore_index=True
        )

    report_dataset_summary(augmented_metadata)

    augmented_images_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # utf-8-sig: 윈도우 메모장/엑셀에서도 한글이 깨지지 않게
    augmented_metadata.to_csv(
        augmented_metadata_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n==========================================")
    print("확장 CSV 저장 :", augmented_metadata_path)
    print("다음 단계 : python src/offline_augmentation/train_augmented.py")


if __name__ == "__main__":
    main()
