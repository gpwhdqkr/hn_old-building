# -*- coding: utf-8 -*-
"""전처리 v2 리사이즈본 델타 보충 스크립트.

metadata_v2.csv의 image_relpath 기준으로 data/processed_images/에
누락된 224x224 리사이즈본만 새로 생성한다. 기존 파일은 절대 덮어쓰지 않는다
(존재하면 무조건 스킵).

리사이즈 파라미터는 기존 resize_images.py와 동일하게 유지한다
(convert("RGB") + resize((224, 224)), PIL 기본 리샘플링) - 기존 37,285장과의
화질 일관성을 위해 바꾸지 말 것.

예상 결과: 기존 리사이즈본이 완비되어 있으므로 "델타 0건"으로 종료.
이 스크립트의 실질 역할은 리사이즈본 완전성 검증 + 미래 구제분 대비다.

실행:
    python src/resize_delta_images_v2.py
"""

import csv
import os
import sys
from pathlib import Path

from PIL import Image

raw_root = Path(os.environ.get(
    "RAW_ROOT",
    r"D:\189.서울시 노후 주택 균열 데이터\01.데이터\1.Training"
))
data_dir = Path(os.environ.get("DATA_DIR", r"D:\hn_old-building"))

metadata_path = data_dir / "data" / "processed" / "metadata_v2.csv"
processed_images_dir = data_dir / "data" / "processed_images"

# image_relpath 접두 → 원본 실경로 루트 역매핑
PREFIX_TO_IMAGE_ROOT = {
    "TS_아파트": raw_root / "원천데이터" / "TS_아파트",
    "Ts_아파트_add": raw_root / "원천데이터_231023_add" / "Ts_아파트",
}


def fail(message):
    print(f"[실패] {message}")
    sys.exit(1)


def main():
    if not metadata_path.exists():
        fail(f"metadata_v2.csv가 없습니다 (create_metadata_v2.py 먼저 실행): {metadata_path}")
    if not processed_images_dir.exists():
        fail(f"processed_images 폴더가 없습니다: {processed_images_dir}")

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    print("metadata_v2.csv 행 수 :", len(rows))

    missing = []
    for row in rows:
        relpath = row["image_relpath"]
        if not (processed_images_dir / relpath).exists():
            missing.append(relpath)

    print("리사이즈본 누락 :", len(missing), "건")

    if not missing:
        print("델타 0건 - 생성할 이미지가 없습니다 (리사이즈본 완전성 확인 완료).")
        return

    created = []
    for relpath in missing:
        prefix, _, rest = relpath.partition("/")
        image_root = PREFIX_TO_IMAGE_ROOT.get(prefix)
        if image_root is None:
            fail(f"알 수 없는 접두 폴더 '{prefix}': {relpath}")

        source_path = image_root / rest
        if not source_path.exists():
            fail(f"원본 이미지가 없습니다: {source_path}")

        saved_path = processed_images_dir / relpath
        if saved_path.exists():
            # 덮어쓰기 방지 이중 확인
            continue

        saved_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path) as image:
            image = image.convert("RGB")
            image = image.resize((224, 224))
            image.save(saved_path)

        created.append(relpath)
        print("생성 :", relpath)

    print("\n델타 리사이즈 완료 :", len(created), "건 생성")


if __name__ == "__main__":
    main()
