# -*- coding: utf-8 -*-
"""이미지 전처리 v2: 미세 결함 보존 중간 캐시 생성 스크립트.

원본 이미지(D:\\189...원천데이터)를 종횡비 유지 + 짧은 변 768로 다운스케일해
data/processed_images_v2/ 에 저장하고, 이미지별 실측 스펙을
data/processed/image_spec_v2.csv 로 기록한다.

설계 원칙: "오프라인은 보존, 최종 입력 크기는 학습 시 결정"
- 종횡비 유지 (왜곡 0) - 혼재 해상도(1080x1440 / 1440x1080 / 1080x1920) 자연 흡수
- 짧은 변 768 (기존 224 대비 균열 등 미세 텍스처 보존; 입력 448~640 실험 커버)
- 짧은 변이 768 이하면 업스케일하지 않고 원본 크기 유지
- Image.Resampling.LANCZOS 명시
- JPEG quality=95, subsampling=0 (4:4:4 - 녹/오염 색 신호 보존)
- 회전·색보정(CLAHE 등) 일절 없음 - 중립 캐시. 음영 대책 CLAHE는 학습 transform의
  on/off 옵션으로 (비가역 방지, 서빙 일관성)
- 실측 W/H를 기록 (라벨 JSON의 Resolution은 [1440,1080] 고정 명목값이라 사용 금지)
- scale = cache/orig 균일 배율 1개 → annotations_v2.jsonl 폴리곤 좌표 x scale로 매핑 끝

학습 transform 표준 (학습 코드가 따를 규격, 여기서는 문서화만):
- train: RandomResizedCrop(448, scale=(0.6,1.0)) + HFlip + 약한 밝기/대비 지터(hue 금지)
  + ImageNet Normalize
- val/test/서빙: Resize(짧은 변 512) → CenterCrop(448) (결정적)
- CLAHE 옵션: cv2, LAB의 L채널만, clipLimit=2.0, train/val/서빙 동일 적용 조건으로 비교
- defect-aware crop 옵션: 폴리곤 bbox x scale과 겹치도록 크롭 유도

재실행 안전: 캐시 이미지는 이미 존재하면 다시 만들지 않고 스킵(기존 파일 무수정).
image_spec_v2.csv는 존재하면 즉시 에러 (덮어쓰기 금지, 수동 삭제 후 재실행).

실행:
    python src/build_image_cache_v2.py
"""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

RAW_ROOT = Path(os.environ.get(
    "RAW_ROOT",
    r"D:\189.서울시 노후 주택 균열 데이터\01.데이터\1.Training"
))
DATA_DIR = Path(os.environ.get("DATA_DIR", r"D:\hn_old-building"))

METADATA_PATH = DATA_DIR / "data" / "processed" / "metadata_v2.csv"
SPEC_PATH = DATA_DIR / "data" / "processed" / "image_spec_v2.csv"
CACHE_ROOT = DATA_DIR / "data" / "processed_images_v2"

PREFIX_TO_IMAGE_ROOT = {
    "TS_아파트": RAW_ROOT / "원천데이터" / "TS_아파트",
    "Ts_아파트_add": RAW_ROOT / "원천데이터_231023_add" / "Ts_아파트",
}

CACHE_SHORT_SIDE = 768
JPEG_QUALITY = 95

SPEC_FIELDS = [
    "source_data_id",
    "image_relpath",
    "orig_width",
    "orig_height",
    "cache_width",
    "cache_height",
    "scale",
]


def fail(message):
    print(f"[실패] {message}")
    sys.exit(1)


def process_one(task):
    """워커 프로세스: 원본 1장을 캐시로 변환하고 스펙을 반환한다."""
    source_data_id, relpath = task
    try:
        prefix, _, rest = relpath.partition("/")
        source_path = PREFIX_TO_IMAGE_ROOT[prefix] / rest
        cache_path = CACHE_ROOT / relpath

        if cache_path.exists():
            # 재실행 안전: 기존 캐시는 건드리지 않고 스펙만 다시 읽는다
            with Image.open(source_path) as image:
                orig_width, orig_height = image.size
            with Image.open(cache_path) as cached:
                cache_width, cache_height = cached.size
            return {
                "source_data_id": source_data_id,
                "image_relpath": relpath,
                "orig_width": orig_width,
                "orig_height": orig_height,
                "cache_width": cache_width,
                "cache_height": cache_height,
                "scale": round(cache_width / orig_width, 8),
                "_skipped": True,
            }

        with Image.open(source_path) as image:
            image = image.convert("RGB")
            orig_width, orig_height = image.size

            short_side = min(orig_width, orig_height)
            if short_side > CACHE_SHORT_SIDE:
                scale = CACHE_SHORT_SIDE / short_side
                cache_width = round(orig_width * scale)
                cache_height = round(orig_height * scale)
                image = image.resize(
                    (cache_width, cache_height),
                    Image.Resampling.LANCZOS,
                )
            else:
                # 업스케일 금지: 원본 크기 유지
                cache_width, cache_height = orig_width, orig_height

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(
                cache_path,
                format="JPEG",
                quality=JPEG_QUALITY,
                subsampling=0,
            )

        return {
            "source_data_id": source_data_id,
            "image_relpath": relpath,
            "orig_width": orig_width,
            "orig_height": orig_height,
            "cache_width": cache_width,
            "cache_height": cache_height,
            "scale": round(cache_width / orig_width, 8),
            "_skipped": False,
        }
    except Exception as error:  # 워커 예외는 목록으로 모아 마지막에 보고
        return {"_error": f"{relpath}: {type(error).__name__}: {error}"}


def main():
    if not METADATA_PATH.exists():
        fail(f"metadata_v2.csv가 없습니다 (라벨 전처리 먼저): {METADATA_PATH}")
    if SPEC_PATH.exists():
        fail(f"출력 파일이 이미 존재합니다 (수동 삭제 후 재실행): {SPEC_PATH}")
    for prefix, image_root in PREFIX_TO_IMAGE_ROOT.items():
        if not image_root.exists():
            fail(f"원천 루트가 없습니다: {image_root}")

    with METADATA_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    # 처리 순서 사전순 고정 (스펙 CSV 재현성)
    tasks = sorted(
        (row["source_data_id"], row["image_relpath"]) for row in rows
    )
    print(f"대상 이미지: {len(tasks)}장")
    print(f"캐시 위치: {CACHE_ROOT}")

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"병렬 워커: {workers}개")

    specs = []
    errors = []
    skipped = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(
            executor.map(process_one, tasks, chunksize=64)
        ):
            if index % 2000 == 0 and index > 0:
                print(f"  {index}/{len(tasks)} 처리 중...")
            if "_error" in result:
                errors.append(result["_error"])
                continue
            if result.pop("_skipped"):
                skipped += 1
            specs.append(result)

    if errors:
        print(f"\n[실패 목록] {len(errors)}건:")
        for message in errors[:20]:
            print(" ", message)
        fail(f"이미지 {len(errors)}건 처리 실패 - 스펙 CSV를 저장하지 않습니다")

    if len(specs) != len(tasks):
        fail(f"처리 수 불일치: {len(specs)} != {len(tasks)}")

    with SPEC_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SPEC_FIELDS)
        writer.writeheader()
        writer.writerows(specs)

    from collections import Counter
    size_counts = Counter(
        (spec["orig_width"], spec["orig_height"]) for spec in specs
    )

    print("\n===== 캐시 생성 완료 =====")
    print("image_spec_v2.csv :", SPEC_PATH)
    print("총 이미지 :", len(specs), f"(기존 캐시 재사용 {skipped}장)")
    print("원본 해상도 분포 (이미지 수):")
    for (width, height), count in size_counts.most_common():
        print(f"  {width}x{height} : {count}")


if __name__ == "__main__":
    main()
