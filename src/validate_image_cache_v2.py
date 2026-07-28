# -*- coding: utf-8 -*-
"""이미지 캐시 v2 검증 스크립트.

processed_images_v2 캐시와 image_spec_v2.csv를 전수 검증하고
결과를 data/processed/image_cache_v2_report.txt 로 저장한다 (콘솔에도 동일 출력).

검증 항목
1. 전수 대사: metadata_split_v2.csv 37,285행 <-> image_spec_v2.csv <-> 캐시 파일 1:1
2. 종횡비 보존: |cache_w/cache_h - orig_w/orig_h| < 0.01 전 행
3. 짧은 변 == 768 (원본이 더 작아 업스케일 금지가 발동한 예외는 카운트)
4. 실측 해상도 분포: 그룹 단위 1080x1440=1,636 / 1440x1080=69 / 1080x1920=10 대조
5. 폴리곤 정합: 그룹당 1장 샘플에서 annotations_v2.jsonl 좌표가 원본 경계 내,
   좌표 x scale이 캐시 경계 내인지
6. 캐시 재로드 무결성 샘플(200장) + 용량 합계

실행:
    python src/validate_image_cache_v2.py
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

DATA_DIR = Path(os.environ.get("DATA_DIR", r"D:\hn_old-building"))

PROCESSED_DIR = DATA_DIR / "data" / "processed"
SPLIT_PATH = PROCESSED_DIR / "metadata_split_v2.csv"
SPEC_PATH = PROCESSED_DIR / "image_spec_v2.csv"
ANNOTATIONS_PATH = PROCESSED_DIR / "annotations_v2.jsonl"
REPORT_PATH = PROCESSED_DIR / "image_cache_v2_report.txt"
CACHE_ROOT = DATA_DIR / "data" / "processed_images_v2"

CACHE_SHORT_SIDE = 768

# 실측 기대값 (이미지 단위 해상도 분포, 2026-07 캐시 생성 시 확정)
EXPECTED_IMAGE_SIZES = {
    (1080, 1440): 35465,
    (1440, 1080): 1662,
    (1080, 1920): 157,
    (1072, 1440): 1,  # S-211111_A_T_3_R_9636015-3 프레임 추출 이상 1장 (폭 8px 부족)
}

# 그룹 내 해상도가 섞여도 되는 알려진 예외 (위 1072x1440 프레임이 속한 그룹)
KNOWN_MIXED_GROUPS = {"S-211111_A_T_3_R_9636015"}

report_lines = []
failures = []


def emit(message=""):
    print(message)
    report_lines.append(str(message))


def check(passed, message):
    tag = "PASS" if passed else "FAIL"
    emit(f"[{tag}] {message}")
    if not passed:
        failures.append(message)


def main():
    for path in (SPLIT_PATH, SPEC_PATH, ANNOTATIONS_PATH):
        if not path.exists():
            print(f"[실패] 입력 파일이 없습니다: {path}")
            sys.exit(1)
    if REPORT_PATH.exists():
        print(f"[실패] 리포트가 이미 존재합니다 (수동 삭제 후 재실행): {REPORT_PATH}")
        sys.exit(1)

    emit("========== 이미지 캐시 v2 검증 리포트 ==========")
    emit(f"캐시: {CACHE_ROOT}")
    emit(f"스펙: {SPEC_PATH}")
    emit()

    split_df = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    spec_df = pd.read_csv(SPEC_PATH, encoding="utf-8-sig")

    # ---------- 1. 전수 대사 ----------
    emit("----- 1. 전수 대사 -----")
    check(len(spec_df) == len(split_df),
          f"스펙 행 수 {len(spec_df)} == 라벨 행 수 {len(split_df)}")
    check(spec_df["source_data_id"].is_unique, "스펙 source_data_id 중복 없음")
    check(set(spec_df["source_data_id"]) == set(split_df["source_data_id"]),
          "스펙과 라벨의 source_data_id 집합 일치")

    merged = split_df.merge(spec_df, on="source_data_id", suffixes=("", "_spec"))
    check((merged["image_relpath"] == merged["image_relpath_spec"]).all(),
          "라벨과 스펙의 image_relpath 일치")

    print("캐시 파일 실존 확인 중 (수십 초 소요)...")
    missing_cache = [
        relpath for relpath in spec_df["image_relpath"]
        if not (CACHE_ROOT / relpath).exists()
    ]
    check(len(missing_cache) == 0, f"캐시 파일 실존 (누락 {len(missing_cache)}건)")

    # ---------- 2~3. 종횡비·짧은 변 ----------
    emit()
    emit("----- 2. 종횡비 / 짧은 변 -----")
    orig_ratio = spec_df["orig_width"] / spec_df["orig_height"]
    cache_ratio = spec_df["cache_width"] / spec_df["cache_height"]
    ratio_bad = int(((orig_ratio - cache_ratio).abs() >= 0.01).sum())
    check(ratio_bad == 0, f"종횡비 보존 위반 {ratio_bad}건 (허용 오차 0.01)")

    cache_short = spec_df[["cache_width", "cache_height"]].min(axis=1)
    orig_short = spec_df[["orig_width", "orig_height"]].min(axis=1)
    no_upscale = spec_df[orig_short <= CACHE_SHORT_SIDE]
    scaled = spec_df[orig_short > CACHE_SHORT_SIDE]
    check((cache_short[scaled.index] == CACHE_SHORT_SIDE).all(),
          f"다운스케일 대상 짧은 변 == {CACHE_SHORT_SIDE}")
    emit(f"업스케일 금지 발동(원본 유지): {len(no_upscale)}건")

    scale_bad = int(
        (spec_df["scale"] - spec_df["cache_width"] / spec_df["orig_width"])
        .abs().gt(1e-6).sum()
    )
    check(scale_bad == 0, f"scale == cache_width/orig_width 정합 위반 {scale_bad}건")

    # ---------- 4. 실측 해상도 분포 ----------
    emit()
    emit("----- 3. 실측 해상도 분포 (그룹 단위) -----")
    group_sizes = (
        merged.groupby("group_id")[["orig_width", "orig_height"]]
        .agg(["nunique"])
    )
    mixed_mask = (
        (group_sizes[("orig_width", "nunique")] > 1)
        | (group_sizes[("orig_height", "nunique")] > 1)
    )
    mixed_group_ids = set(group_sizes[mixed_mask].index)
    check(mixed_group_ids <= KNOWN_MIXED_GROUPS,
          f"알려진 예외 외 그룹 내 해상도 혼재: {sorted(mixed_group_ids - KNOWN_MIXED_GROUPS)[:5]}")

    image_size_counts = Counter(
        zip(merged["orig_width"], merged["orig_height"])
    )
    for size, expected in EXPECTED_IMAGE_SIZES.items():
        actual = image_size_counts.get(size, 0)
        check(actual == expected,
              f"{size[0]}x{size[1]} 이미지 수 {actual} (기대 {expected})")
    unexpected = set(image_size_counts) - set(EXPECTED_IMAGE_SIZES)
    check(not unexpected, f"예상 외 해상도 {sorted(unexpected)}")

    # ---------- 5. 폴리곤 정합 (그룹당 1장 샘플) ----------
    emit()
    emit("----- 4. 폴리곤 정합 (그룹당 1장 샘플) -----")
    sample_ids = set(
        merged.sort_values("source_data_id")
        .groupby("group_id")["source_data_id"].first()
    )
    spec_by_id = spec_df.set_index("source_data_id")

    poly_out_of_orig = 0
    poly_out_of_cache = 0
    sampled_annotations = 0
    with ANNOTATIONS_PATH.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            record = json.loads(line)
            source_id = record["source_data_id"]
            if source_id not in sample_ids:
                continue
            sampled_annotations += 1
            spec = spec_by_id.loc[source_id]
            # 좌표는 두 형태: polygon(평탄화 [x1,y1,...]) 또는 bbox([x0,y0,x1,y1] 코너)
            polygon = record.get("polygon") or []
            bbox = record.get("bbox") or []
            if polygon:
                xs = polygon[0::2]
                ys = polygon[1::2]
            elif len(bbox) == 4:
                xs = [bbox[0], bbox[2]]
                ys = [bbox[1], bbox[3]]
            else:
                continue
            if (max(xs) > spec["orig_width"] or max(ys) > spec["orig_height"]
                    or min(xs) < 0 or min(ys) < 0):
                poly_out_of_orig += 1
            scale = spec["scale"]
            if (max(xs) * scale > spec["cache_width"] + 1
                    or max(ys) * scale > spec["cache_height"] + 1):
                poly_out_of_cache += 1

    emit(f"샘플 그룹 수: {len(sample_ids)} / 검사한 annotation: {sampled_annotations}")
    check(poly_out_of_orig == 0,
          f"폴리곤 좌표가 원본(실측 W/H) 경계 밖인 annotation {poly_out_of_orig}건")
    check(poly_out_of_cache == 0,
          f"폴리곤 x scale이 캐시 경계 밖인 annotation {poly_out_of_cache}건")

    # ---------- 6. 재로드 무결성 + 용량 ----------
    emit()
    emit("----- 5. 캐시 재로드 무결성 / 용량 -----")
    rng = random.Random(42)
    reload_sample = rng.sample(list(spec_df.itertuples(index=False)), 200)
    reload_bad = 0
    for spec in reload_sample:
        try:
            with Image.open(CACHE_ROOT / spec.image_relpath) as image:
                if image.size != (spec.cache_width, spec.cache_height):
                    reload_bad += 1
        except Exception:
            reload_bad += 1
    check(reload_bad == 0, f"재로드 샘플 200장 중 불량 {reload_bad}건")

    total_bytes = sum(
        (CACHE_ROOT / relpath).stat().st_size
        for relpath in spec_df["image_relpath"]
    )
    emit(f"캐시 총 용량: {total_bytes / 1024**3:.2f} GB "
         f"(평균 {total_bytes / len(spec_df) / 1024:.0f} KB/장)")

    # ---------- 결과 ----------
    emit()
    if failures:
        emit(f"========== 검증 실패 {len(failures)}건 ==========")
        for message in failures:
            emit(f"  - {message}")
    else:
        emit("========== 전 항목 통과 ==========")

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8-sig")
    print(f"\n리포트 저장: {REPORT_PATH}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
