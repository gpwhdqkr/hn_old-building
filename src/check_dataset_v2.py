# -*- coding: utf-8 -*-
"""전처리 v2 사전 점검 스크립트 (산출물 없음, 콘솔 리포트만).

원천 데이터(D:\\189.서울시 노후 주택 균열 데이터)를 스캔해서
create_metadata_v2.py를 실행해도 되는 상태인지 확인한다.

점검 항목
- 라벨/원천 루트 4개 존재 여부
- RGB 라벨 JSON 수 (기대 37,285), 열화상 JSON 수 (제외 대상)
- JSON 스키마 키 존재 (Raw_Data_ID / Middle_ID / Shooting_ID / Annotations)
- 이미지 미매칭 / 빈 Annotations / Class_ID 혼합 / 유효 외 Class_ID 카운트
- (middle_id x 등급) 분포표

기대값과 어긋나면 종료 코드 1 (진행 여부는 사람이 판단).

실행:
    python src/check_dataset_v2.py
원천 데이터가 다른 위치에 있으면:
    (PowerShell) $env:RAW_ROOT="E:/다른위치/1.Training"; python src/check_dataset_v2.py
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

raw_root = Path(os.environ.get(
    "RAW_ROOT",
    r"D:\189.서울시 노후 주택 균열 데이터\01.데이터\1.Training"
))

# 본편/add편의 라벨·원천 루트와, 기존 CSV(processed_images)에서 쓰는 접두 폴더명.
# add편은 원천 폴더명(Ts_아파트)과 접두(Ts_아파트_add)가 다르므로 매핑이 필수다.
EDITIONS = [
    {
        "name": "본편",
        "label_root": raw_root / "라벨링데이터" / "TL_아파트",
        "image_root": raw_root / "원천데이터" / "TS_아파트",
        "prefix": "TS_아파트",
    },
    {
        "name": "add편",
        "label_root": raw_root / "라벨링데이터_231023_add" / "Tl_아파트",
        "image_root": raw_root / "원천데이터_231023_add" / "Ts_아파트",
        "prefix": "Ts_아파트_add",
    },
]

MIDDLE_NAMES = {
    "C": "구조물(균열)",
    "P": "구조물(박리, 박락)",
    "X": "구조물(철근 노출)",
    "F": "대지",
    "T": "마감",
    "L": "생활",
    "W": "창호",
}

VALID_CLASS_IDS = {"1", "2", "3"}

# 기대값 (2026-07 실측)
EXPECTED_RGB_JSON_COUNT = 37285


def main():
    problems = []

    for edition in EDITIONS:
        for key in ("label_root", "image_root"):
            if not edition[key].exists():
                problems.append(f"루트 없음: {edition[key]}")

    if problems:
        for problem in problems:
            print("[점검 실패]", problem)
        sys.exit(1)

    rgb_json_count = 0
    thermal_json_count = 0
    missing_image_count = 0
    empty_annotation_count = 0
    mixed_class_count = 0
    invalid_class_count = 0
    shooting_mismatch_count = 0
    schema_error_count = 0
    parse_error_count = 0

    # (middle_id, 대표 class_id) 분포
    distribution = Counter()

    for edition in EDITIONS:
        label_root = edition["label_root"]
        image_root = edition["image_root"]

        json_paths = sorted(label_root.rglob("*.json"))

        for index, json_path in enumerate(json_paths):
            if index % 5000 == 0:
                print(f"  [{edition['name']}] {index}/{len(json_paths)} 스캔 중...")

            # 열화상 폴더는 폴더명으로 1차 배제
            if json_path.parent.name != "RGB":
                thermal_json_count += 1
                continue

            try:
                with json_path.open("r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parse_error_count += 1
                print("  [파싱 실패]", json_path)
                continue

            try:
                raw_data_id = data["Raw_Data_Info"]["Raw_Data_ID"]
                source_info = data["Source_Data_Info"]
                middle_id = source_info["Middle_ID"]
                shooting_id = source_info["Shooting_ID"]
                annotations = data["Learning_Data_Info"]["Annotations"]
            except KeyError as error:
                schema_error_count += 1
                print(f"  [스키마 키 없음 {error}]", json_path)
                continue

            # RGB 폴더인데 Shooting_ID가 R이 아닌 경우 (이중 확인)
            if shooting_id != "R":
                shooting_mismatch_count += 1
                continue

            rgb_json_count += 1

            relative_path = json_path.relative_to(label_root)
            image_path = image_root / relative_path.with_suffix(".jpg")
            if not image_path.exists():
                missing_image_count += 1

            if not annotations:
                empty_annotation_count += 1
                continue

            class_ids = {str(ann["Class_ID"]) for ann in annotations}

            if not class_ids <= VALID_CLASS_IDS:
                invalid_class_count += 1
                continue

            if len(class_ids) > 1:
                mixed_class_count += 1

            representative_class = max(int(value) for value in class_ids)
            distribution[(middle_id, representative_class)] += 1

            _ = raw_data_id  # 스키마 존재 확인용

    print("\n===== 사전 점검 결과 =====")
    print("RGB JSON 수 :", rgb_json_count, f"(기대 {EXPECTED_RGB_JSON_COUNT})")
    print("열화상(비RGB 폴더) JSON 수 :", thermal_json_count)
    print("Shooting_ID 불일치(RGB 폴더인데 R 아님) :", shooting_mismatch_count)
    print("이미지 미매칭 :", missing_image_count)
    print("빈 Annotations :", empty_annotation_count)
    print("Class_ID 혼합(구제 대상) :", mixed_class_count)
    print("유효 외 Class_ID :", invalid_class_count)
    print("스키마 키 누락 :", schema_error_count)
    print("JSON 파싱 실패 :", parse_error_count)

    print("\n===== (middle_id x 대표등급) 분포 =====")
    for middle_id in sorted(MIDDLE_NAMES):
        counts = [distribution.get((middle_id, grade), 0) for grade in (1, 2, 3)]
        print(
            f"{middle_id} {MIDDLE_NAMES[middle_id]:<12}"
            f"\t우수 {counts[0]:>6}\t보통 {counts[1]:>6}\t불량 {counts[2]:>6}"
        )

    unknown_middles = {key[0] for key in distribution} - set(MIDDLE_NAMES)
    if unknown_middles:
        print("[경고] 매핑에 없는 Middle_ID 발견:", unknown_middles)

    failed = (
        rgb_json_count != EXPECTED_RGB_JSON_COUNT
        or missing_image_count > 0
        or schema_error_count > 0
        or parse_error_count > 0
        or bool(unknown_middles)
    )

    if failed:
        print("\n[점검 실패] 기대값과 다릅니다. 위 카운트를 확인하세요.")
        sys.exit(1)

    print("\n사전 점검 통과 - create_metadata_v2.py를 실행해도 됩니다.")
    if mixed_class_count == 0:
        print("(참고: Class_ID 혼합이 0건이므로 v2의 혼합 구제 로직은 no-op으로 동작합니다)")


if __name__ == "__main__":
    main()
