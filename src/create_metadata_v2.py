# -*- coding: utf-8 -*-
"""전처리 v2 메타데이터 생성 스크립트.

원천 라벨 JSON(D:\\189...)을 단일 스캔으로 훑어 두 산출물을 동시에 생성한다.
  1) data/processed/metadata_v2.csv      - 이미지 1장 = 1행 (utf-8-sig)
  2) data/processed/annotations_v2.jsonl - annotation 1개 = 1행 (BOM 없는 utf-8)

v1(create_matadata.py)과 달라진 점
- 라벨 이원화: model_label 0=우수(모든 annotation이 Class_ID 1), 1=불량(결함 annotation이 하나라도 있으면)
- Class_ID 혼합 JSON을 버리지 않고 구제 (대표 class_id = max 등급, is_mixed 플래그)
- middle_id(결함 종류 7종) / middle_name 컬럼 추가
- group_id를 파일명 파싱이 아니라 Raw_Data_Info.Raw_Data_ID에서 직접 읽음
- 폴리곤을 annotations_v2.jsonl로 보존 (평탄화 [x1,y1,x2,y2,...] 원본 그대로, 1440x1080 기준)
- 이미지 매칭을 rglob stem 딕셔너리 대신 라벨<->원천 경로 미러링 치환으로 결정적으로 수행

image_path는 기존 CSV와 동일한 가상 포맷(D:/hn_old-building_raw/raw/images/...)으로
재구성하므로, "raw/images/" 마커로 잘라 쓰는 기존 소비 코드가 무수정으로 읽을 수 있다.
새 코드는 image_relpath(processed_images 기준 POSIX 상대경로)를 쓰면 된다.

기존 산출물은 절대 덮어쓰지 않는다 - 출력 파일이 이미 있으면 즉시 에러.

실행:
    python src/create_metadata_v2.py
경로가 다르면:
    (PowerShell) $env:DATA_DIR="D:/hn_old-building"; $env:RAW_ROOT="..."; python src/create_metadata_v2.py
"""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

raw_root = Path(os.environ.get(
    "RAW_ROOT",
    r"D:\189.서울시 노후 주택 균열 데이터\01.데이터\1.Training"
))

# 산출물은 워크트리가 아니라 메인 리포 데이터 폴더에 저장한다.
# (Path(__file__).parent.parent 패턴을 쓰면 워크트리의 data/를 가리키므로 금지)
data_dir = Path(os.environ.get("DATA_DIR", r"D:\hn_old-building"))

metadata_path = data_dir / "data" / "processed" / "metadata_v2.csv"
annotations_path = data_dir / "data" / "processed" / "annotations_v2.jsonl"

# 기존 CSV와 호환되는 가상 경로 접두 (소비 코드가 "raw/images/" 마커로 자름)
VIRTUAL_PREFIX = "D:/hn_old-building_raw/raw/images"

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

# 이원화 라벨 (binclf 관례)
BINARY_CLASS_NAMES = {0: "우수", 1: "불량"}

FIELD_NAMES = [
    "image_path",
    "image_relpath",
    "source_data_id",
    "group_id",
    "class_id",
    "model_label",
    "class_name",
    "middle_id",
    "middle_name",
    "n_annotations",
    "n_defect_annotations",
    "is_mixed",
]


def fail(message):
    print(f"[실패] {message}")
    sys.exit(1)


def main():
    # 기존 파일 덮어쓰기 금지
    for output_path in (metadata_path, annotations_path):
        if output_path.exists():
            fail(f"출력 파일이 이미 존재합니다 (수동 삭제 후 재실행): {output_path}")

    if not metadata_path.parent.exists():
        fail(f"출력 폴더가 없습니다: {metadata_path.parent}")

    for edition in EDITIONS:
        if not edition["label_root"].exists():
            fail(f"라벨 루트가 없습니다: {edition['label_root']}")
        if not edition["image_root"].exists():
            fail(f"원천 루트가 없습니다: {edition['image_root']}")

    metadata_rows = []
    annotation_records = []

    skip_counts = Counter()
    # 방어 검증 카운터 (0이어야 정상인 항목들)
    defensive_counts = Counter()

    seen_source_ids = {}

    # 폴더별 {stem: 실제 파일명} 캐시.
    # Windows는 exists()가 대소문자를 무시하므로, 디스크상의 실제 파일명(.JPG 등)을
    # 그대로 써야 리눅스(런팟)에서도 경로가 깨지지 않고 v1 image_path와도 일치한다.
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
    directory_cache = {}

    def actual_image_name(image_dir, stem):
        if image_dir not in directory_cache:
            listing = {}
            if image_dir.exists():
                for entry in image_dir.iterdir():
                    if entry.suffix.lower() in IMAGE_SUFFIXES:
                        listing[entry.stem] = entry.name
            directory_cache[image_dir] = listing
        return directory_cache[image_dir].get(stem)

    for edition in EDITIONS:
        label_root = edition["label_root"]
        image_root = edition["image_root"]
        prefix = edition["prefix"]

        # 처리 순서를 경로 사전순으로 고정 (산출물 재현성)
        json_paths = sorted(label_root.rglob("*.json"))
        print(f"[{edition['name']}] JSON {len(json_paths)}개 스캔 시작")

        for index, json_path in enumerate(json_paths):
            if index % 5000 == 0 and index > 0:
                print(f"  {index}/{len(json_paths)} 처리 중...")

            # 열화상 제외 1차: 폴더명
            if json_path.parent.name != "RGB":
                skip_counts["열화상(폴더)"] += 1
                continue

            with json_path.open("r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            raw_data_id = data["Raw_Data_Info"]["Raw_Data_ID"]
            source_info = data["Source_Data_Info"]
            annotations = data["Learning_Data_Info"]["Annotations"]

            source_data_id = source_info["Source_Data_ID"]
            middle_id = source_info["Middle_ID"]
            shooting_id = source_info["Shooting_ID"]

            # 열화상 제외 2차: JSON 필드 (폴더와 불일치하면 카운트만 하고 제외)
            if shooting_id != "R":
                skip_counts["열화상(Shooting_ID)"] += 1
                defensive_counts["RGB폴더인데 Shooting_ID!=R"] += 1
                continue

            if not annotations:
                skip_counts["빈 Annotations"] += 1
                continue

            class_ids = {str(ann["Class_ID"]) for ann in annotations}
            if not class_ids <= VALID_CLASS_IDS:
                skip_counts["유효 외 Class_ID"] += 1
                continue

            # 라벨<->원천 경로 미러링으로 결정적 1:1 매칭
            # (파일명은 디스크상의 실제 표기를 사용 - 확장자 대소문자 보존)
            relative_path = json_path.relative_to(label_root)
            image_name = actual_image_name(
                image_root / relative_path.parent, json_path.stem
            )
            if image_name is None:
                skip_counts["이미지 미매칭"] += 1
                continue

            # --- 방어 검증 (행은 살리되 카운트) ---
            if json_path.stem != source_data_id:
                defensive_counts["파일명 stem != Source_Data_ID"] += 1
            if source_data_id.rsplit("-", 1)[0] != raw_data_id:
                defensive_counts["파일명 파생 group != Raw_Data_ID"] += 1
            if middle_id not in MIDDLE_NAMES:
                fail(f"매핑에 없는 Middle_ID '{middle_id}': {json_path}")
            if relative_path.parts[0] != MIDDLE_NAMES[middle_id]:
                defensive_counts["카테고리 폴더명 != middle_name"] += 1

            # 본편/add편 간 동일 source_data_id 중복은 hard-fail
            if source_data_id in seen_source_ids:
                fail(
                    f"source_data_id 중복: {source_data_id}\n"
                    f"  기존: {seen_source_ids[source_data_id]}\n"
                    f"  신규: {json_path}"
                )
            seen_source_ids[source_data_id] = json_path

            # --- 라벨 산출 (이원화) ---
            representative_class = max(int(value) for value in class_ids)
            model_label = 0 if class_ids == {"1"} else 1
            n_defect = sum(
                1 for ann in annotations if str(ann["Class_ID"]) in {"2", "3"}
            )
            is_mixed = len(class_ids) > 1
            if is_mixed:
                defensive_counts["Class_ID 혼합 구제"] += 1

            relative_jpg = (relative_path.parent / image_name).as_posix()

            metadata_rows.append({
                "image_path": f"{VIRTUAL_PREFIX}/{prefix}/{relative_jpg}",
                "image_relpath": f"{prefix}/{relative_jpg}",
                "source_data_id": source_data_id,
                "group_id": raw_data_id,
                "class_id": representative_class,
                "model_label": model_label,
                "class_name": BINARY_CLASS_NAMES[model_label],
                "middle_id": middle_id,
                "middle_name": MIDDLE_NAMES[middle_id],
                "n_annotations": len(annotations),
                "n_defect_annotations": n_defect,
                "is_mixed": int(is_mixed),
            })

            for ann_index, annotation in enumerate(annotations):
                annotation_records.append({
                    "source_data_id": source_data_id,
                    "ann_index": ann_index,
                    "class_id": str(annotation["Class_ID"]),
                    "type": annotation.get("Type", ""),
                    "polygon": annotation.get("polygon", []),
                })

    if not metadata_rows:
        fail("생성된 행이 없습니다 - 원천 경로를 확인하세요.")

    # --- 저장 ---
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows(metadata_rows)

    # JSONL은 BOM 없는 utf-8 (json.loads가 BOM을 못 읽으므로)
    with annotations_path.open("w", encoding="utf-8", newline="\n") as jsonl_file:
        for record in annotation_records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False))
            jsonl_file.write("\n")

    # --- 자체 검증 ---
    if len(seen_source_ids) != len(metadata_rows):
        fail("source_data_id 수와 행 수가 다릅니다 (중복 검출 로직 오류)")

    total_annotations = sum(row["n_annotations"] for row in metadata_rows)
    if total_annotations != len(annotation_records):
        fail(
            f"n_annotations 합({total_annotations})과 "
            f"JSONL 행 수({len(annotation_records)})가 다릅니다"
        )

    label_counts = Counter(row["model_label"] for row in metadata_rows)
    middle_counts = Counter(row["middle_id"] for row in metadata_rows)

    print("\n===== 생성 완료 =====")
    print("metadata_v2.csv :", metadata_path)
    print("annotations_v2.jsonl :", annotations_path)
    print("이미지 행 수 :", len(metadata_rows))
    print("annotation 행 수 :", len(annotation_records))
    print("그룹 수 :", len({row["group_id"] for row in metadata_rows}))

    print("\n----- 스킵 카운트 -----")
    for key, value in sorted(skip_counts.items()):
        print(f"{key} : {value}")

    print("\n----- 방어 검증 카운트 (0이면 정상) -----")
    if defensive_counts:
        for key, value in sorted(defensive_counts.items()):
            print(f"{key} : {value}")
    else:
        print("전 항목 0 - 이상 없음")
    rescued = defensive_counts.get("Class_ID 혼합 구제", 0)
    print(f"(혼합 구제 {rescued}건 - 0건이면 구제 로직이 no-op으로 동작한 것으로 정상)")

    print("\n----- 이원화 라벨 분포 -----")
    print("우수(0) :", label_counts.get(0, 0))
    print("불량(1) :", label_counts.get(1, 0))

    print("\n----- middle_id 분포 -----")
    for middle_id in sorted(MIDDLE_NAMES):
        print(
            f"{middle_id} {MIDDLE_NAMES[middle_id]:<12}: "
            f"{middle_counts.get(middle_id, 0)}"
        )


if __name__ == "__main__":
    main()
