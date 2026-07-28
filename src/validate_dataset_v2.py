# -*- coding: utf-8 -*-
"""전처리 v2 최종 검증·리포트 스크립트.

metadata_split_v2.csv / annotations_v2.jsonl을 종합 검증하고
결과를 data/processed/preprocess_v2_report.txt로 저장한다 (콘솔에도 동일 출력).

검증 항목
1. 행 수 대사: CSV 행 수 == JSONL 고유 source_data_id 수, sum(n_annotations) == JSONL 행 수
2. 기존(v1) metadata.csv 교차 대조: v2 >= v1, 차집합 = 혼합 구제분,
   공통 행의 class_id/group_id/image_path 일치
3. 라벨 무결성: class_id<->model_label 이원화 규칙, binclf 기준 분포 {0: 3266, 1: 34019} 대조
4. 경로 무결성: '?' 문자 0건(한글 인코딩 파손 검출), "raw/images/" 마커 정확히 1회,
   processed_images 리사이즈본 실존 100%, 원본(D:\\189...) 실존 100%
5. 스플릿 무결성: 그룹 교집합 0, 층별 비율, v1 split과 일치율(참고용)
6. utf-8-sig 재로드 무결성

주의: v1 metadata_split.csv와 split 배정이 다른 것은 정상이다 (계층화 방식이 다름).
v2 CSV를 v1 산출물과 혼용하지 말 것.

실행:
    python src/validate_dataset_v2.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

raw_root = Path(os.environ.get(
    "RAW_ROOT",
    r"D:\189.서울시 노후 주택 균열 데이터\01.데이터\1.Training"
))
data_dir = Path(os.environ.get("DATA_DIR", r"D:\hn_old-building"))

processed_dir = data_dir / "data" / "processed"
split_v2_path = processed_dir / "metadata_split_v2.csv"
annotations_path = processed_dir / "annotations_v2.jsonl"
v1_metadata_path = processed_dir / "metadata.csv"
v1_split_path = processed_dir / "metadata_split.csv"
report_path = processed_dir / "preprocess_v2_report.txt"
processed_images_dir = data_dir / "data" / "processed_images"

PREFIX_TO_IMAGE_ROOT = {
    "TS_아파트": raw_root / "원천데이터" / "TS_아파트",
    "Ts_아파트_add": raw_root / "원천데이터_231023_add" / "Ts_아파트",
}

# binclf 기준값 (구제분이 있으면 그만큼 차이가 나는 것이 정상)
BINCLF_LABEL_COUNTS = {0: 3266, 1: 34019}

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
    if not split_v2_path.exists():
        print(f"[실패] metadata_split_v2.csv가 없습니다: {split_v2_path}")
        sys.exit(1)
    if not annotations_path.exists():
        print(f"[실패] annotations_v2.jsonl이 없습니다: {annotations_path}")
        sys.exit(1)
    if report_path.exists():
        print(f"[실패] 리포트 파일이 이미 존재합니다 (수동 삭제 후 재실행): {report_path}")
        sys.exit(1)

    emit("========== 전처리 v2 최종 검증 리포트 ==========")
    emit(f"대상: {split_v2_path}")
    emit(f"      {annotations_path}")
    emit()

    df = pd.read_csv(split_v2_path, encoding="utf-8-sig")

    # ---------- 1. 행 수 대사 ----------
    emit("----- 1. 행 수 대사 -----")
    annotation_count = 0
    annotation_ids = set()
    annotation_class_ids = set()
    with annotations_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            record = json.loads(line)
            annotation_count += 1
            annotation_ids.add(record["source_data_id"])
            annotation_class_ids.add(record["class_id"])

    emit(f"CSV 행 수: {len(df)}")
    emit(f"JSONL 행 수: {annotation_count} / 고유 source_data_id: {len(annotation_ids)}")
    check(df["source_data_id"].is_unique, "CSV source_data_id 중복 없음")
    check(
        set(df["source_data_id"]) == annotation_ids,
        "CSV와 JSONL의 source_data_id 집합 일치",
    )
    check(
        int(df["n_annotations"].sum()) == annotation_count,
        f"sum(n_annotations)={int(df['n_annotations'].sum())} == JSONL 행 수={annotation_count}",
    )
    check(
        annotation_class_ids <= {"1", "2", "3"},
        f"JSONL class_id 값 유효 ({sorted(annotation_class_ids)})",
    )

    # ---------- 2. v1 교차 대조 ----------
    emit()
    emit("----- 2. 기존(v1) metadata.csv 교차 대조 -----")
    if v1_metadata_path.exists():
        v1 = pd.read_csv(v1_metadata_path, encoding="utf-8-sig")
        v1_ids = set(v1["source_data_id"])
        v2_ids = set(df["source_data_id"])

        check(v1_ids <= v2_ids, f"v2({len(v2_ids)}) >= v1({len(v1_ids)})")
        rescued = sorted(v2_ids - v1_ids)
        emit(f"v2 - v1 차집합(혼합 구제분): {len(rescued)}건"
             + (f" - 예: {rescued[:5]}" if rescued else " (0건이면 구제 no-op으로 정상)"))

        merged = v1.merge(df, on="source_data_id", suffixes=("_v1", "_v2"))
        check(len(merged) == len(v1_ids), "공통 행 merge 수 == v1 행 수")
        check(
            int((merged["class_id_v1"] != merged["class_id_v2"]).sum()) == 0,
            "공통 행 class_id 일치 (v1 단일값 == v2 max)",
        )
        check(
            int((merged["group_id_v1"] != merged["group_id_v2"]).sum()) == 0,
            "공통 행 group_id 일치 (파일명 파생 == Raw_Data_ID)",
        )
        check(
            int((merged["image_path_v1"] != merged["image_path_v2"]).sum()) == 0,
            "공통 행 image_path 문자열 완전 일치 (기존 소비 코드 호환)",
        )
    else:
        emit(f"[경고] v1 metadata.csv가 없어 교차 대조를 건너뜁니다: {v1_metadata_path}")

    # ---------- 3. 라벨 무결성 ----------
    emit()
    emit("----- 3. 라벨 무결성 (이원화) -----")
    bad_mapping = df[~(
        ((df["class_id"] == 1) & (df["model_label"] == 0))
        | (df["class_id"].isin([2, 3]) & (df["model_label"] == 1))
    )]
    check(len(bad_mapping) == 0, f"class_id<->model_label 이원화 규칙 위반 {len(bad_mapping)}건")
    check(
        set(df["class_name"]) <= {"우수", "불량"},
        f"class_name 값 유효 ({sorted(set(df['class_name']))})",
    )

    label_counts = df["model_label"].value_counts().to_dict()
    rescued_count = len(set(df["source_data_id"]) - v1_ids) if v1_metadata_path.exists() else 0
    emit(f"이원화 분포: {label_counts} (binclf 기준 {BINCLF_LABEL_COUNTS}, 구제분 {rescued_count}건 차이 허용)")
    check(
        label_counts.get(0, 0) + label_counts.get(1, 0)
        == sum(BINCLF_LABEL_COUNTS.values()) + rescued_count,
        "총 행 수 == binclf 기준 합 + 구제분",
    )
    if rescued_count == 0:
        check(label_counts == BINCLF_LABEL_COUNTS, "이원화 분포가 binclf 기준값과 정확히 일치")

    mixed_count = int(df["is_mixed"].sum())
    emit(f"is_mixed=1 행 수: {mixed_count}")
    check(
        int(((df["n_defect_annotations"] > 0) != (df["model_label"] == 1)).sum()) == 0,
        "n_defect_annotations>0 <=> model_label==1",
    )

    # ---------- 4. 경로 무결성 ----------
    emit()
    emit("----- 4. 경로 무결성 -----")
    check(
        not df["image_path"].str.contains(r"\?", regex=True).any(),
        "image_path에 '?' 없음 (한글 인코딩 파손 검출)",
    )
    check(
        (df["image_path"].str.count("raw/images/") == 1).all(),
        "image_path에 'raw/images/' 마커 정확히 1회",
    )
    check(
        (df["image_path"] == "D:/hn_old-building_raw/raw/images/" + df["image_relpath"]).all(),
        "image_path == 가상 접두 + image_relpath 정합",
    )

    print("processed_images 실존 확인 중 (수십 초 소요)...")
    missing_resized = [
        relpath for relpath in df["image_relpath"]
        if not (processed_images_dir / relpath).exists()
    ]
    check(len(missing_resized) == 0, f"processed_images 리사이즈본 실존 (누락 {len(missing_resized)}건)")

    print("원본 이미지 실존 확인 중 (수십 초 소요)...")
    missing_original = []
    for relpath in df["image_relpath"]:
        prefix, _, rest = relpath.partition("/")
        image_root = PREFIX_TO_IMAGE_ROOT.get(prefix)
        if image_root is None or not (image_root / rest).exists():
            missing_original.append(relpath)
    check(len(missing_original) == 0, f"원본 이미지 실존 (누락 {len(missing_original)}건)")

    # ---------- 5. 스플릿 무결성 ----------
    emit()
    emit("----- 5. 스플릿 무결성 -----")
    check(
        set(df["split"]) == {"train", "validation", "test"},
        f"split 값 유효 ({sorted(set(df['split']))})",
    )

    split_groups = {
        name: set(df.loc[df["split"] == name, "group_id"])
        for name in ("train", "validation", "test")
    }
    overlap_total = (
        len(split_groups["train"] & split_groups["validation"])
        + len(split_groups["train"] & split_groups["test"])
        + len(split_groups["validation"] & split_groups["test"])
    )
    check(overlap_total == 0, "train/validation/test 그룹 교집합 0 (누수 없음)")

    emit()
    emit("split별 이미지 수 / 그룹 수:")
    for name in ("train", "validation", "test"):
        image_count = int((df["split"] == name).sum())
        emit(f"  {name:<11}: {image_count:>6}장 / {len(split_groups[name]):>4}그룹"
             f" ({image_count / len(df):.3f})")

    emit()
    emit("split x class_name (이미지 수):")
    emit(pd.crosstab(df["split"], df["class_name"])
         .reindex(["train", "validation", "test"], fill_value=0).to_string())
    emit()
    emit("split x middle_id (이미지 수):")
    emit(pd.crosstab(df["split"], df["middle_id"])
         .reindex(["train", "validation", "test"], fill_value=0).to_string())

    if v1_split_path.exists():
        v1_split = pd.read_csv(v1_split_path, encoding="utf-8-sig")
        merged_split = v1_split[["source_data_id", "split"]].merge(
            df[["source_data_id", "split"]], on="source_data_id",
            suffixes=("_v1", "_v2"),
        )
        agreement = float((merged_split["split_v1"] == merged_split["split_v2"]).mean())
        emit()
        emit(f"[참고] v1 metadata_split.csv와 split 일치율: {agreement:.1%} "
             "- 계층화 방식이 달라 불일치가 정상. v1 산출물과 혼용 금지.")

    # ---------- 6. 재로드 무결성 ----------
    emit()
    emit("----- 6. utf-8-sig 재로드 무결성 -----")
    reloaded = pd.read_csv(split_v2_path, encoding="utf-8-sig")
    check(len(reloaded) == len(df), "재로드 행 수 일치")
    check(
        reloaded["class_name"].isin(["우수", "불량"]).all()
        and reloaded["middle_name"].str.contains(r"\?", regex=True).sum() == 0,
        "재로드 한글 무결 (class_name/middle_name)",
    )

    # ---------- 결과 ----------
    emit()
    if failures:
        emit(f"========== 검증 실패 {len(failures)}건 ==========")
        for message in failures:
            emit(f"  - {message}")
    else:
        emit("========== 전 항목 통과 ==========")

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8-sig")
    print(f"\n리포트 저장: {report_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
