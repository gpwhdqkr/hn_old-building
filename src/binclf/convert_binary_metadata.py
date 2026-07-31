# -*- coding: utf-8 -*-
"""이원화 메타데이터 변환 스크립트 (일회성, 로컬 실행 전용).

data/processed/metadata_split_binary3.xlsx (보통→불량 병합 이원화 정본)를
학습 코드가 읽을 수 있는 metadata_split_binary3.csv (utf-8-sig)로 변환한다.

왜 CSV로 변환하는가
- 학습/평가 코드는 저장소 관례대로 pd.read_csv(encoding="utf-8-sig")만 쓴다.
- xlsx를 직접 읽으려면 openpyxl이 필요한데, 런팟(클라우드)에는 설치하지 않는다.
  변환된 CSV를 git에 커밋해 두면 런팟에서는 git pull만으로 데이터 정의가 도착한다.

주의
- 이 스크립트만 openpyxl이 필요하다 (pip install openpyxl). 학습 코드는 불필요.
- metadata_split_binary2.csv는 한글이 깨진 손상본이므로 절대 사용하지 않는다.

실행 (로컬 1회):
    python src/binclf/convert_binary_metadata.py
데이터가 다른 위치에 있으면:
    (리눅스)     DATA_DIR=/workspace/hn_old-building python src/binclf/convert_binary_metadata.py
    (PowerShell) $env:DATA_DIR="D:/hn_old-building"; python src/binclf/convert_binary_metadata.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parent.parent.parent
data_dir = Path(os.environ.get("DATA_DIR", project_dir))

xlsx_path = data_dir / "data" / "processed" / "metadata_split_binary3.xlsx"
csv_path = data_dir / "data" / "processed" / "metadata_split_binary3.csv"
original_csv_path = data_dir / "data" / "processed" / "metadata_split.csv"

# 검증 기준값 (2026-07 검증 완료된 binary3.xlsx 실측치)
EXPECTED_ROW_COUNT = 37285
EXPECTED_COLUMNS = [
    "image_path", "source_data_id", "class_id",
    "model_label", "class_name", "group_id", "split",
]
EXPECTED_LABEL_COUNTS = {0: 3266, 1: 34019}  # 0=우수, 1=불량(보통 병합)


def fail(message):
    print(f"[검증 실패] {message}")
    sys.exit(1)


def main():
    if not xlsx_path.exists():
        fail(f"원본 xlsx가 없습니다: {xlsx_path}")

    print(f"읽는 중: {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    # --- 검증 1: 행 수 / 컬럼 ---
    if len(df) != EXPECTED_ROW_COUNT:
        fail(f"행 수 불일치: {len(df)} (기대 {EXPECTED_ROW_COUNT})")
    if list(df.columns) != EXPECTED_COLUMNS:
        fail(f"컬럼 불일치: {list(df.columns)}")

    # --- 검증 2: 이원화 라벨 분포 (0=우수, 1=불량) ---
    label_counts = df["model_label"].value_counts().to_dict()
    if label_counts != EXPECTED_LABEL_COUNTS:
        fail(f"model_label 분포 불일치: {label_counts} (기대 {EXPECTED_LABEL_COUNTS})")

    # --- 검증 3: class_id → model_label 매핑 (1→0, 2·3→1) ---
    bad_mapping = df[~(
        ((df["class_id"] == 1) & (df["model_label"] == 0))
        | (df["class_id"].isin([2, 3]) & (df["model_label"] == 1))
    )]
    if len(bad_mapping) > 0:
        fail(f"class_id→model_label 매핑 위반 {len(bad_mapping)}건")

    # --- 검증 4: 경로 한글 파손 여부 (binary2.csv 사고 재발 방지) ---
    if df["image_path"].str.contains(r"\?", regex=True).any():
        fail("image_path에 '?' 문자 발견 — 한글 인코딩이 깨진 파일입니다")

    # --- 검증 5: 원본 metadata_split.csv와 행 단위 정합성 ---
    if original_csv_path.exists():
        original = pd.read_csv(original_csv_path, encoding="utf-8-sig")
        merged = original.merge(
            df, on="source_data_id", suffixes=("_orig", "_bin")
        )
        if len(merged) != EXPECTED_ROW_COUNT:
            fail(f"원본과 source_data_id 매칭 실패: {len(merged)}행")
        if (merged["split_orig"] != merged["split_bin"]).any():
            fail("원본과 split이 다른 행이 있습니다")
        if (merged["image_path_orig"] != merged["image_path_bin"]).any():
            fail("원본과 image_path가 다른 행이 있습니다")
        print("원본 metadata_split.csv와 split/경로 정합성 확인 완료")
    else:
        print(f"[경고] 원본 CSV가 없어 정합성 비교를 건너뜁니다: {original_csv_path}")

    # --- 저장 후 재로드 무결성 확인 ---
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    reloaded = pd.read_csv(csv_path, encoding="utf-8-sig")

    if len(reloaded) != EXPECTED_ROW_COUNT:
        fail("재로드 행 수 불일치")
    if reloaded["image_path"].str.contains(r"\?", regex=True).any():
        fail("재로드한 CSV의 image_path에 '?' 발견 — 인코딩 문제")
    if not (reloaded["class_name"].isin(["우수", "불량"]).all()):
        fail("재로드한 class_name에 우수/불량 외 값 존재 — 인코딩 문제")

    split_counts = reloaded.groupby(["split", "model_label"]).size()
    print("변환 완료:", csv_path)
    print(f"총 {len(reloaded)}행 | 라벨 분포 {label_counts}")
    print("split별 분포:")
    print(split_counts.to_string())
    print("모든 검증 통과 — 이 CSV를 git에 커밋해서 런팟에 배포하세요.")


if __name__ == "__main__":
    main()
