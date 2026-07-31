# -*- coding: utf-8 -*-
"""전처리 v3 메타데이터 생성: metadata_split_v2.csv에서 middle_id 제거.

v3 파이프라인(src/v3/ — 이원화 단일 과제)은 middle_id(결함 종류 7종)를
전혀 사용하지 않으므로, CSV에서 middle_id/middle_name 컬럼을 제거한
정본을 새로 만든다.

중요: split 배정(train/validation/test)은 v2와 **행 단위로 동일하게 보존**한다.
재분할하지 않는 이유 — v2 split은 비디오 그룹(Raw_Data_ID) 단위로 누수를
막고 (이원화 클래스 × middle_id) 계층화까지 검증된 상태라, 컬럼만 빼고
배정을 유지하면 그 검증을 그대로 물려받는다. (계층화에 middle_id가 쓰였다는
사실은 배정 결과의 품질에만 기여했을 뿐, 소비 시점에는 컬럼이 필요 없다.)

산출물 (기존 산출물은 절대 덮어쓰지 않는다 - 이미 있으면 즉시 에러):
  data/processed/metadata_split_v3.csv  (utf-8-sig)
  컬럼: image_path, image_relpath, source_data_id, group_id, class_id,
        model_label, class_name, n_annotations, n_defect_annotations,
        is_mixed, split   (v2에서 middle_id, middle_name만 제거)

실행:
    python src/make_metadata_split_v3.py
경로가 다르면:
    (PowerShell) $env:DATA_DIR="D:/hn_old-building"; python src/make_metadata_split_v3.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

data_dir = Path(os.environ.get("DATA_DIR", r"D:\hn_old-building"))

source_path = data_dir / "data" / "processed" / "metadata_split_v2.csv"
output_path = data_dir / "data" / "processed" / "metadata_split_v3.csv"

DROP_COLUMNS = ["middle_id", "middle_name"]


def main():
    if output_path.exists():
        sys.exit(f"[에러] 산출물이 이미 존재합니다 (덮어쓰기 금지): {output_path}")
    if not source_path.exists():
        sys.exit(f"[에러] 입력이 없습니다: {source_path}")

    metadata = pd.read_csv(source_path, encoding="utf-8-sig")

    missing = [column for column in DROP_COLUMNS if column not in metadata.columns]
    if missing:
        sys.exit(f"[에러] v2 CSV에 기대한 컬럼이 없습니다: {missing}")

    trimmed = metadata.drop(columns=DROP_COLUMNS)

    # 검증: 컬럼 제거 외에는 아무것도 변하지 않았는지 (행 수/순서/값)
    if len(trimmed) != len(metadata):
        sys.exit("[에러] 행 수가 변했습니다 — 컬럼 제거만 해야 합니다")
    if not trimmed.equals(metadata[trimmed.columns]):
        sys.exit("[에러] 잔여 컬럼 값이 변했습니다 — 컬럼 제거만 해야 합니다")

    trimmed.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {output_path}")
    print(f"행 수: {len(trimmed)} (v2와 동일)")
    print(f"제거한 컬럼: {DROP_COLUMNS}")
    print(f"남은 컬럼: {list(trimmed.columns)}")
    print("\nsplit x model_label (이미지 수) - v2와 동일해야 함:")
    print(pd.crosstab(trimmed["split"], trimmed["model_label"]))


if __name__ == "__main__":
    main()
