# -*- coding: utf-8 -*-
"""전처리 v2 계층화 그룹 스플릿 스크립트.

metadata_v2.csv를 읽어 (middle_id x 그룹 이원화 라벨) 계층별로
그룹(비디오, group_id) 단위 70/15/15 스플릿을 수행하고
metadata_split_v2.csv (전 컬럼 + split)를 생성한다.

v1(split_data.py)과 달라진 점
- 계층화 키: class_id 단독 → (middle_id x 그룹 이원화 라벨) 14층
- 그룹 내 binary 혼합은 에러가 아니라 경고 (혼합 구제 취지 유지).
  계층화 키에는 그룹 대표 라벨(max model_label)을 사용 - 하나라도 불량이면 불량 층.
- 층 순회·그룹 리스트를 사전순 정렬 후 셔플 (seed 42 재현성 보장)
- v1의 vaildation_count 오타 버그를 재현하지 않고, 층별
  n_train + n_val + n_test == n 어서션으로 봉인

기존 산출물은 절대 덮어쓰지 않는다 - 출력 파일이 이미 있으면 즉시 에러.

실행:
    python src/split_data_v2.py
"""

import os
import random
import sys
from pathlib import Path

import pandas as pd

data_dir = Path(os.environ.get("DATA_DIR", r"D:\hn_old-building"))

metadata_path = data_dir / "data" / "processed" / "metadata_v2.csv"
split_metadata_path = data_dir / "data" / "processed" / "metadata_split_v2.csv"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42


def fail(message):
    print(f"[실패] {message}")
    sys.exit(1)


def split_counts(group_count):
    """층 내 그룹 수 → (train, validation, test) 그룹 수.

    소수층 폴백: n==1은 train만, n==2는 train+test,
    n>=3은 세 스플릿 모두 최소 1 그룹 보장 (초과분은 train에서 차감).
    """
    if group_count == 1:
        counts = (1, 0, 0)
    elif group_count == 2:
        counts = (1, 0, 1)
    else:
        train_count = round(group_count * TRAIN_RATIO)
        validation_count = round(group_count * VALIDATION_RATIO)
        test_count = group_count - train_count - validation_count

        if validation_count < 1:
            validation_count = 1
        if test_count < 1:
            test_count = 1
        train_count = group_count - validation_count - test_count

        if train_count < 1:
            fail(f"층 그룹 수 {group_count}에서 train을 1개도 배정하지 못했습니다")

        counts = (train_count, validation_count, test_count)

    # v1 오타 버그(vaildation_count) 재발 방지: 합이 어긋나면 즉시 중단
    assert sum(counts) == group_count, (counts, group_count)
    return counts


def main():
    if not metadata_path.exists():
        fail(f"metadata_v2.csv가 없습니다 (create_metadata_v2.py 먼저 실행): {metadata_path}")
    if split_metadata_path.exists():
        fail(f"출력 파일이 이미 존재합니다 (수동 삭제 후 재실행): {split_metadata_path}")

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")

    required_columns = {"source_data_id", "group_id", "model_label", "middle_id"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        fail(f"metadata_v2.csv에 필요한 열이 없습니다: {missing_columns}")
    if metadata.empty:
        fail("metadata_v2.csv에 데이터가 없습니다")

    # --- 그룹 테이블: group_id → (middle_id, 그룹 이원화 라벨) ---
    group_middle_nunique = metadata.groupby("group_id")["middle_id"].nunique()
    invalid_groups = group_middle_nunique[group_middle_nunique > 1]
    if not invalid_groups.empty:
        fail(
            "하나의 group_id 안에 서로 다른 middle_id가 들어 있습니다: "
            f"{invalid_groups.index.tolist()[:10]}"
        )

    group_table = metadata.groupby("group_id").agg(
        middle_id=("middle_id", "first"),
        group_binary_label=("model_label", "max"),
        label_nunique=("model_label", "nunique"),
    )

    mixed_binary_groups = int((group_table["label_nunique"] > 1).sum())
    if mixed_binary_groups > 0:
        # 혼합 구제 취지상 에러가 아니라 경고 - 계층화는 그룹 대표 라벨(max)로 수행
        print(
            f"[경고] 그룹 내 이원화 라벨이 섞인 그룹 {mixed_binary_groups}개 - "
            "그룹 대표 라벨(max=불량)로 계층화합니다"
        )

    # --- 계층화 스플릿 (층 순회·그룹 리스트 모두 사전순 고정 → seed 재현성) ---
    random_generator = random.Random(RANDOM_SEED)
    group_split = {}
    stratum_summary = []

    strata = group_table.groupby(["middle_id", "group_binary_label"]).groups
    for stratum_key in sorted(strata.keys()):
        group_ids = sorted(strata[stratum_key])
        random_generator.shuffle(group_ids)

        train_count, validation_count, test_count = split_counts(len(group_ids))

        for group_id in group_ids[:train_count]:
            group_split[group_id] = "train"
        for group_id in group_ids[train_count:train_count + validation_count]:
            group_split[group_id] = "validation"
        for group_id in group_ids[train_count + validation_count:]:
            group_split[group_id] = "test"

        stratum_summary.append(
            (stratum_key, len(group_ids), train_count, validation_count, test_count)
        )

    # --- 프레임 행에 split 매핑 ---
    metadata["split"] = metadata["group_id"].map(group_split)
    if metadata["split"].isna().any():
        fail("split이 지정되지 않은 데이터가 있습니다")

    # --- 그룹 누수 검증 (v1 관례 유지) ---
    split_group_ids = {
        name: set(metadata.loc[metadata["split"] == name, "group_id"])
        for name in ("train", "validation", "test")
    }
    for first, second in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    ):
        overlap = split_group_ids[first] & split_group_ids[second]
        if overlap:
            fail(f"{first}과 {second}에 같은 group_id가 있습니다: {sorted(overlap)[:5]}")

    metadata.to_csv(split_metadata_path, index=False, encoding="utf-8-sig")

    # --- 리포트 ---
    split_count = metadata["split"].value_counts()

    print("데이터 분리 완료")
    print("저장 위치 :", split_metadata_path)

    print("\n===== 전체 이미지 수 =====")
    print("전체 :", len(metadata))
    for name in ("train", "validation", "test"):
        print(f"{name} : {split_count.get(name, 0)}")

    print("\n===== 그룹 수 =====")
    for name in ("train", "validation", "test"):
        print(f"{name} : {len(split_group_ids[name])}")
    print("그룹 중복 없음")

    print("\n===== split x 이원화 클래스 (이미지 수) =====")
    print(
        pd.crosstab(metadata["split"], metadata["class_name"])
        .reindex(["train", "validation", "test"], fill_value=0)
    )

    print("\n===== split x middle_id (이미지 수) =====")
    print(
        pd.crosstab(metadata["split"], metadata["middle_id"])
        .reindex(["train", "validation", "test"], fill_value=0)
    )

    print("\n===== 층별 그룹 배분 (middle_id, 그룹라벨) → 전체/train/val/test =====")
    for stratum_key, total, train_count, validation_count, test_count in stratum_summary:
        print(
            f"{stratum_key} : {total:>4}그룹 → "
            f"train {train_count:>3} / val {validation_count:>3} / test {test_count:>3}"
            f"  (train 비율 {train_count / total:.2f})"
        )


if __name__ == "__main__":
    main()
