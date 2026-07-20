import random
from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parent.parent

metadata_path = project_dir / "data" / "processed" / "metadata.csv"
#분리결과를 담을 새 csv
split_metadata_path = project_dir / "data" / "processed" / "metadata_split.csv"

#metadata.csv를 읽어서 표형태로 가져옴
metadata = pd.read_csv(
    metadata_path,
    encoding="utf-8-sig"
)

#분리에 필요한 열이름
required_columns = {
    "image_path",
    "class_id",
    "class_name",
    "group_id"
}
# 실제 들어있는 열 이름
actual_columns = set(metadata.columns)

#필요한 열중 csv에 없는 열을 찾는다
missing_columns = required_columns - actual_columns

# 필요한 열 없으면 프로그램 중단
if missing_columns:
    raise ValueError(
        f"metadata.csv에 필요한 열이 없습니다: {missing_columns}"
    )

if metadata.empty:
    raise ValueError("metdata.csv에 데이터가 없습니다")

#group_id안에 다른등급이 섞였는지 검사
group_class_count = (
    metadata
    .groupby("group_id")["class_id"]
    .nunique()
)

#등급이 두 개 이상 섞여 있는 그룹을 찾음
invalid_groups = group_class_count[
    group_class_count > 1
]

if not invalid_groups.empty:
    raise ValueError(
        "하나의 group_id 안에 서로 다른 Class_ID가 들어 있습니다"
    )

random_generator = random.Random(42)

#각 group_id가 어느 데이터에 들어갈지 저장
group_split = {}

for class_id, class_data in metadata.groupby("class_id"):

    #현재 등급에 속한 group_id를 중복 없이 가져옴
    group_ids = (
        class_data["group_id"]
        .drop_duplicates()
        .tolist()
    )

    #group_id순서 무작위로 섞기
    random_generator.shuffle(group_ids)

    #현재 등급의 전체 그룹 수
    group_count = len(group_ids)

    train_count = int(group_count * 0.70)
    validation_count = int(group_count * 0.15)

    #그룹이 3개 이상이면 각각 최소 한 그룹씩 배치
    if group_count >= 3:
        train_count = max(1, train_count)
        validation_count = max(1, validation_count)

        #test에도 최소 한 그룹이 남도록 조정
        if train_count + validation_count >= group_count:
            train_count = group_count - 2
            vaildation_count = 1
    #학습그룹
    train_groups = group_ids[:train_count]

    #검증그룹
    validation_groups = group_ids[
        train_count:
        train_count + validation_count
    ]

    #테스트그룹
    test_group = group_ids[
        train_count + validation_count:
    ]

    for group_id in train_groups:
        group_split[group_id] = "train"
    
    for group_id in validation_groups:
        group_split[group_id] = "validation"
    
    for group_id in test_group:
        group_split[group_id] = "test"

metadata["split"] = metadata["group_id"].map(
    group_split
)

if metadata["split"].isna().any():
    raise ValueError("split이 지정되지 않은 데이터가 있습니다.")

train_group_ids = set(
    metadata.loc[
        metadata["split"] == "train",
        "group_id"
    ]
)

validation_group_ids = set(
    metadata.loc[
        metadata["split"] == "validation",
        "group_id"
    ]
)

test_group_ids = set(
    metadata.loc[
        metadata["split"] == "test",
        "group_id"
    ]
)

#같은 그룹이 서로 다른 데이터에 겹치는지 확인
if train_group_ids & validation_group_ids:
    raise ValueError(
        "train과 validation에 같은 group_id가 있습니다"
    )

if train_group_ids & test_group_ids:
    raise ValueError(
        "train과 test에 같은 group_id가 있습니다"
    )

if validation_group_ids & test_group_ids:
    raise ValueError(
        "validation과 test에 같은 group_id가 있습니다"
    )

# split 열이 추가된 csv를 새로 저장한다.
metadata.to_csv(
    split_metadata_path,
    index=False,
    encoding="utf-8-sig"
)
    
split_count = metadata["split"].value_counts()

class_split_count = pd.crosstab(
    metadata["split"],
    metadata["class_name"]
)

class_split_count = class_split_count.reindex(
    ["train", "validation", "test"],
    fill_value=0
)

print("데이터 분리 완료")
print("저장 위치: ", split_metadata_path)

print("\n===== 전체 이미지 수 =====")
print("전체 : ", len(metadata))
print("Train : ", split_count.get("train",0))
print(
    "validation : ",
    split_count.get("validation", 0)
)
print("Test : ", split_count.get("test",0))

print("\n===== 등급 분리 =====")
print(class_split_count)

print("\n===== 그룹 중복 =====")
print("Train 그룹 : ", len(train_group_ids))
print(
    "Validation 그룹",
    len(validation_group_ids)
)
print("Test 그룹 : ", len(test_group_ids))
print("그룹 중복 없음")