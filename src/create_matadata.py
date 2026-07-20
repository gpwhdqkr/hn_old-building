import csv
import json
from collections import Counter
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent

image_dir = project_dir / "data" / "raw" / "images"
labels_dir = project_dir / "data" / "raw" / "labels"

processed_dir = project_dir / "data" / "processed"
metadata_path = processed_dir / "metadata.csv"

processed_dir.mkdir(parents=True, exist_ok=True)

class_name = {
    "1": "우수",
    "2": "보통",
    "3": "불량"
}

model_labels = {
    "1": 0,
    "2": 1,
    "3": 2
}

metadata_rows = []

json_files = sorted(labels_dir.glob("*.json"))

for json_path in json_files:
    with json_path.open(
        "r",
        encoding = "utf-8"
    ) as json_file:
        
        data = json.load(json_file)
    
    source_info = data["Source_Data_Info"]
    learning_info = data["Learning_Data_Info"]

    shooting_id = source_info["Shooting_ID"]
    source_data_id = source_info["Source_Data_ID"]

    annotations = learning_info["Annotations"]

    if shooting_id != "R":
        continue

    if not annotations:
        continue
        #중복값을 저장하지않음
    class_ids = set()

    for annotation in annotations:
        class_id_value = str(annotation["Class_ID"])
        class_ids.add(class_id_value)
    
    if len(class_ids) != 1:
        continue

    class_id = class_ids.pop()

    if class_id not in class_name:
        continue

    image_path = image_dir / f"{source_data_id}.jpg"

    if not image_path.exists():
        continue

    relative_image_path = image_path.relative_to(
        project_dir
    ).as_posix()

    group_id = source_data_id.rsplit("-", 1)[0]

    metadata_rows.append({
        "image_path": relative_image_path,
        "source_data_id": source_data_id,
        "class_id": class_id,
        "model_label": model_labels[class_id],
        "class_name": class_name[class_id],
        "group_id": group_id
    })

field_names = [
    "image_path",
    "source_data_id",
    "class_id",
    "model_label",
    "class_name",
    "group_id"
]

#csv파일 새로 만들기
with metadata_path.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:
    
    writer = csv.DictWriter(
        csv_file,
        fieldnames=field_names
    )

    #첫줄에 열이름 기록
    writer.writeheader()

    #metadata_rows내용 csv에 기록
    writer.writerows(metadata_rows)

class_count = Counter(
    row["class_name"]
    for row in metadata_rows
)

print("metadata.csv 생성 완료")
print("저장 위치 : ", metadata_path)
print("전체 데이터 수 : ", len(metadata_rows))

print("우수 : ", class_count["우수"])
print("보통 : ", class_count["보통"])
print("불량 : ", class_count["불량"])
