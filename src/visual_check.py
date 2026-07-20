import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

#한글
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

project_dir = Path(__file__).resolve().parent.parent
raw_dir = Path(r"D:\hn_old-building_raw\raw")

image_dir = raw_dir / "images"
labels_dir = raw_dir / "labels"

class_names = {
    "1": "우수",
    "2": "보통",
    "3": "불량"
}

#등급별 이미지 경로 저장 변수
images_by_class = defaultdict(list)

#등급마다 몇장을 볼 것 인지
sample_count = 5

random.seed(42)

json_files = list(labels_dir.rglob("*.json"))

random.shuffle(json_files)

image_paths = [
    image_path
    for image_path in image_dir.rglob("*")
    if image_path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png"
    }
]

image_by_name = {}

for image_path in image_paths:
    image_by_name[image_path.stem] = image_path

for json_path in json_files:
    with json_path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    # json의 원천데이터 정보와 학습정보 꺼내기
    source_info = data["Source_Data_Info"]
    learning_info = data["Learning_Data_Info"]

    shooting_id = source_info["Shooting_ID"]
    source_data_id = source_info["Source_Data_ID"]
    annotations = learning_info["Annotations"]

    if shooting_id != "R":
        continue

    if not annotations:
        continue

    class_ids = set()

    for annotation in annotations:
        class_id = str(annotation["Class_ID"])
        class_ids.add(class_id)

    if len(class_ids) != 1:
        continue

    class_id = class_ids.pop()

    if class_id not in class_names:
        continue

    image_path = image_by_name.get(source_data_id)

    if image_path is None:
        continue

    images_by_class[class_id].append(image_path)

    if all(
    len(images_by_class[class_id]) >= sample_count
    for class_id in class_names
    ):
        break

figure, axes = plt.subplots(
    3,
    sample_count,
    figsize=(18,10)
)

#우수, 보통, 불량을 순서대로 반복
for row_index, class_id in enumerate(["1","2","3"]):
    class_name = class_names[class_id]
    image_paths = images_by_class[class_id]

    #해당긍급에서 무작위 5장 선택
    selected_images = random.sample(
        image_paths,
        min(sample_count, len(image_paths))
    )

    #선택된 이미지를 한 장씩 출력
    for column_index in range(sample_count):

        axis = axes[row_index][column_index]
        #선택된 이미지가 있는 위치만 출력
        if column_index < len(selected_images):
            image_path = selected_images[column_index]

            #이미지 파일 열기
            image = Image.open(image_path)

            #이미지를 화면에 표시
            axis.imshow(image)

            #이미지 위에 등급과 파일면을 표시
            axis.set_title(
                f"{class_name} (Class_ID={class_id})\n"
                f"{image_path.name}",
                fontsize=8
            )
            print(
                f"{class_name}: {image_path.name}"
            )

        axis.axis("off")
plt.tight_layout()
plt.show()