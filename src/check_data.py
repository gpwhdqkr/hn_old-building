#파이썬 내장 모듈의 클래스
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent

raw_dir = Path(r"D:\hn_old-building_raw\raw")

image_dir = raw_dir / "images"
labels_dir = raw_dir / "labels"

class_names = {
    "1": "우수",
    "2": "보통",
    "3": "불량"
}

# 검사 결과 저장할 변수
from collections import Counter
class_count = Counter()

missing_images = []
empty_annotations = []
invalid_class_files = []
invalid_json_files = []
thermal_files = [] 

json_files = list(labels_dir.rglob("*.json"))
print(f"찾은 json 파일 수 : {len(json_files)}개")

image_paths = [
            image_path for image_path in image_dir.rglob("*")
            if image_path.suffix.lower() in {
                ".jpg",".jpeg",".png"
            }
        ]
image_by_name = {}

for image_path in image_paths:
    image_by_name[image_path.stem] = image_path

print(f"찾은 이미지 파일 수 : {len(image_paths)}개")
   
import json

for json_path in json_files:
    try:
        #"r"은 read 읽기 "w"는 새로쓰기 "a"는 기존내용 이어쓰기
        with json_path.open("r",encoding="utf-8") as file:
            data = json.load(file)

        #json 내부에서 필요한 부분 가져오기
        source_info = data["Source_Data_Info"]
        learning_info= data["Learning_Data_Info"]

        shooting_id = source_info["Shooting_ID"]
        source_data_id = source_info["Source_Data_ID"]
        annotations = learning_info["Annotations"]

        if shooting_id != "R":
            #R=일반사진 "T"=열화상 열화상은제외시키는 코드
            thermal_files.append(json_path.name)
            continue
        
        # json 대응 하는 jpg 경로
        image_path = image_by_name.get(source_data_id)

        if image_path is None:
            missing_images.append(source_data_id)
            continue

        # 대응하는 이미지 없을시 기록
        if not image_path.exists():
            missing_images.append(image_path.name)
            continue

        # json안의 id가져오기
        if not annotations : 
            empty_annotations.append(json_path.name)
            continue

        class_ids = {
            str(annotation["Class_ID"])
            for annotation in annotations
        }

        #하나의 sjon에 서로다른 등급이 들어 있으면 오류로기록
        if len(class_ids) != 1:
            invalid_class_files.append(json_path.name)
            continue

        class_id = class_ids.pop()

        if class_id not in class_names:
            invalid_class_files.append(json_path.name)
            continue

        class_count[class_id] += 1

    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        OSError
    ) as error:
        
        invalid_json_files.append(json_path.name)

        print(f"\n파일 처리 실패 : {json_path.name}")
        print(f"오류 내용 : {error}")

print("\n================================================")

for class_id, class_name in class_names.items():
    print(f"{class_name}: {class_count[class_id]}장")

normal_count = sum(class_count.values())

print("검사결과")
print(f"정상 RGB 데이터 : {normal_count}개")
print(f"열화상 JSON : {len(thermal_files)}개")
print(f"이미지가 없는 JSON : {len(missing_images)}개")
print(f"Annotations가 빈 JSON : {len(empty_annotations)}개")
print(f"Class_ID가 잘못된 JSON : {len(invalid_class_files)}개")
print(f"읽을 수 없는 JSON : {len(invalid_json_files)}개" ) 

if missing_images:
    print("\n이미지가 없는 json의 이미지명")

    for file_name in missing_images[:10]:
        print(f"-{file_name}")

if empty_annotations:
    print("\nAnnotations가 비어 있는 json")

    for file_name in empty_annotations[:10]:
        print(f"-{file_name}")

if invalid_class_files:
    print("\nClass_id에 문제가 있는 Json")

    for file_name in invalid_class_files[:10]:
        print(f"-{file_name}")

if invalid_json_files:
    print("\n읽을 수 없는 JSON")

    for file_name in invalid_json_files[:10]:
        print(f"-{file_name}")


