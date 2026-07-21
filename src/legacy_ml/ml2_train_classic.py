import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42  #재현성을  위한 시드
IMAGE_SIZE = 64 #64x64 회색조로 축소
MAX_TRAIN_PER_CLASS = 2000 #SVM 학습 시간 때문에 클래스당 학습 이미지 상한
PCA_COMPONENTS = 100 #4096

# 팀 전처리 산출물 (팀원 비교의 기준점 — 절대 재분할하지 않음)
team_csv_path = Path(r"D:\teamproject\data\processed\metadata_split.csv")

# 팀 csv에 적힌 경로 접두어 (팀원 pc 기준)
team_prefix = "D:/hn_old-building_raw/raw/images/TS_아파트"

# 내 pc의 실제 데이터 위치(기본 + 2023-10-23 추가분 순서로 탐색)
my_roots = [
    Path(r"D:\189.서울시 노후 주택 균열 데이터\01.데이터\1.Training\원천데이터\TS_아파트"),
    Path(r"D:\189.서울시 노후 주택 균열 데이터\01.데이터\1.Training\원천데이터_231023_add\Ts_아파트"),
]

ml2_dir = Path(r"C:\Users\admin\Desktop\TeamProject_first\ml2")
data_dir = ml2_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

class_names = ["우수", "보통", "불량"]

# 팀 csv 읽기
metadata = pd.read_csv(team_csv_path, encoding="utf-8-sig")
print("전체 행 수:", len(metadata))
print(metadata["split"].value_counts())

# 팀원 pc경로 -> 내 pc 경로 치환

def fix_path(team_path):
    #접두어 뒤의 공통 부분(카테고리/등급/RGB/파일명)만 떼어냄
    suffix = team_path[len(team_prefix):].lstrip("/")

    #후보 폴더 2곳에 차례로 붙여보고, 실제 존재하는 쪽을 선택
    for root in my_roots:
        candidate = root / suffix
        if candidate.exists():
            return str(candidate)
        
    return None # 어디에도 없음

metadata["local_path"] = metadata["image_path"].apply(fix_path) #fix_path 함수로 경로 변경

missing_count = metadata["local_path"].isna().sum() #빈값 잇는지 True/False 한 뒤에 전부 더함.
print("찾지 못한 이미지 수(0이어야 정상):", missing_count)

metadata = metadata.dropna(subset=["local_path"]).reset_index(drop=True) #local_path 열의 결측치 삭제 후에 인덱스 초기화

# 팀이 나눈 split을 그대로 사용(재분할 금지)
train_data = metadata[metadata["split"] == "train"]
test_data = metadata[metadata["split"] == "test"].reset_index(drop=True)

# 클래스당 최대 MAX_TRAIN_PER_CLASS장만 사용 (시드 고정 -> 항상 같은 표본)
sampled_parts = []
for label, group in train_data.groupby("model_label"):
    n = min(len(group), MAX_TRAIN_PER_CLASS)
    sampled_parts.append(group.sample(n=n, random_state=SEED))
train_data = pd.concat(sampled_parts).reset_index(drop=True)

print("학습 이미지 수:", len(train_data))
print("평가(test) 이미지 수:", len(test_data))

# 이미지 -> 숫자 벡터

def load_features(dataframe, cache_name):
    # 한 번 변환한 결과는 npz로 저장해 두고 다음 실행부터 재사용
    cache_path = data_dir / f"{cache_name}_{IMAGE_SIZE}.npz"

    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["x"], cached["y"]
    
    features = []
    labels = []
    for row in dataframe.itertuples():
        with Image.open(row.local_path) as image_file:
            image =image_file.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )
        # 0~255 픽셀을 0~1 사이 실수 50176개로 펼침
        features.append(
            np.asarray(image, dtype=np.float32).flatten()/ 255.0
        )
        labels.append(row.model_label)

    x= np.stack(features)
    y= np.array(labels)
    np.savez_compressed(cache_path, x=x, y=y)
    return x, y

print("\n이미지 변환 중... (처음 한 번만 오래 걸림)")
x_train, y_train = load_features(train_data, f"train_{MAX_TRAIN_PER_CLASS}")
x_test, y_test = load_features(test_data, "test")