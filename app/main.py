import os
import datetime
from pathlib import Path
from flask import Flask, render_template, request
from pymongo import MongoClient
import torch
from torch import nn
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
from PIL import Image
# 사각화
import numpy as np
import cv2  # 🔴 [마지막 퍼즐 추가!] OpenCV 라이브러리를 불러옵니다.
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)

# 1. MongoDB 설정
client = MongoClient('mongodb://localhost:27017')
db = client['db_name']
collection = db['detection_logs']

# 2. 경로 설정 및 AI 모델 로드
project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "model" / "best_mobilenet_v2_baseline.pth"

# CPU/GPU 설정 (서버는 안전하게 CPU로 구동하는 것을 권장)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 클래스 이름 정의
class_names = {0: "우수", 1: "보통", 2: "불량"}

# 팀원 코드와 동일한 뼈대의 빈 모델 생성
model = mobilenet_v2()
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 3)

# 학습된 파일(.pth)이 있으면 가중치 로드
if model_path.exists():
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"🎉 성공: AI 가중치 파일을 성공적으로 불러왔습니다! (가장 높았던 정확도: {checkpoint.get('validation_accuracy', 0):.2%})")
else:
    print(f"⚠️ 경고: {model_path} 경로에 모델 파일이 없습니다. 먼저 팀원 코드를 돌려 모델을 생성해야 합니다.")

model = model.to(device)
model.eval()  # 예측 모드로 설정

# 3. 이미지 전처리 함수 정의 (AI 모델이 읽을 수 있는 형태로 변환)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV2 기본 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 기본 정규화 값
        std=[0.229, 0.224, 0.225]
    )
])

# 4. 라우터 설정
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files.get('house_image')
    
    if uploaded_file and uploaded_file.filename != '':
        # static/images 폴더가 없으면 자동 생성
        save_dir = os.path.join('static', 'images')
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, uploaded_file.filename)
        uploaded_file.save(file_path)
        
        # 빨간 박스가 쳐질 새로운 결과물 이미지 파일의 저장 경로 정의
        result_file_name = f"result_{uploaded_file.filename}"
        result_file_path = os.path.join(save_dir, result_file_name)
        
        # 🟢 [버그 해결 포인트 1] 어떤 에러가 나도 무조건 기본 원본 경로를 가지도록 
        # try문 바로 바깥 상단에 안전하게 변수를 대피시켜 선언합니다.
        display_image_path = file_path 
        
        # --- 🔥 AI 모델 예측 프로세스 시작 ---
        try:
            # 이미지 열기 및 전처리
            image = Image.open(file_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device) # 배치 차원 추가
            
            # AI 예측 실행
            with torch.no_grad():
                outputs = model(image_tensor)
                prediction = outputs.argmax(dim=1).item()
                result_status = class_names.get(prediction, "알 수 없음")
                
            # ---------------------------------------------------------------------------
            # 🔴 역추적 및 빨간 사각형 상자 그리기 로직 (Grad-CAM + OpenCV)
            # ---------------------------------------------------------------------------
            # AI 판정 결과가 '보통(1)' 또는 '불량(2)'인 결함 상태일 때만 상자를 그립니다.
            if prediction in[1, 2]:
                
                # 🟢 [버그 해결 포인트 2] 혹시 위쪽 전역 변수에서 누락되었을 상황을 대비해
                # 역추적을 시작하는 이 순간에 target_layers를 한 번 더 확실하게 선언해 박아줍니다.
                target_layers = [model.features[-1]]
                
                # 1. AI와 라이브러리가 알아들을 수 있는 표준 작업지시서(Target) 발행
                cam = GradCAM(model=model, target_layers=target_layers)
                targets = [ClassifierOutputTarget(prediction)]
                
                # 2. AI 뇌세포 회로를 거꾸로 미분 역추적하여 0.0 ~ 1.0 점수의 흑백 지도 생성
                grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0] # 배치 차원 제거
                
                # 3. 중요 기여도가 50%(0.5) 이상인 픽셀만 남기고 나머지는 까맣게 지우는 이진화 작업
                binary_map = (grayscale_cam > 0.5).astype(np.uint8) * 255
                
                # 4. 흰색으로 뭉쳐진 결함 부위 덩어리들의 가장 바깥쪽 테두리 외곽선(Contours)을 추출
                contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 5. 빨간 잉크를 문지를 원본 도화지를 메모리에 로드하고, 원본의 진짜 가로(w)/세로(h) 크기 측정
                output_img = cv2.imread(file_path)
                h, w, _ = output_img.shape
                
                # 6. 발견된 결함 덩어리들을 하나씩 순서대로 네모 상자로 가두는 반복문 실행
                for contour in contours:
                    # 미니 바둑판(224x224) 기준의 가장 작은 사각형 시작 좌표(x,y)와 가로세로 폭(box_w, box_h) 추출
                    x, y, box_w, box_h = cv2.boundingRect(contour)
                    
                    # [⭐️ 비율 연산] 224 기준 미니 좌표를 사용자가 올린 대형 원본 해상도 크기로 확대 소환
                    x1 = int(x * (w / 224))
                    y1 = int(y * (h / 224))
                    x2 = int((x + box_w) * (w / 224))
                    y2 = int((y + box_h) * (h / 224))
                    
                    # 7. 원본 도화지 위에 계산된 좌표로 순수한 빨간색(BGR: 0, 0, 255) 테두리 선(두께 3px) 긋기
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # 8. 빨간 상자 머리 위에 결함 구역임을 표시하는 안내 글자 삽입
                    cv2.putText(output_img, "Defect Area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 9. 빨간 낙서가 완료된 도화지를 하드디스크에 새 파일명으로 최종 영구 저장
                cv2.imwrite(result_file_path, output_img)
                
                # 10. 배달용 최종 이미지 경로 변수를 '빨간 박스 사진' 경로로 갈아끼우기
                display_image_path = result_file_path
                
        except Exception as e:
            result_status = "분류 실패 (에러 발생)"
            print(f"예측 및 CAM 시각화 중 에러 발생: {e}")
        # --- 🌲 AI 모델 예측 프로세스 끝 ---
        
        # DB 기록 (AI 예측 결과 및 결과 이미지 경로 포함)
        doc = {
            "file_name" : uploaded_file.filename,
            "save_path" : file_path,
            "result_path" : display_image_path,
            "status" : result_status,  
            "created-at" : datetime.datetime.now()
        }
        collection.insert_one(doc)
        
        return render_template(
            'result.html', 
            user_image_url=f"/{display_image_path}", 
            ai_result=result_status         
        )
    
    return """
    <script>
        alert("검사할 주택 사진 파일이 선택되지 않았습니다.");
        window.location.href = "/";
    </script>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)