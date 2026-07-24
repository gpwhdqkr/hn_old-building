import os
from datetime import datetime
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
import uuid # 함수 맨 위에 이 라이브러리가 필요합니다.
import cv2  # 🔴 [마지막 퍼즐 추가!] OpenCV 라이브러리를 불러옵니다.
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)

# 1. MongoDB 설정
client = MongoClient('mongodb://localhost:27017')
db = client['db_name']
origin_collection = db['origin_detection_logs']
result_collection = db['result_detection_logs']

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
         # 📂 images 안의 origin과 result 폴더 설정 및 생성
        origin_dir = os.path.join('static', 'images', 'origin')
        result_dir = os.path.join('static', 'images', 'result')
        os.makedirs(origin_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        
        # 🔑 확장자(.jpg 등)만 분리한 뒤, 무작위 영문숫자(uuid)로 파일명 생성
        file_ext = os.path.splitext(uploaded_file.filename)[1] 
        unique_filename = f"{uuid.uuid4().hex}{file_ext}" # 예: a1b2c3d4.jpg

        # 💾 내부 저장 및 처리는 안전한 영문 이름으로 수행
        file_path = os.path.join(origin_dir, unique_filename)
        uploaded_file.save(file_path)

        # 결과 파일명도 영문 이름 기반으로 매칭
        result_file_name = f"result_{unique_filename}"
        result_file_path = os.path.join(result_dir, result_file_name)
        
        
        current_time = datetime.now() # 에러 방지용 미리 선언
        
        # 🟢 [버그 해결 포인트 1] 어떤 에러가 나도 무조건 기본 원본 경로를 가지도록 
        # try문 바로 바깥 상단에 안전하게 변수를 대피시켜 선언합니다.
        display_image_path = file_path 
        is_result_created = False  # 🟢 결과 파일이 실제로 생성되었는지 추적할 플래그
        
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
                
                  # 5. 원본 도화지를 로드하고 크기 측정
                output_img = cv2.imread(file_path)
                h, w, _ = output_img.shape
                
                # 🔥 [개선 포인트] 이미지 크기에 비례하는 다이나믹 두께 계산
                # 가로 폭이 클수록 선과 글씨가 자동으로 더 두꺼워집니다. (최소 두께 5px 보장)
                base_thickness = max(5, int(w / 300)) 
                font_scale = max(0.8, w / 1000)      # 글자 크기도 해상도에 맞게 확대

                # 6. 발견된 결함 덩어리들을 하나씩 순서대로 네모 상자로 가두는 반복문 실행
                for contour in contours:
                    x, y, box_w, box_h = cv2.boundingRect(contour)
                    
                    x1 = int(x * (w / 224))
                    y1 = int(y * (h / 224))
                    x2 = int((x + box_w) * (w / 224))
                    y2 = int((y + box_h) * (h / 224))
                    
                    # 7. 🔴 계산된 다이나믹 두께(base_thickness)로 빨간색 테두리 선 긋기
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), base_thickness)
                    
                    # 8. 🔴 계산된 글자 크기(font_scale)와 두께로 안내 글자 삽입
                    # 글씨가 박스 밖 화면 위로 넘어가지 않도록 좌표 조정 (y1 - 15)
                    text_y = y1 - 15 if y1 - 15 > 30 else y1 + 30
                    cv2.putText(output_img, "Defect Area", (x1, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), max(2, int(base_thickness * 0.6)))
                
                # 9. 빨간 낙서가 완료된 도화지를 하드디스크에 새 파일명으로 최종 영구 저장
                cv2.imwrite(result_file_path, output_img)
                
                # 10. 배달용 최종 이미지 경로 변수를 '빨간 박스 사진' 경로로 갈아끼우기
                display_image_path = result_file_path
                is_result_created = True  # 🟢 결함이 발견되어 결과 이미지가 파일로 저장됨!
            else:
                # 🟢 '우수(0)' 판정일 때: 원본에 EXCELLENT 글자를 써서 결과물 생성
                try:
                    # 1. 한글 경로 안전 로드
                    img_array = np.fromfile(file_path, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    # 2. 이미지 중앙 좌표 계산 (가로 1/4, 세로 1/2 지점)
                    h, w, _ = img.shape
                    text_position = (int(w / 4), int(h / 2))
                    
                    # 3. 초록색 글씨 쓰기 (BGR: 0, 255, 0)
                    cv2.putText(img, "EXCELLENT (GOOD)", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # 4. 결과 폴더에 한글 경로 깨짐 없이 안전하게 저장
                    _, encoded_img = cv2.imencode('.png', img)
                    encoded_img.tofile(result_file_path)
                    
                    # 5. 웹 화면에 보여줄 경로를 결과 이미지로 지정
                    display_image_path = result_file_path
                    is_result_created = True
                except Exception as img_err:
                    print(f"우수 이미지 생성 중 에러 발생: {img_err}")
                    # 실패 시 차선책으로 원본 이미지라도 띄우도록 설정
                    display_image_path = file_path
        except Exception as e:
            result_status = "분류 실패 (에러 발생)"
            print(f"예측 및 CAM 시각화 중 에러 발생: {e}")
        # --- 🌲 AI 모델 예측 프로세스 끝 ---
        
        current_time = datetime.now() # KST 기준 로컬 시간 또는 표준 시간 사용
        
        # 1. 원본 이미지 테이블 기록
        origin_doc = {
            "origin_file_name": uploaded_file.filename,
            "save_path": file_path,
            "status": result_status,  # 정의서의 '우수/보통 등' 상태 값 저장
            "create_at": current_time
        }
        origin_collection.insert_one(origin_doc)
        
        # 2. 결과 이미지 테이블 기록 (빨간 박스가 그려진 결과물이 존재할 때만 저장)
        if is_result_created:
            result_doc = {
                "result_file_name": result_file_name,
                "save_path": result_file_path,
                "status": result_status,
                "create_at": current_time
            }
            result_collection.insert_one(result_doc)
        
      # 🟢 [버그 완벽 해결] HTML 템플릿의 변수 이름과 역할을 1:1로 정확하게 매칭합니다.
        
        # 1. 파일 경로의 역슬래시(\)를 슬래시(/)로 통일하여 웹 브라우저 인식 오류 방지
        web_origin_path = f"/{file_path.replace('\\', '/')}"
        web_result_path = f"/{display_image_path.replace('\\', '/')}"

        return render_template(
            'result.html', 
            user_image_url=web_origin_path,  # 👈 왼쪽: 무조건 순수한 원본 사진 경로!
            cam_image_url=web_result_path,   # 👈 오른쪽: 결과 사진 경로 (보통/불량 CAM 또는 우수 도장)!
            ai_result=result_status          # 👈 상단 중앙: 판정 결과 텍스트 ("우수", "보통", "불량")
        )
    
    return """
    <script>
        alert("검사할 주택 사진 파일이 선택되지 않았습니다.");
        window.location.href = "/";
    </script>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)