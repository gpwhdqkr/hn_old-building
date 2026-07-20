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
                
        except Exception as e:
            result_status = "분류 실패 (에러 발생)"
            print(f"예측 중 에러 발생: {e}")
        # --- 🌲 AI 모델 예측 프로세스 끝 ---
        
        # DB 기록 (AI 예측 결과 포함)
        doc = {
            "file_name" : uploaded_file.filename,
            "save_path" : file_path,
            "status" : result_status,  # "분류중" 대신 실제 AI 판정 결과("우수", "보통", "불량") 저장
            "created-at" : datetime.datetime.now()
        }
        collection.insert_one(doc)
        
        return render_template(
    'result.html', 
    user_image_url=f"/{file_path}", # html의 {{ user_image_url }} 자리에 들어감
    ai_result=result_status         # html의 {{ ai_result }} 자리에 들어감
)
    
    return """
    <script>
        alert("검사할 주택 사진 파일이 선택되지 않았습니다.");
        window.location.href = "/";
    </script>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)