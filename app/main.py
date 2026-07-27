import os
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request
import torch
import cv2
import numpy as np

# 💡 두 개의 로컬 모듈을 임포트합니다.
from ai_engine import ApartmentClassifier
from cv_processor import draw_defect_bounding_boxes, draw_excellent_text_stamp

app = Flask(__name__)

# main.py 상단 수정
project_dir = Path(__file__).resolve().parent

# EfficientNet-B0 이름으로 교체된 가중치 파일 경로
model_path = project_dir.parent / "model" / "best_efficientnet_b0_finetuned_epoch20.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 서버가 켜질 때 EfficientNet 기반 객체를 메모리에 단 한 번 올립니다.
classifier = ApartmentClassifier(model_path, device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files.get('house_image')
    
    if not uploaded_file or uploaded_file.filename == '':
        return '<script>alert("검사할 주택 사진 파일이 선택되지 않았습니다."); window.location.href = "/";</script>'
        
    # =========================================================================
    # 🔒 [방어 가드 1단계] 파일 용량 체크 (하드디스크 저장 전 50MB 이하 제한)
    # =========================================================================
    uploaded_file.seek(0, os.SEEK_END)
    file_size_bytes = uploaded_file.tell()
    uploaded_file.seek(0)  # [중요] 체크 완료 후 파일 포인터를 반드시 다시 처음으로 리셋

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    if file_size_bytes > MAX_FILE_SIZE:
        return '<script>alert("파일 용량이 너무 큽니다. 50MB 이하의 이미지만 업로드해 주세요."); window.location.href = "/";</script>'

    # =========================================================================
    # 🔒 [방어 가드 2단계] 이미지 확장자 필터링
    # =========================================================================
    allowed_extensions = {'.png', '.jpg', '.jpeg'}
    file_extension = os.path.splitext(uploaded_file.filename)[1].lower() 
    
    if file_extension not in allowed_extensions:
        return '<script>alert("허용되지 않은 파일 형식입니다. JPG, JPEG, PNG 이미지만 업로드해 주세요."); window.location.href = "/";</script>'
        
    # 폴더 구조 빌드
    origin_dir = os.path.join('static', 'images', 'origin')
    result_dir = os.path.join('static', 'images', 'result')
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(origin_dir, unique_filename)
    uploaded_file.save(file_path)

    # =========================================================================
    # 🔒 [방어 가드 3단계] 이미지 실제 해상도 크기 제한 (가로/세로 1024픽셀 이하)
    # =========================================================================
    try:
        img_array = np.fromfile(file_path, np.uint8)
        check_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if check_img is None:
            raise ValueError("손상되었거나 올바르지 않은 이미지 파일 구조")
            
        h, w, _ = check_img.shape
        
        if w > 1024 or h > 1024:
            # 해상도 조건을 불만족하면 저장했던 원본 임시 파일을 서버 디스크에서 즉시 지우고 차단합니다.
            if os.path.exists(file_path):
                os.remove(file_path)
            return f'<script>alert("이미지 해상도가 너무 큽니다. 가로 및 세로가 1024픽셀 이하인 사진을 올려주세요. (업로드된 크기: {w}x{h})"); window.location.href = "/";</script>'
            
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"❌ 업로드 이미지 무결성 검사 중 에러: {e}")
        return '<script>alert("이미지 파일을 검증하는 도중 오류가 발생했습니다. 정상적인 사진 파일인지 확인해 주세요."); window.location.href = "/";</script>'
    # =========================================================================

    result_file_name = f"result_{unique_filename}"
    result_file_path = os.path.join(result_dir, result_file_name)
    
    # 방어적 제어 변수 선언
    result_status = "분류 실패 (프로세스 오류)"
    display_image_path = file_path 

    try:
        # ❶ AI 엔진 호출 (ai_engine.py) - EfficientNet-B0 기반 실시간 텐서 전처리 후 추론 진행
        prediction, result_status, grayscale_cam, inference_time = classifier.predict_and_get_cam(file_path)
        
        # ❷ OpenCV 이미지 프로세서 호출 (cv_processor.py) - 검증된 안전한 파일만 처리함
        if prediction in [1, 2] and grayscale_cam is not None:
            draw_defect_bounding_boxes(file_path, result_file_path, grayscale_cam)
            display_image_path = result_file_path
        elif prediction == 0:
            draw_excellent_text_stamp(file_path, result_file_path)
            display_image_path = result_file_path
            
    except Exception as e:
        print(f"❌ 추론/시각화 파이프라인 에러 발생: {e}")
        result_status = "분류 실패 (에러 발생)"

    # 4. 웹 표준 경로 슬래시 정제 및 응답 렌더링
    web_origin_path = f"/{file_path.replace('\\', '/')}"
    web_result_path = f"/{display_image_path.replace('\\', '/')}"

    return render_template(
        'result.html', 
        user_image_url=web_origin_path,  
        cam_image_url=web_result_path,   
        ai_result=result_status          
    )

if __name__ == '__main__':
    # 로컬 테스트용이므로 기존 포트와 디버그 모드를 유지합니다.
    app.run(host='0.0.0.0', port=5000, debug=True)