import os
import cv2
import numpy as np

def _read_img(file_path):
    """한글 경로 호환 원본 이미지 로드"""
    img_array = np.fromfile(file_path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def _save_img(file_path, img):
    """한글 경로 호환 이미지 저장"""
    _, encoded_img = cv2.imencode('.png', img)
    encoded_img.tofile(file_path)

def draw_defect_bounding_boxes(file_path, result_file_path, grayscale_cam):
    """
    Grad-CAM 마스크를 기반으로 다이나믹한 빨간색 사각형 상자와 텍스트를 그리는 함수
    """
    # 🔒 [안전 가드] Grad-CAM의 출력 차원이 3차원(Batch 포함)일 경우, 첫 번째 이미지만 가져오도록 방어합니다.
    if len(grayscale_cam.shape) == 3:
        grayscale_cam = grayscale_cam[0]

    # 1. Grad-CAM 마스크 이진화 (0~255 스케일 변환)
    binary_map = (grayscale_cam > 0.5).astype(np.uint8) * 255
    
    # 2. 업로드된 원본 이미지 로드 (최대 가로/세로 1024 이하 상태)
    output_img = _read_img(file_path)
    h, w, _ = output_img.shape
    
    # 3. 모델의 224x224 영역 맵핑 출력을 현재 원본 이미지 크기(w, h)에 1:1 역매칭
    binary_map_resized = cv2.resize(binary_map, (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(binary_map_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # [보완] 1024 해상도 제한 스펙에 최적화된 선 두께 및 글자 크기 비율 조정
    base_thickness = max(3, int(w / 250)) 
    font_scale = max(0.6, w / 900)      

    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        
        # 작은 노이즈 영역은 제외하고 유효 결함만 박스 마킹
        if box_w > 15 and box_h > 15:
            # 결함 영역 빨간색 박스 드로잉
            cv2.rectangle(output_img, (x, y), (x + box_w, y + box_h), (0, 0, 255), base_thickness)
            
            # 글자가 이미지 상단 경계를 벗어나지 않도록 Y 좌표 가드 설정
            text_y = y - 12 if y - 12 > 25 else y + 25
            
            # 텍스트 받아쓰기
            cv2.putText(output_img, "Defect Area", (x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), max(1, int(base_thickness * 0.5)))
            
    # 최종 결과물 디스크 저장
    _save_img(result_file_path, output_img)

def draw_excellent_text_stamp(file_path, result_file_path):
    """우수 등급일 때 이미지 해상도 비율을 계산하여 정중앙에 녹색 도장을 찍는 함수"""
    img = _read_img(file_path)
    h, w, _ = img.shape
    
    # [보완] 이미지 너비(w)에 연동되는 다이나믹 스탬프 구현
    dynamic_font_scale = max(1.0, w / 600)
    dynamic_thickness = max(2, int(w / 300))
    
    # EXCELLENT 문구의 대략적인 길이를 계산하여 완벽한 중앙 정렬 좌표 설정
    text_w = int(dynamic_font_scale * 300)
    text_x = max(10, int((w - text_w) / 2))
    text_y = int(h / 2)
    
    cv2.putText(img, "EXCELLENT (GOOD)", (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, (0, 255, 0), dynamic_thickness)
    _save_img(result_file_path, img)