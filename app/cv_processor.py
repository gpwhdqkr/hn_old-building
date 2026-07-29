import cv2
import numpy as np
import torch  # 🔒 LayerCAM 텐서 타입 체크를 위해 유지

# 전처리 역매핑에 필요한 서빙 규격 (단일 출처: ai_engine.py)
from ai_engine import EVAL_RESIZE_SHORT, INPUT_SIZE

# layercam_binclf_v3.py의 오버레이와 동일한 블렌드 강도
HEATMAP_ALPHA = 0.45


def _read_img(file_path):
    """한글 경로 호환 원본 이미지 로드"""
    img_array = np.fromfile(file_path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def _save_img(file_path, img):
    """한글 경로 호환 이미지 저장"""
    _, encoded_img = cv2.imencode('.png', img)
    encoded_img.tofile(file_path)


def _normalize_cam(grayscale_cam):
    """텐서/3차원 배열 등 어떤 형태로 와도 (H, W) float32 0~1 넘파이로 정리한다."""
    # 🔒 [클라우드/배포 안전 가드] 파이토치 텐서 대비 CPU 넘파이 강제 변환
    if isinstance(grayscale_cam, torch.Tensor):
        grayscale_cam = grayscale_cam.detach().cpu().numpy()
    grayscale_cam = np.asarray(grayscale_cam, dtype=np.float32)

    # 🔒 [안전 가드] (1, H, W)든 (H, W, 1)이든 2차원(H, W)으로 압축
    if grayscale_cam.ndim == 3:
        if grayscale_cam.shape[0] == 1:
            grayscale_cam = grayscale_cam[0]
        elif grayscale_cam.shape[-1] == 1:
            grayscale_cam = grayscale_cam[:, :, 0]

    return np.clip(grayscale_cam, 0.0, 1.0)


def _cam_region_in_origin(width, height):
    """전처리(Resize 512 짧은 변 → CenterCrop 448)에서 히트맵(448×448)이
    원본 이미지의 어느 영역에 대응하는지 (x0, y0, 한 변 픽셀)를 역산한다.

    모델은 이미지 전체가 아니라 이 중앙 정사각 영역만 보고 판정하므로,
    히트맵/마커를 원본 전체에 펴 바르면 위치가 어긋난다 (역매핑 필수).
    """
    ratio = EVAL_RESIZE_SHORT / min(width, height)
    crop_size = int(round(INPUT_SIZE / ratio))
    crop_size = min(crop_size, width, height)  # 반올림으로 원본을 넘는 것 방지
    x0 = (width - crop_size) // 2
    y0 = (height - crop_size) // 2
    return x0, y0, crop_size


def draw_defect_heatmap_overlay(file_path, result_file_path, grayscale_cam, cam_peak_xy):
    """LayerCAM 히트맵을 원본의 대응 영역에 오버레이하고 피크 마커를 찍는다.

    반환: 원본 이미지 좌표계의 피크 (x, y).
    피크는 히트맵 최대점 기반 '근사 위치'다 — UI에서 "이 지점 확인 요망
    (근사치)"처럼 정확한 박스가 아님을 반드시 명시할 것.
    """
    cam = _normalize_cam(grayscale_cam)

    # 업로드된 원본 이미지 로드 (최대 가로/세로 1024 이하 상태)
    output_img = _read_img(file_path)
    h, w, _ = output_img.shape
    x0, y0, crop_size = _cam_region_in_origin(w, h)

    # 1. 히트맵을 원본 대응 영역 크기로 확대 후 JET 컬러맵(결함부 빨강) 적용
    cam_resized = cv2.resize(cam, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 2. 대응 영역에만 알파 블렌드 — 영역 밖은 원본 그대로 (모델이 안 본 부분)
    region = output_img[y0:y0 + crop_size, x0:x0 + crop_size]
    output_img[y0:y0 + crop_size, x0:x0 + crop_size] = cv2.addWeighted(
        region, 1.0 - HEATMAP_ALPHA, heat_color, HEATMAP_ALPHA, 0
    )

    # 3. 피크 좌표: 448 크롭 좌표계 → 원본 좌표계 변환 후 마커 드로잉
    scale = crop_size / float(INPUT_SIZE)
    peak_x = x0 + int(round(cam_peak_xy[0] * scale))
    peak_y = y0 + int(round(cam_peak_xy[1] * scale))
    peak_x = max(0, min(peak_x, w - 1))
    peak_y = max(0, min(peak_y, h - 1))

    # 1024 해상도 제한 스펙에 연동되는 다이나믹 마커/글자 크기
    radius = max(10, w // 40)
    thickness = max(2, w // 300)
    font_scale = max(0.6, w / 900)

    # 흰 테두리 + 빨간 원 이중 드로잉 (어두운/밝은 배경 모두에서 가시성 확보)
    cv2.circle(output_img, (peak_x, peak_y), radius, (255, 255, 255), thickness + 2)
    cv2.circle(output_img, (peak_x, peak_y), radius, (0, 0, 255), thickness)

    # 글자가 이미지 경계를 벗어나지 않도록 좌표 가드
    text_y = peak_y - radius - 12 if peak_y - radius - 12 > 25 else peak_y + radius + 28
    text_x = max(5, min(peak_x - radius, w - 220))
    cv2.putText(output_img, "Check Here", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 2)
    cv2.putText(output_img, "Check Here", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), max(1, thickness))

    # 최종 결과물 디스크 저장
    _save_img(result_file_path, output_img)

    return peak_x, peak_y


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
