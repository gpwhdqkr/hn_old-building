import cv2
import numpy as np
import torch  # 🔒 LayerCAM 텐서 타입 체크를 위해 유지

# 전처리 역매핑에 필요한 서빙 규격 (단일 출처: ai_engine.py)
from ai_engine import EVAL_RESIZE_SHORT, INPUT_SIZE

# 히트맵 이진화 기준 — 이 값 이상으로 반응한 영역만 박스로 잡는다.
# 박스가 너무 많이/자잘하게 잡히면 0.6~0.7로 올려서 조정.
BOX_THRESHOLD = 0.5

# 노이즈 박스 제거: 한 변이 이 픽셀 이하인 박스는 무시 (히트맵 좌표계 기준)
MIN_BOX_SIZE = 15


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
    박스를 원본 전체 좌표에 그대로 그리면 위치가 어긋난다 (역매핑 필수).
    """
    ratio = EVAL_RESIZE_SHORT / min(width, height)
    crop_size = int(round(INPUT_SIZE / ratio))
    crop_size = min(crop_size, width, height)  # 반올림으로 원본을 넘는 것 방지
    x0 = (width - crop_size) // 2
    y0 = (height - crop_size) // 2
    return x0, y0, crop_size


def draw_defect_bounding_boxes(file_path, result_file_path, grayscale_cam, cam_peak_xy):
    """LayerCAM 히트맵을 이진화해 결함 근사 영역에 빨간 박스를 그린다.
    (히트맵·마커는 이미지에 그리지 않는다 — 최종 확정 사양)

    반환: 원본 이미지 좌표계의 히트맵 피크 (x, y) — 이미지에는 표시하지 않지만
    프론트 데이터(④ 확인 요망 지점)로 전달된다. 박스·피크 모두 히트맵 기반
    근사치이므로 UI에서 정밀 경계가 아님을 명시할 것.
    """
    cam = _normalize_cam(grayscale_cam)

    # 업로드된 원본 이미지 로드 (최대 가로/세로 1024 이하 상태)
    output_img = _read_img(file_path)
    h, w, _ = output_img.shape
    x0, y0, crop_size = _cam_region_in_origin(w, h)

    # 1. 히트맵을 원본 대응 영역 크기로 확대 후 이진화 → 컨투어 추출
    cam_resized = cv2.resize(cam, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    binary_map = (cam_resized > BOX_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1024 해상도 제한 스펙에 최적화된 선 두께 및 글자 크기 비율
    base_thickness = max(3, int(w / 250))
    font_scale = max(0.6, w / 900)

    for contour in contours:
        bx, by, box_w, box_h = cv2.boundingRect(contour)

        # 작은 노이즈 영역은 제외하고 유효 결함만 박스 마킹
        if box_w > MIN_BOX_SIZE and box_h > MIN_BOX_SIZE:
            # 크롭 영역 좌표 → 원본 좌표로 오프셋 이동 후 빨간 박스 드로잉
            point1 = (x0 + bx, y0 + by)
            point2 = (x0 + bx + box_w, y0 + by + box_h)
            cv2.rectangle(output_img, point1, point2, (0, 0, 255), base_thickness)

            # 글자가 이미지 상단 경계를 벗어나지 않도록 Y 좌표 가드 설정
            text_y = point1[1] - 12 if point1[1] - 12 > 25 else point1[1] + 25
            cv2.putText(output_img, "Defect Area", (point1[0], text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                        max(1, int(base_thickness * 0.5)))

    # 2. 피크 좌표: 448 크롭 좌표계 → 원본 좌표계 (데이터 전달용, 드로잉 없음)
    scale = crop_size / float(INPUT_SIZE)
    peak_x = max(0, min(x0 + int(round(cam_peak_xy[0] * scale)), w - 1))
    peak_y = max(0, min(y0 + int(round(cam_peak_xy[1] * scale)), h - 1))

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
