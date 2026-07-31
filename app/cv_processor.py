import os          # ← 추가
import cv2
import numpy as np
import torch  # 🔒 LayerCAM 텐서 타입 체크를 위해 유지

# 전처리 역매핑에 필요한 서빙 규격 (단일 출처: ai_engine.py)
from ai_engine import EVAL_RESIZE_SHORT, INPUT_SIZE

# 히트맵 이진화 기준 — 이 값 이상으로 반응한 영역만 박스로 잡는다.
# 박스가 너무 많이/자잘하게 잡히면 0.6~0.7로 올려서 조정.
BOX_THRESHOLD = 0.5

# [약한 히트맵 방어 1단계] 확신도가 낮은 불량은 히트맵 반응이 약해 절대 임계값
# (BOX_THRESHOLD)을 넘는 영역이 없을 수 있다. 이때는 "히트맵 최대값 × 이 비율"
# 지점으로 임계값을 자동 완화한다 (강한 히트맵은 기존 동작 그대로 유지).
RELATIVE_THRESHOLD_RATIO = 0.7

# 노이즈 박스 제거: 한 변이 이 픽셀 이하인 박스는 무시 (히트맵 좌표계 기준)
MIN_BOX_SIZE = 15

# [약한 히트맵 방어 2단계] 크기 필터까지 통과한 박스가 0개면 피크 좌표 중심으로
# 그리는 보장 박스의 반변 길이 = 크롭 영역 한 변 × 이 비율 (불량 = 박스 ≥ 1 계약)
FALLBACK_BOX_HALF_RATIO = 0.08

# 히트맵 오버레이 투명도 — 낮추면 원본 질감이, 올리면 반응 분포가 잘 보인다.
# 0.45는 src/v3/layercam_binclf_v3.py의 검증 오버레이와 같은 값.
HEATMAP_ALPHA = 0.45


def _read_img(file_path):
    """한글 경로 호환 원본 이미지 로드"""
    img_array = np.fromfile(file_path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def _save_img(file_path, img):
    """한글 경로 호환 이미지 저장.

    인코딩 포맷을 목표 파일명의 확장자에 맞춘다. 항상 '.png'로 인코딩하면
    result_xxx.jpg 안에 PNG 바이트가 들어가, 서버가 Content-Type을
    image/jpeg로 잘못 선언하고 용량도 2배 이상 부풀린다.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        # 박스 선/글자 경계가 뭉개지지 않도록 품질을 높게 유지
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif ext == '.png':
        params = []
    else:
        # 알 수 없는 확장자는 무손실 PNG로 폴백 (확장자도 맞춰줌)
        ext, params = '.png', []
        file_path = os.path.splitext(file_path)[0] + '.png'

    success, encoded_img = cv2.imencode(ext, img, params)
    if not success:
        raise ValueError(f"결과 이미지 인코딩 실패 (확장자: {ext})")
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

    # [방어 1단계] 히트맵이 약해 절대 임계값을 넘는 영역이 없으면 박스가 0개가
    # 되므로, 최대값 대비 상대 임계값과 비교해 낮은 쪽을 쓴다. min()으로 묶여
    # 있어 강한 히트맵(최대값 ≈ 1)은 기존 BOX_THRESHOLD 동작과 완전히 동일하다.
    peak_value = float(cam_resized.max())
    threshold = min(BOX_THRESHOLD, peak_value * RELATIVE_THRESHOLD_RATIO)
    binary_map = (cam_resized > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1024 해상도 제한 스펙에 최적화된 선 두께 및 글자 크기 비율
    base_thickness = max(3, int(w / 250))
    font_scale = max(0.6, w / 900)

    def _draw_box(point1, point2):
        cv2.rectangle(output_img, point1, point2, (0, 0, 255), base_thickness)
        # 글자가 이미지 상단 경계를 벗어나지 않도록 Y 좌표 가드 설정
        text_y = point1[1] - 12 if point1[1] - 12 > 25 else point1[1] + 25
        cv2.putText(output_img, "Defect Area", (point1[0], text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                    max(1, int(base_thickness * 0.5)))

    boxes_drawn = 0
    for contour in contours:
        bx, by, box_w, box_h = cv2.boundingRect(contour)

        # 작은 노이즈 영역은 제외하고 유효 결함만 박스 마킹
        if box_w > MIN_BOX_SIZE and box_h > MIN_BOX_SIZE:
            # 크롭 영역 좌표 → 원본 좌표로 오프셋 이동 후 빨간 박스 드로잉
            _draw_box((x0 + bx, y0 + by), (x0 + bx + box_w, y0 + by + box_h))
            boxes_drawn += 1

    # 2. 피크 좌표: 448 크롭 좌표계 → 원본 좌표계 (데이터 전달용 + 2단계 가드)
    scale = crop_size / float(INPUT_SIZE)
    peak_x = max(0, min(x0 + int(round(cam_peak_xy[0] * scale)), w - 1))
    peak_y = max(0, min(y0 + int(round(cam_peak_xy[1] * scale)), h - 1))

    # [방어 2단계] 크기 필터까지 거친 박스가 하나도 없으면 피크 좌표 중심의
    # 보장 박스 1개를 그린다 → "불량 판정 = 박스 ≥ 1개" 계약을 항상 만족.
    if boxes_drawn == 0:
        half = max(MIN_BOX_SIZE * 2, int(crop_size * FALLBACK_BOX_HALF_RATIO))
        point1 = (max(0, peak_x - half), max(0, peak_y - half))
        point2 = (min(w - 1, peak_x + half), min(h - 1, peak_y + half))
        _draw_box(point1, point2)

    # 최종 결과물 디스크 저장
    _save_img(result_file_path, output_img)

    return peak_x, peak_y


def draw_heatmap_overlay(file_path, result_file_path, grayscale_cam):
    """LayerCAM 히트맵을 JET 컬러맵으로 원본에 겹쳐 저장한다 (박스·마커 미표시).

    프론트 before/after 슬라이더의 after 레이어 중 "히트맵 탭"용 이미지다.
    박스와 좌표계·크기가 완전히 동일해야 탭을 바꿔도 이미지가 튀지 않으므로
    역매핑은 draw_defect_bounding_boxes와 같은 _cam_region_in_origin을 쓴다.

    모델이 실제로 본 중앙 크롭 영역에만 블렌딩하고 그 경계를 흰 실선으로 그려
    "판정 근거 영역"이 이미지 전체가 아니라는 점을 시각적으로 명시한다.
    """
    cam = _normalize_cam(grayscale_cam)

    output_img = _read_img(file_path)
    h, w, _ = output_img.shape
    x0, y0, crop_size = _cam_region_in_origin(w, h)

    # 히트맵(448×448)을 원본 대응 영역 크기로 확대 후 JET 컬러맵 적용
    cam_resized = cv2.resize(cam, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    region = output_img[y0:y0 + crop_size, x0:x0 + crop_size]
    output_img[y0:y0 + crop_size, x0:x0 + crop_size] = cv2.addWeighted(
        region, 1.0 - HEATMAP_ALPHA, heat, HEATMAP_ALPHA, 0
    )

    # 분석 영역 경계선 + 범례 문구 (1024 이하 해상도 스펙에 맞춘 비율)
    thickness = max(2, int(w / 400))
    cv2.rectangle(output_img, (x0, y0), (x0 + crop_size - 1, y0 + crop_size - 1),
                  (255, 255, 255), thickness)
    cv2.putText(output_img, "LayerCAM Heatmap", (x0 + 8, max(28, y0 + 32)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1100), (255, 255, 255),
                max(1, thickness))

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
