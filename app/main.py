import os
import uuid
from datetime import datetime
from pathlib import Path
# ➔ [수정 코드] 맨 뒤에 ', send_from_directory'를 추가합니다.
from flask import Flask, render_template, request, send_from_directory, make_response, jsonify, abort  # 🆕 [진단 이력 기능] make_response, jsonify, abort 추가 (2026-07-31)
import torch
import cv2
import numpy as np
from bson import ObjectId  # 🍃 MongoDB ObjectId 생성을 위해 임포트
from pymongo import MongoClient

# 💡 두 개의 로컬 모듈을 임포트합니다.
from ai_engine import ApartmentClassifier
from cv_processor import draw_defect_bounding_boxes, draw_excellent_text_stamp, draw_heatmap_overlay

app = Flask(__name__)
# main.py 상단 수정
project_dir = Path(__file__).resolve().parent

# =========================================================================
# 🔒 [프론트 구조 구원 가드] templates 폴더 내 정적 파일 강제 매핑 규칙
# =========================================================================
@app.route('/style.css')
def serve_css():
    # 브라우저가 /style.css를 요청하면 templates 폴더 안에서 찾아 반환합니다.
    return send_from_directory('templates', 'style.css')

@app.route('/script.js')
def serve_script():
    # 브라우저가 /script.js를 요청하면 templates 폴더 안에서 찾아 반환합니다.
    return send_from_directory('templates', 'script.js')
# =========================================================================

# 최종 선정 모델: ConvNeXt-Tiny 이원화(binclf_v3, 448 입력) 가중치 경로
model_path = project_dir.parent / "model" / "best_convnext_tiny_binclf_v3_finetuned_v3cta.pth"

# 🔒 [클라우드/로컬 공용] CUDA 환경 유무를 자동 체크하여 디바이스 할당
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 서버가 켜질 때 ConvNeXt-Tiny 기반 객체를 메모리에 단 한 번 올립니다.
classifier = ApartmentClassifier(model_path, device)

# =========================================================================
# 🍃 [MongoDB 설정] 로컬 및 클라우드 호환용 컨텍스트 선언
# =========================================================================
# 클라우드 배포 시 환경 변수(Environment Variable) 환경에 맞추어 주소를 유연하게 전환 가능합니다.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["apartment_inspection_db"]
collection = db["inspection_logs"]
# =========================================================================

@app.route('/')
def home():
    resp = make_response(render_template('finally.html'))                       # 🆕 [진단 이력 기능] 수정된 줄 (2026-07-31)
    if not request.cookies.get(CLIENT_ID_COOKIE):                              # 🆕 [진단 이력 기능] 추가된 줄 — 상수는 파일 하단 블록
        resp.set_cookie(CLIENT_ID_COOKIE, uuid.uuid4().hex,                    # 🆕 [진단 이력 기능] 추가된 줄
                        max_age=CLIENT_ID_MAX_AGE, httponly=True, samesite='Lax')  # 🆕 [진단 이력 기능] 추가된 줄
    return resp                                                                 # 🆕 [진단 이력 기능] 수정된 줄

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
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.jfif'}
    file_extension = os.path.splitext(uploaded_file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        return '<script>alert("허용되지 않은 파일 형식입니다. JPG, JPEG, PNG, JFIF 이미지만 업로드해 주세요."); window.location.href = "/";</script>'

    # 💡 [핵심 교정] .jfif 환경 등에서의 OpenCV 호환성 및 경로 깨짐 방지를 위해 내부 저장 확장자를 .jpg로 통일
    save_extension = '.jpg' if file_extension == '.jfif' else file_extension

    # 폴더 구조 빌드
    origin_dir = os.path.join('static', 'images', 'origin')
    result_dir = os.path.join('static', 'images', 'result')
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # file_extension 대신 안전한 save_extension 기반으로 고유 파일명 생성
    unique_filename = f"{uuid.uuid4().hex}{save_extension}"
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
    # 히트맵 탭 전용 결과물 (박스 이미지와 같은 크기·좌표계로 저장된다)
    heatmap_file_name = f"heatmap_{unique_filename}"
    heatmap_file_path = os.path.join(result_dir, heatmap_file_name)

    # 방어적 제어 변수 선언 (에러 시 프론트로 None이 내려가도록)
    prediction = -1
    result_status = "분류 실패 (프로세스 오류)"
    defect_probability = None
    inference_time_ms = None
    peak_x = None
    peak_y = None
    display_image_path = file_path
    heatmap_display_path = None   # 히트맵이 없는 경우(우수/에러)는 None 유지

    try:
        # ❶ AI 엔진 호출 (ai_engine.py) — 이원화 판정 + 불량 확률 + LayerCAM + 추론 ms
        (prediction, result_status, defect_probability,
         grayscale_cam, cam_peak_xy, inference_time_ms) = classifier.predict_and_get_cam(file_path)

        # ❷ OpenCV 이미지 프로세서 호출 (cv_processor.py)
        if prediction == 1 and grayscale_cam is not None:
            # 불량: 결함 근사 박스 드로잉 (최종 확정 사양 — 히트맵/마커 미표시).
            # 반환값은 원본 좌표계 피크 (x, y) — 데이터로만 프론트에 전달
            peak_x, peak_y = draw_defect_bounding_boxes(
                file_path, result_file_path, grayscale_cam, cam_peak_xy
            )
            display_image_path = result_file_path

            # [추가] 같은 히트맵으로 JET 오버레이 이미지를 한 장 더 굽는다 (프론트 히트맵 탭용).
            # 🔒 히트맵 생성이 실패해도 박스 결과 화면은 살아 있어야 한다 (탭만 비활성)
            try:
                draw_heatmap_overlay(file_path, heatmap_file_path, grayscale_cam)
                heatmap_display_path = heatmap_file_path
            except Exception as heatmap_err:
                print(f"❌ 히트맵 오버레이 생성 실패(박스 결과는 유지): {heatmap_err}")
        elif prediction == 0:
            draw_excellent_text_stamp(file_path, result_file_path)
            display_image_path = result_file_path

    except Exception as e:
        print(f"❌ 추론/시각화 파이프라인 에러 발생: {e}")
        result_status = "분류 실패 (에러 발생)"

    # 웹 표준 경로 슬래시 정제
    web_origin_path = f"/{file_path.replace('\\', '/')}"
    web_result_path = f"/{display_image_path.replace('\\', '/')}"
    web_heatmap_path = (
        f"/{heatmap_display_path.replace('\\', '/')}" if heatmap_display_path else None
    )

    # 판정된 클래스의 확률 % (우수면 1-p, 불량이면 p) — 프론트 표시용
    if defect_probability is None:
        probability_percent = None
    elif prediction == 1:
        probability_percent = round(defect_probability * 100, 1)
    else:
        probability_percent = round((1 - defect_probability) * 100, 1)

    # =========================================================================
    # 🍃 [MongoDB 데이터 저장] 기존 BSON 구조 유지 + v3 추론 필드 추가
    # =========================================================================
    if "실패" in result_status:
        db_status = "오류"
    else:
        db_status = "우수" if prediction == 0 else "불량"

    log_document = {
        "_id": ObjectId(),                               # 다큐먼트 고유 ID
        "origin_id": ObjectId(),                         # 원본 참조용 고유 ID
        "client_id": request.cookies.get(CLIENT_ID_COOKIE),  # 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)
        "origin_file_name": unique_filename,                 # 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)
        "origin_save_path": file_path,                       # 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)
        "result_file_name": result_file_name,            # 결과 파일명
        "save_path": display_image_path,                 # 서버 내부 물리 저장 경로 (역슬래시 유지)
        "heatmap_file_name": (                           # 히트맵 이미지 파일명 (없으면 None)
            heatmap_file_name if heatmap_display_path else None
        ),
        "heatmap_save_path": heatmap_display_path,       # 히트맵 물리 저장 경로 (없으면 None)
        "status": db_status,                             # "우수" / "불량" / "오류"
        "defect_probability": (                          # 불량 확률 원값 (분석/재튜닝용)
            round(defect_probability, 6) if defect_probability is not None else None
        ),
        "inference_time_ms": inference_time_ms,          # 추론 소요 시간 (ms)
        "create_at": datetime.now()                   # ISO UTC 타임스탬프 형식
    }

    try:
        collection.insert_one(log_document)
    except Exception as mongo_err:
        # DB 트랜잭션 장애가 유저의 웹 결과 화면 출력을 방해하지 않도록 격리 조치
        print(f"❌ MongoDB 저장 오류: {mongo_err}")
    # =========================================================================
    
    if "실패" in result_status:
        return ('<script>alert("진단 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.");'
            ' window.location.href = "/";</script>')

    # 프론트 연동 6종 출력 — 변수 설명은 docs/FRONTEND_GUIDE.md 참고
    return render_template(
        'f_result.html',
        # -- 기존 변수 (하위 호환 유지) --
        user_image_url=web_origin_path,      # 원본 이미지 URL
        cam_image_url=web_result_path,       # ① 판정 근거 시각화 이미지 URL (결함 박스/스탬프)
        heatmap_image_url=web_heatmap_path,  # [추가] ⑥ LayerCAM 히트맵 오버레이 URL (우수/에러 시 None)
        ai_result=result_status,             # ② 판정 결과: "우수" / "불량" / "분류 실패 (...)"
        # -- v3 신규 변수 --
        probability_percent=probability_percent,  # ③ 판정 클래스의 확률 % (0~100, 소수 1자리)
        peak_x=peak_x,                       # ④ 확인 요망 지점 x (원본 픽셀 좌표, 근사치. 우수면 None)
        peak_y=peak_y,                       # ④ 확인 요망 지점 y (원본 픽셀 좌표, 근사치. 우수면 None)
        inference_ms=inference_time_ms       # ⑤ 이미지 1장 추론 속도 (ms)
    )

# =========================================================================
# ── 🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31) ──────────────────────
# 좌측 사이드바 진단 이력. 설계: docs/superpowers/specs/2026-07-31-history-sidebar-design.md
# =========================================================================

CLIENT_ID_COOKIE = "hn_client_id"     # 이력 식별용 쿠키 이름 (로그인이 없어 쿠키로만 구분)
CLIENT_ID_MAX_AGE = 60 * 60 * 24 * 2  # 2일. 이력도 사실상 이틀치가 된다
HISTORY_LIMIT = 20                    # 목록에 띄우는 최대 건수

@app.route('/history')
def history_list():
    """쿠키 주인의 최근 진단 20건을 JSON으로. 오류 건도 포함한다(프론트가 X로 표시)."""
    client_id = request.cookies.get(CLIENT_ID_COOKIE)
    if not client_id:
        return jsonify({"items": []})

    try:
        cursor = (collection.find({"client_id": client_id})
                  .sort("create_at", -1)
                  .limit(HISTORY_LIMIT))

        items = []
        for doc in cursor:
            origin = doc.get("origin_save_path")
            created = doc.get("create_at")
            items.append({
                "id": str(doc["_id"]),
                # 연도는 빼고 월/일 시:분만 (요구사항)
                "date": created.strftime("%m/%d %H:%M") if created else "",
                "status": doc.get("status", "오류"),
                # 물리 경로(역슬래시)를 웹 URL로. 원본이 없는 구 레코드는 None
                "thumb": f"/{origin.replace('\\', '/')}" if origin else None,
            })
        return jsonify({"items": items})

    except Exception as history_err:
        # DB 장애가 화면을 죽이지 않도록 격리 — insert_one과 같은 방침
        print(f"❌ 이력 목록 조회 오류: {history_err}")
        return jsonify({"items": []})

@app.route('/history/<item_id>')
def history_detail(item_id):
    """저장된 진단 1건을 /predict와 똑같은 f_result.html 조각으로 복원한다.
    프론트는 이 응답을 기존 injectBackendResult()에 그대로 넘기면 된다."""
    client_id = request.cookies.get(CLIENT_ID_COOKIE)
    if not client_id:
        abort(404)

    try:
        doc = collection.find_one({"_id": ObjectId(item_id)})
    except Exception as detail_err:
        # ObjectId 형식 오류 또는 DB 장애
        print(f"❌ 이력 상세 조회 오류: {detail_err}")
        abort(404)

    # 남의 기록 열람 차단 + 복원할 결과가 없는 건 제외
    if not doc or doc.get("client_id") != client_id or doc.get("status") == "오류":
        abort(404)

    origin = doc.get("origin_save_path")
    result = doc.get("save_path")
    heatmap = doc.get("heatmap_save_path")
    defect_probability = doc.get("defect_probability")

    # f_result.html의 막대그래프가 probability_percent를 반드시 쓰므로 없으면 복원 불가
    if defect_probability is None:
        abort(404)

    # ⚠️ 아래 두 계산(URL 변환·확률 %)은 predict()의 169-181행과 같은 내용입니다.
    #    팀 작업 충돌을 피하려고 predict()를 건드리지 않고 의도적으로 중복시켰습니다.
    #    확률 표시 방식이나 경로 변환을 바꿀 때는 반드시 양쪽을 같이 고쳐 주세요.
    #    나중에 정리할 여유가 생기면 아래 헬퍼를 살리고 양쪽 인라인 계산을
    #    헬퍼 호출로 바꾸면 됩니다.
    #
    # def to_web_path(path):
    #     """윈도우 물리 경로를 웹 URL로. 없으면 None."""
    #     return f"/{path.replace('\\', '/')}" if path else None
    #
    # def to_probability_percent(defect_probability, is_defect):
    #     """불량 확률 원값 → 판정된 클래스의 % 값."""
    #     if defect_probability is None:
    #         return None
    #     return round((defect_probability if is_defect else 1 - defect_probability) * 100, 1)

    is_defect = doc.get("status") == "불량"
    if is_defect:
        probability_percent = round(defect_probability * 100, 1)
    else:
        probability_percent = round((1 - defect_probability) * 100, 1)

    return render_template(
        'f_result.html',
        user_image_url=f"/{origin.replace('\\', '/')}" if origin else "",
        cam_image_url=f"/{result.replace('\\', '/')}" if result else "",
        heatmap_image_url=f"/{heatmap.replace('\\', '/')}" if heatmap else None,
        ai_result=doc.get("status", ""),
        probability_percent=probability_percent,
        peak_x=None,   # f_result.html이 쓰지 않아 DB에 저장하지 않는다
        peak_y=None,
        inference_ms=doc.get("inference_time_ms"),
    )

# ── 🆕 [진단 이력 기능] 여기까지 ────────────────────────────────────────

if __name__ == '__main__':
    # 클라우드 컨테이너 포트 바인딩 및 외부 접속 유연화를 위해 기본 호스트 오픈 적용
    app.run(host='0.0.0.0', port=5000, debug=True)
