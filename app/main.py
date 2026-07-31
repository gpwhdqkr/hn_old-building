import os
import uuid
import threading
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape
from flask import Flask, render_template, request, send_from_directory, send_file, make_response, jsonify, abort  # 🆕 [진단 이력 기능] make_response, jsonify, abort 추가 (2026-07-31)
import torch
import cv2
import numpy as np
from bson import ObjectId
from pymongo import MongoClient

# 로컬 코어 모듈 임포트
from ai_engine import ApartmentClassifier
from cv_processor import draw_defect_bounding_boxes, draw_excellent_text_stamp, draw_heatmap_overlay

app = Flask(__name__)
project_dir = Path(__file__).resolve().parent

# =========================================================================
# 🔒 [프론트 정적 파일 가드]
# =========================================================================
@app.route('/style.css')
def serve_css():
    return send_from_directory('templates', 'style.css')

@app.route('/script.js')
def serve_script():
    return send_from_directory('templates', 'script.js')

# 모델 및 디바이스 할당
model_path = project_dir.parent / "model" / "best_convnext_tiny_binclf_v3_finetuned_v3cta.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = ApartmentClassifier(model_path, device)

# MongoDB 초기 설정
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["apartment_inspection_db"]
collection = db["inspection_logs"]


@app.route('/')
def home():
    resp = make_response(render_template('finally.html'))                       # 🆕 [진단 이력 기능] 수정된 줄 (2026-07-31)
    if not request.cookies.get(CLIENT_ID_COOKIE):                              # 🆕 [진단 이력 기능] 추가된 줄 — 상수는 파일 하단 블록
        resp.set_cookie(CLIENT_ID_COOKIE, uuid.uuid4().hex,                    # 🆕 [진단 이력 기능] 추가된 줄
                        max_age=CLIENT_ID_MAX_AGE, httponly=True, samesite='Lax')  # 🆕 [진단 이력 기능] 추가된 줄
    return resp                                                                 # 🆕 [진단 이력 기능] 수정된 줄


def sanitize_origin_name(raw_name):
    """업로드된 원본 파일명을 표시용으로 정리한다 (확장자 유지).

    파일은 UUID 이름으로 저장하므로(아래 unique_filename) 이 값은 화면·보고서
    표시 전용이다. 일부 브라우저가 전체 경로를 보내므로 basename만 취하고,
    제어문자를 걷어낸 뒤 길이를 제한한다. 쓸 게 없으면 빈 문자열.
    """
    if not raw_name:
        return ""
    name = str(raw_name).replace('\\', '/').split('/')[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:255]


@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files.get('house_image')
    if not uploaded_file or uploaded_file.filename == '':
        return '<script>alert("검사할 주택 사진 파일이 선택되지 않았습니다."); window.location.href = "/";</script>'

    # 1단계: 용량 가드
    uploaded_file.seek(0, os.SEEK_END)
    file_size_bytes = uploaded_file.tell()
    uploaded_file.seek(0)

    if file_size_bytes > 50 * 1024 * 1024:
        return '<script>alert("파일 용량이 너무 큽니다. 50MB 이하의 이미지만 업로드해 주세요."); window.location.href = "/";</script>'

    # 2단계: 확장자 필터
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.jfif'}
    file_extension = os.path.splitext(uploaded_file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        return '<script>alert("허용되지 않은 파일 형식입니다. JPG, JPEG, PNG, JFIF 이미지만 업로드해 주세요."); window.location.href = "/";</script>'

    save_extension = '.jpg' if file_extension == '.jfif' else file_extension

    origin_dir = os.path.join('static', 'images', 'origin')
    result_dir = os.path.join('static', 'images', 'result')
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 저장은 UUID로 하되(충돌·경로조작 방지), 사용자가 올린 이름은 표시용으로 따로 남긴다.
    origin_display_name = sanitize_origin_name(uploaded_file.filename)
    unique_filename = f"{uuid.uuid4().hex}{save_extension}"
    file_path = os.path.join(origin_dir, unique_filename)
    uploaded_file.save(file_path)

    # 3단계: 초고속 해상도 가드 (전체 픽셀 로드 방식을 피하고 Pillow 구조 기반 0.001초 만에 종횡 가로세로 추출)
    try:
        from PIL import Image as PILImage
        with PILImage.open(file_path) as img_check:
            w, h = img_check.size
        if w > 1024 or h > 1024:
            if os.path.exists(file_path):
                os.remove(file_path)
            return f'<script>alert("이미지 해상도가 너무 큽니다. 가로 및 세로가 1024픽셀 이하인 사진을 올려주세요. (업로드된 크기: {w}x{h})"); window.location.href = "/";</script>'
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return '<script>alert("이미지 파일 구조 검증 중 오류가 발생했습니다."); window.location.href = "/";</script>'

    result_file_name = f"result_{unique_filename}"
    result_file_path = os.path.join(result_dir, result_file_name)
    heatmap_file_name = f"heatmap_{unique_filename}"
    heatmap_file_path = os.path.join(result_dir, heatmap_file_name)

    prediction = -1
    result_status = "분류 실패 (프로세스 오류)"
    defect_probability = None
    inference_time_ms = None
    peak_x, peak_y = None, None
    display_image_path = file_path
    heatmap_display_path = None

    try:
        (prediction, result_status, defect_probability,
         grayscale_cam, cam_peak_xy, inference_time_ms) = classifier.predict_and_get_cam(file_path)

        if prediction == 1 and grayscale_cam is not None:
            peak_x, peak_y = draw_defect_bounding_boxes(file_path, result_file_path, grayscale_cam, cam_peak_xy)
            display_image_path = result_file_path
            try:
                draw_heatmap_overlay(file_path, heatmap_file_path, grayscale_cam)
                heatmap_display_path = heatmap_file_path
            except Exception as heatmap_err:
                print(f"❌ 히트맵 생성 실패: {heatmap_err}")
        elif prediction == 0:
            draw_excellent_text_stamp(file_path, result_file_path)
            display_image_path = result_file_path
    except Exception as e:
        print(f"❌ 추론 파이프라인 에러: {e}")
        result_status = "분류 실패 (에러 발생)"

    web_origin_path = f"/{file_path.replace('\\', '/')}"
    web_result_path = f"/{display_image_path.replace('\\', '/')}"
    web_heatmap_path = f"/{heatmap_display_path.replace('\\', '/')}" if heatmap_display_path else None

    if defect_probability is None:
        probability_percent = None
    elif prediction == 1:
        probability_percent = round(defect_probability * 100, 1)
    else:
        probability_percent = round((1 - defect_probability) * 100, 1)

    # MongoDB 비동기 스레드 보완 (지연 원천 봉쇄)
    db_status = "오류" if "실패" in result_status else ("우수" if prediction == 0 else "불량")
    log_document = {
        "_id": ObjectId(),
        "origin_id": ObjectId(),
        "client_id": request.cookies.get(CLIENT_ID_COOKIE),  # 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)
        "origin_file_name": unique_filename,                 # 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)
        # ⚠️ 위 origin_file_name은 이름과 달리 UUID 저장명이다. 사용자가 올린 실제
        #    파일명(확장자 포함)은 아래 origin_display_name에 들어간다.
        "origin_display_name": origin_display_name,
        "origin_save_path": file_path,                       # 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)
        "result_file_name": result_file_name,
        "save_path": display_image_path,
        "heatmap_file_name": heatmap_file_name if heatmap_display_path else None,
        "heatmap_save_path": heatmap_display_path,
        "status": db_status,
        "defect_probability": round(defect_probability, 6) if defect_probability is not None else None,
        "inference_time_ms": inference_time_ms,
        "create_at": datetime.now()
    }

    def _async_mongo_insert(doc):
        try:
            client_tmp = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            client_tmp["apartment_inspection_db"]["inspection_logs"].insert_one(doc)
        except Exception:
            pass

    threading.Thread(target=_async_mongo_insert, args=(log_document,), daemon=True).start()

    if "실패" in result_status:
        return '<script>alert("진단 처리 중 오류가 발생했습니다."); window.location.href = "/";</script>'

    # 프론트 연동 6종 출력 — 변수 설명은 docs/FRONTEND_GUIDE.md 참고
    return render_template(
        'f_result.html',
        user_image_url=web_origin_path,
        cam_image_url=web_result_path,
        heatmap_image_url=web_heatmap_path,
        ai_result=result_status,
        probability_percent=probability_percent,
        peak_x=peak_x, peak_y=peak_y,
        inference_ms=inference_time_ms,
        # PDF 보고서가 쓰는 날것 그대로의 불량 확률 소수점 (f_result.html의 data-prob으로 나간다)
        raw_defect_probability=defect_probability,
        # 보고서에 찍을 원본 파일명 (f_result.html의 data-name으로 나간다)
        origin_display_name=origin_display_name
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
        # DB 장애가 화면을 죽이지 않도록 격리하되, "이력이 없다"고 단언하지는 않는다
        print(f"❌ 이력 목록 조회 오류: {history_err}")
        return jsonify({"items": [], "error": True})

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

    # ⚠️ 아래 두 계산(URL 변환·확률 %)은 predict()의 172-185행과 같은 내용입니다.
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
         # 🌟 [이 줄을 무조건 추가] AI가 계산한 날것 그대로의 불량률 소수점을 화면에 숨겨서 보냅니다.
        raw_defect_probability=defect_probability,
        # 이 필드가 없던 시절의 구 레코드는 빈 값 → 보고서가 UUID로 대체 표기한다
        origin_display_name=doc.get("origin_display_name") or ""
    )

# ── 🆕 [진단 이력 기능] 여기까지 ────────────────────────────────────────

# =========================================================================
# 📄 [완벽 정제] 덮어쓰기 버그 박멸 및 수학적 확률 가드 완결판 라우터
# =========================================================================
@app.route('/api/download-report')
def download_report():
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ModuleNotFoundError:
        return '<script>alert("서버에 reportlab이 설치되지 않았습니다."); window.location.href = "/";</script>'

    font_name = "MalgunGothic"
    win_font_path = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(win_font_path):
        try:
            pdfmetrics.registerFont(TTFont(font_name, win_font_path))
        except Exception:
            font_name = "Helvetica"
    else:
        font_name = "Helvetica"

      # 1. 프론트엔드가 수집하여 보낸 고유 파라미터 수령
    user_image_url = request.args.get('user_image', '')
    cam_image_url = request.args.get('cam_image', '')
    heatmap_image_url = request.args.get('heatmap_image', '')
    ai_result_raw = request.args.get('ai_result', '우수')
    inference_ms = request.args.get('inference_ms', '0')
    raw_prob_str = request.args.get('raw_prob', '')
    origin_name = sanitize_origin_name(request.args.get('origin_name', ''))

    # 🌟 [신뢰도 가드] 주입 단계에서 넘어온 문자열을 실수형(Float)으로 형변환한다.
    # 값이 없거나 깨졌을 때 임의 기본값(구 0.999)을 채우면 모든 보고서가 99.9% 불량으로
    # 나온다. 그래서 실패하면 None으로 두고, 판정은 아래에서 ai_result에만 맡긴다.
    try:
        defect_prob = float(raw_prob_str)
    except (TypeError, ValueError):
        defect_prob = None
    if defect_prob is not None and not 0.0 <= defect_prob <= 1.0:
        defect_prob = None

    # 대소문자 및 URL 인코딩 파편 방어 가드 적용
    ai_result_upper = ai_result_raw.upper()
    if "DEFECT" in ai_result_upper or "불량" in ai_result_upper:
        ai_result = "불량"
    else:
        ai_result = "우수"

    origin_path = user_image_url.lstrip('/')
    bbox_path = cam_image_url.lstrip('/')
    heatmap_path = heatmap_image_url.lstrip('/') if heatmap_image_url else None

    # 사용자가 올린 원본 파일명을 확장자까지 그대로 찍는다.
    # 이 값이 없는 건(구 이력 레코드 등)은 예전처럼 UUID 저장명으로 대체한다.
    photo_filename = os.path.basename(origin_path)
    photo_id = origin_name or os.path.splitext(photo_filename)[0].upper()

    # 셀 폭(180pt)을 넘기면 표가 세로로 늘어지므로 확장자를 남기고 줄인다
    if len(photo_id) > 52:
        _stem, _ext = os.path.splitext(photo_id)
        photo_id = f"{_stem[:max(1, 52 - len(_ext) - 1)]}…{_ext}"

    # Paragraph는 마크업을 파싱하므로 &, <, > 가 든 파일명은 이스케이프해야 한다
    photo_id = xml_escape(photo_id)

    os.makedirs('generated', exist_ok=True)
    pdf_path = "generated/exterior_wall_diagnosis_report.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName=font_name, fontSize=20, leading=24, alignment=1, spaceAfter=20)
    section_title = ParagraphStyle('SecTitle', parent=styles['Heading2'], fontName=font_name, fontSize=12, leading=16, textColor=colors.HexColor('#2563eb'), spaceBefore=15, spaceAfter=8)
    body_style = ParagraphStyle('BodyText', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=14, textColor=colors.HexColor('#334155'))
    id_style = ParagraphStyle('IdText', parent=body_style, fontSize=9, leading=12, alignment=1, wordWrap='CJK')

    story.append(Paragraph("<b>AI 노후 건물 외벽 진단 시스템 판정 보고서</b>", title_style))
    story.append(Spacer(1, 10))

     # =========================================================================
    # 🌟 [최종 완결] 원문 문구 100% 유지 및 수치 기반 강제 동기화 라우터
    # =========================================================================
    # 소프트맥스 확률 분포를 1:1로 대조 연산하여 리얼 데이터 도출
    # 확률 원값이 유실된 경우에는 수치를 지어내지 않고 "확률 정보 없음"으로 표기한다.
    if defect_prob is not None:
        real_excellent_percent = round((1.0 - defect_prob) * 100, 1)
        real_defect_percent = round(defect_prob * 100, 1)
        probability_display = f"우수 {real_excellent_percent}%, 불량 {real_defect_percent}%"
        defect_indicator = f"{real_defect_percent}%"
        excellent_indicator = f"{real_excellent_percent}%"
    else:
        real_excellent_percent = None
        real_defect_percent = None
        probability_display = "확률 정보 없음"
        defect_indicator = "확률 정보 없음"
        excellent_indicator = "확률 정보 없음"

    # 우수/불량 판정은 ai_engine이 운영 임계값(self.threshold)으로 이미 내린 결론이
    # ai_result로 넘어온 것이다. 여기서 50% 기준으로 다시 판정하면 운영 임계값과
    # 어긋나므로 재판정하지 않는다.

    meta_widths = [90, 180, 90, 180]
    meta_data = [
        [Paragraph("<b>원본 파일명</b>", body_style), Paragraph(photo_id, id_style), Paragraph("<b>진단 일시</b>", body_style), Paragraph(datetime.now().strftime("%Y. %m. %d %H:%M:%S"), body_style)],
        [Paragraph("<b>순수 추론 속도</b>", body_style), Paragraph(f"{inference_ms} ms", body_style), Paragraph("<b>AI 판정 확률</b>", body_style), Paragraph(probability_display, body_style)]
    ]
    meta_table = Table(meta_data, colWidths=meta_widths)
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')), ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#f8fafc')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6), ('TOPPADDING', (0,0), (-1,-1), 6)
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>■ 종합 판정 결과</b>", section_title))
    
    # 🌟 원본 요구 명세 문구를 토시 하나 바꾸지 않고 100% 유지하여 조건부 분기합니다.
    if ai_result == "불량":
        result_color = "#e11d48"
        result_text = f"<b>[불량] - 구조적 하자가 감지되었습니다. (AI 불량 판단 지표: {defect_indicator})</b>"
        detail_desc = (f"외벽 레이어 내에서 불량 확률 {defect_indicator}의 연산치로 "
                       f"외관 손상 및 결함 요인이 감지되었습니다. 다만, 본 판정 결과는 인공지능의 판단이므로 "
                       f"안전을 위해 반드시 건축구조 전문가의 현장 정밀 육안 진단과 소견이 필요합니다.")
    else:
        result_color = "#10b981"
        result_text = f"<b>[우수] - 건축물 외벽 상태가 안정적인 수준으로 확인되었습니다. (AI 우수 판단 지표: {excellent_indicator})</b>"
        detail_desc = (f"외벽 레이어 내에서 우수 확률 {excellent_indicator}의 안전율로 하자요인이 검지되지 않았습니다. "
                       f"현재 외벽의 상태가 안정적인 수준으로 유지되고 있는 것으로 판정됩니다. 다만, 보다 정밀한 안전성 확보를 위해 "
                       f"건축구조 전문가의 현장 정밀 육안 점검 및 일상 관리를 병행하는 것을 권장합니다.")

    status_widths = [100, 440]
    status_data = [
        [Paragraph("<b>AI 판정 결과</b>", body_style), Paragraph(f"<font color='{result_color}'>{result_text}</font>", body_style)],
        [Paragraph("<b>종합 진단 소견</b>", body_style), Paragraph(detail_desc, body_style)]
    ]
    status_table = Table(status_data, colWidths=status_widths)
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')), ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')), ('PADDING', (0,0), (-1,-1), 8)
    ]))
    story.append(status_table)
    story.append(Spacer(1, 15))

    # 불량은 3열(원본·히트맵·B-Box), 우수는 2열이다.
    # 우수의 ②·③은 같은 도장 이미지를 두 번 싣던 중복이라 ③ 상태 각인을 걷어냈다.
    story.append(Paragraph(
        "<b>■ 컴퓨터 비전 증거 자료 (3-Layer Analysis)</b>" if ai_result == "불량"
        else "<b>■ 컴퓨터 비전 증거 자료 (2-Layer Analysis)</b>", section_title))

    # Pillow 정보 구조만 가볍게 파싱하여 연산 지연 속도를 0.001초 미만으로 차단
    def _get_ratio_preserved_image(img_path, max_cell_w=172, max_cell_h=130):
        if not img_path or not os.path.exists(img_path):
            return Paragraph("<font color='#94a3b8'>[데이터 누락]</font>", body_style)
        try:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as img_p:
                orig_w, orig_h = img_p.size
            aspect = orig_w / float(orig_h)
            if aspect >= (max_cell_w / float(max_cell_h)):
                final_w, final_h = max_cell_w, int(max_cell_w / aspect)
            else:
                final_h, final_w = max_cell_h, int(max_cell_h * aspect)
            return Image(img_path, width=final_w, height=final_h)
        except Exception:
            return Image(img_path, width=max_cell_w, height=max_cell_h)

    if ai_result == "불량":
        img_widths = [180, 180, 180]
        if heatmap_path and os.path.exists(heatmap_path):
            img_heat = _get_ratio_preserved_image(heatmap_path)
        else:
            img_heat = Paragraph("<font color='#94a3b8'>[LayerCAM 미기동<br/>(우수 상태)]</font>", body_style)
        img_data = [
            [_get_ratio_preserved_image(origin_path), img_heat, _get_ratio_preserved_image(bbox_path)],
            [Paragraph("<b>① 원본 파일 (Before)</b>", body_style),
             Paragraph("<b>② 히트맵 분포 (Heatmap)</b>", body_style),
             Paragraph("<b>③ 결함 바운딩 (B-Box)</b>", body_style)]
        ]
    else:
        # 2열이라 셀이 넓어진 만큼 이미지도 키운다 (표 전체 폭은 위 판정표와 같은 540)
        img_widths = [270, 270]
        img_data = [
            [_get_ratio_preserved_image(origin_path, 260, 190),
             _get_ratio_preserved_image(bbox_path, 260, 190)],
            [Paragraph("<b>① 원본 파일 (Before)</b>", body_style),
             Paragraph("<b>② 우수 인증 (Excellent)</b>", body_style)]
        ]
    img_table = Table(img_data, colWidths=img_widths)
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')), ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f1f5f9')),
        ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6)
    ]))
    story.append(img_table)

    foot_style = ParagraphStyle('Foot', parent=body_style, fontName=font_name, fontSize=7, leading=10, textColor=colors.HexColor('#94a3b8'))
    story.append(Spacer(1, 30))
    story.append(Paragraph("본 보고서는 인공지능의 자가 외벽 분석 참조용 서식입니다. 본 판정 결과는 분쟁이나 매매 등을 위한 법적 공식 안전진단조사서의 효력을 대체할 수 없으며, 촬영 당시의 광량, 해상도, 렌즈 왜곡률에 따라 국소 영역 판정 오차가 발생할 수 있으므로 최종적인 안전성 평가는 전문가의 진단에 의해야 합니다.", foot_style))

    doc.build(story)
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)