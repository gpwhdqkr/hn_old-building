# 진단 이력 사이드바 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 화면 좌측에 숨어 있는 사이드바를 열면 자기가 최근 돌린 진단 20건이 뜨고, 항목을 누르면 그때의 결과 화면이 그대로 복원된다.

**Architecture:** 쿠키(`hn_client_id`, 2일)로 사용자를 구분해 MongoDB `inspection_logs`에 같이 저장한다. `GET /history`가 목록 JSON을, `GET /history/<id>`가 `/predict`와 **똑같은 `f_result.html` 조각**을 반환한다. 프론트는 기존 `injectBackendResult()`를 그대로 재사용하므로 슬라이더·탭 로직에 손대지 않는다.

**Tech Stack:** Flask 3.1.3 · pymongo 4.17.0 · Python 3.14.6 · 바닐라 JS (빌드 도구 없음)

**설계 근거:** [docs/superpowers/specs/2026-07-31-history-sidebar-design.md](../specs/2026-07-31-history-sidebar-design.md)

## Global Constraints

- **자동 테스트를 만들지 않는다.** 저장소에 pytest·requirements.txt·CI가 전혀 없다. 각 Task는 대신 **실행 가능한 수동 검증 명령**으로 끝난다. 이는 스펙에서 승인된 결정이다.
- **`predict()` 본문을 건드리지 않는다.** 예외는 `log_document` 딕셔너리에 3줄을 더하는 것뿐. 방어 가드·추론 호출·URL 변환·확률 계산·`return render_template(...)`은 한 줄도 수정 금지.
- **새 코드는 파일 끝(또는 기존 구조 바깥)에 몰아 넣는다.** 팀원이 같은 파일을 동시에 고치고 있어 겹치는 줄을 만들지 않는 것이 목적이다.
  - `main.py` → `if __name__ == '__main__':` **직전**
  - `style.css` → 파일 맨 끝
  - `script.js` → `DOMContentLoaded` 콜백 맨 끝 (상단 `const` 선언부에 끼워 넣지 말 것)
  - `finally.html` → `.system-container` 바로 앞
- **마커 주석 필수.** 새 블록은 `🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31)` / `여기까지`로 감싼다. 기존 줄을 고친 자리는 줄 끝에 `# 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)`.
- **썸네일 56x56px, 목록 20건, 날짜 형식 `MM/DD HH:MM`** (연도 없음).
- **쿠키:** 이름 `hn_client_id`, 값 `uuid4().hex`, `max_age=172800`, `httponly=True`, `samesite='Lax'`.
- **`f_result.html`, `ai_engine.py`, `cv_processor.py`는 수정 금지.**
- 한글 주석을 쓴다. 기존 코드가 전부 한글 주석이다.

## 서버 실행 (모든 Task의 검증에서 공통)

```bash
cd D:/hn_old-building/.claude/worktrees/mongodb-image-save-issue-8ac202/app && python main.py
```

`http://localhost:5000` 으로 접속한다. `debug=True`라 파일을 고치면 자동 재기동된다.
개발 PC에 모델 가중치·MongoDB(69건)가 모두 있어 종단 검증이 된다.

`app/static/images/`는 이 워크트리에 없어 기동 시 새로 생긴다. 기존 69건이 가리키는
파일은 본 저장소 쪽에만 있지만 그 레코드들은 `client_id`가 없어 목록에 안 뜬다.

---

## File Structure

| 파일 | 책임 | 상태 |
|---|---|---|
| `app/main.py` | 쿠키 발급, DB 필드 3개, `/history`, `/history/<id>` | 수정 |
| `app/templates/finally.html` | 햄버거 버튼 · 사이드바 · 스크림 마크업 | 수정 |
| `app/templates/style.css` | 사이드바 스타일 | 수정 (파일 끝 추가) |
| `app/templates/script.js` | 목록 조회·렌더·복원 | 수정 (콜백 끝 추가) |
| `docs/PROJECT_MAP.md`, `docs/FRONTEND_GUIDE.md`, `CLAUDE.md` | 문서 갱신 | 수정 |

새로 만드는 소스 파일은 없다. 이 저장소는 파일 4개에 웹 서비스 전체가 들어 있는 구조이고,
기능 하나 때문에 모듈을 쪼개면 팀원의 병합 부담만 커진다.

---

## Task 1: 쿠키 발급 + DB 필드 3개

**Files:**
- Modify: `app/main.py:6` (flask import), `app/main.py:54-56` (`home()`), `app/main.py:191-206` (`log_document`), `if __name__` 직전 (상수 블록)

**Interfaces:**
- Produces: 모듈 상수 `CLIENT_ID_COOKIE = "hn_client_id"`, `CLIENT_ID_MAX_AGE = 172800`, `HISTORY_LIMIT = 20` — Task 2·3이 쓴다
- Produces: DB 문서에 `client_id`(str|None), `origin_file_name`(str), `origin_save_path`(str) 필드

- [ ] **Step 1: flask import에 3개 추가**

`app/main.py:6`을 아래로 교체한다.

```python
from flask import Flask, render_template, request, send_from_directory, make_response, jsonify, abort  # 🆕 [진단 이력 기능] make_response, jsonify, abort 추가 (2026-07-31)
```

- [ ] **Step 2: 파일 하단에 상수 블록 추가**

`if __name__ == '__main__':` **바로 앞**에 넣는다. 모듈 레벨 이름은 호출 시점에 해석되므로
위쪽 `home()`에서 참조해도 문제없다.

```python
# =========================================================================
# ── 🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31) ──────────────────────
# 좌측 사이드바 진단 이력. 설계: docs/superpowers/specs/2026-07-31-history-sidebar-design.md
# =========================================================================

CLIENT_ID_COOKIE = "hn_client_id"     # 이력 식별용 쿠키 이름 (로그인이 없어 쿠키로만 구분)
CLIENT_ID_MAX_AGE = 60 * 60 * 24 * 2  # 2일. 이력도 사실상 이틀치가 된다
HISTORY_LIMIT = 20                    # 목록에 띄우는 최대 건수

# ── 🆕 [진단 이력 기능] 여기까지 ────────────────────────────────────────
```

- [ ] **Step 3: `home()`에서 쿠키 발급**

`app/main.py:54-56`을 아래로 교체한다. `/predict`는 읽기만 하므로 발급은 여기 한 곳뿐이다.

```python
@app.route('/')
def home():
    resp = make_response(render_template('finally.html'))                       # 🆕 [진단 이력 기능] 수정된 줄 (2026-07-31)
    if not request.cookies.get(CLIENT_ID_COOKIE):                              # 🆕 [진단 이력 기능] 추가된 줄 — 상수는 파일 하단 블록
        resp.set_cookie(CLIENT_ID_COOKIE, uuid.uuid4().hex,                    # 🆕 [진단 이력 기능] 추가된 줄
                        max_age=CLIENT_ID_MAX_AGE, httponly=True, samesite='Lax')  # 🆕 [진단 이력 기능] 추가된 줄
    return resp                                                                 # 🆕 [진단 이력 기능] 수정된 줄
```

- [ ] **Step 4: `log_document`에 필드 3개 추가**

`app/main.py`의 `log_document` 딕셔너리에서 `"origin_id": ObjectId(),` **바로 다음 줄**에
아래 3줄을 넣는다. 딕셔너리 안 순수 추가이므로 `predict()`의 다른 줄은 건드리지 않는다.

```python
        "client_id": request.cookies.get(CLIENT_ID_COOKIE),  # 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)
        "origin_file_name": unique_filename,                 # 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)
        "origin_save_path": file_path,                       # 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)
```

- [ ] **Step 5: 서버를 띄우고 쿠키 확인**

서버를 기동하고 `http://localhost:5000` 접속 후 DevTools → Application → Cookies:

기대 — `hn_client_id`가 32자 hex, `HttpOnly` 체크, `Expires`가 **약 2일 뒤**.

- [ ] **Step 6: 진단 1건을 돌리고 DB 필드 확인**

화면에서 사진 1장을 업로드해 진단을 끝낸 뒤:

```bash
python -c "from pymongo import MongoClient; d=MongoClient('mongodb://localhost:27017/')['apartment_inspection_db']['inspection_logs'].find_one(sort=[('create_at',-1)]); print({k:d.get(k) for k in ['client_id','origin_file_name','origin_save_path','status']})"
```

기대 — `client_id`가 브라우저 쿠키 값과 같고, `origin_save_path`가
`static\images\origin\<hex>.jpg` 형태이며, `origin_file_name`이 그 파일명과 일치한다.
`client_id`가 `None`이면 Step 3의 쿠키 발급이 안 된 것이다.

- [ ] **Step 7: 기존 화면이 안 깨졌는지 확인**

같은 진단 화면에서 불량 판정이면 before/after 슬라이더와 히트맵/박스 탭이 예전처럼
동작하는지 본다. 기대 — 변화 없음. `predict()`를 안 건드렸으므로 달라지면 안 된다.

- [ ] **Step 8: 커밋**

```bash
git add app/main.py
git commit -m "이력 기능: client_id 쿠키 발급과 원본 경로 저장 추가"
```

---

## Task 2: `GET /history` — 목록 JSON

**Files:**
- Modify: `app/main.py` (Task 1이 만든 하단 블록 안, `여기까지` 마커 앞)

**Interfaces:**
- Consumes: `CLIENT_ID_COOKIE`, `HISTORY_LIMIT` (Task 1)
- Produces: `GET /history` → `{"items": [{"id": str, "date": str, "status": str, "thumb": str|null}]}`

- [ ] **Step 1: 라우트 작성**

Task 1의 하단 블록 안, `# ── 🆕 [진단 이력 기능] 여기까지 ──` 주석 **바로 앞**에 넣는다.

```python
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
```

- [ ] **Step 2: 쿠키를 달고 요청해 목록 확인**

Task 1에서 확인한 쿠키 값을 넣어 실행한다 (`<쿠키값>`을 실제 32자 hex로 교체).

```bash
curl -s -b "hn_client_id=<쿠키값>" http://localhost:5000/history
```

기대 — Task 1에서 돌린 진단 1건이 들어 있는 JSON. `date`가 `07/31 14:22` 형태,
`thumb`이 `/static/images/origin/<hex>.jpg`, `status`가 `우수` 또는 `불량`.

- [ ] **Step 3: 쿠키 없이 요청하면 빈 목록인지 확인**

```bash
curl -s http://localhost:5000/history
```

기대 — 정확히 `{"items":[]}`. 에러나 500이 아니어야 한다.

- [ ] **Step 4: 썸네일 URL이 실제로 열리는지 확인**

Step 2가 준 `thumb` 값을 그대로 브라우저 주소창에 붙여 넣는다
(예: `http://localhost:5000/static/images/origin/<hex>.jpg`).

기대 — 업로드했던 원본 사진이 뜬다. 404면 역슬래시 변환이 잘못된 것이다.

- [ ] **Step 5: DB를 끈 상태에서도 빈 목록인지 확인**

관리자 PowerShell에서:

```bash
Stop-Service MongoDB
```

그 뒤 Step 2를 다시 실행한다. 기대 — 빈 목록 `{"items":[]}`, 서버는 살아 있음.
확인이 끝나면 반드시 되살린다:

```bash
Start-Service MongoDB
```

- [ ] **Step 6: 커밋**

```bash
git add app/main.py
git commit -m "이력 기능: GET /history 목록 라우트 추가"
```

---

## Task 3: `GET /history/<id>` — 결과 조각 복원

**Files:**
- Modify: `app/main.py` (하단 블록 안, `history_list` 다음)

**Interfaces:**
- Consumes: `CLIENT_ID_COOKIE` (Task 1), `GET /history`가 준 `id` (Task 2)
- Produces: `GET /history/<item_id>` → `/predict`와 동일한 `f_result.html` 조각 HTML (200) 또는 404

- [ ] **Step 1: 라우트 작성**

`history_list` 함수 **바로 다음**, `여기까지` 마커 앞에 넣는다.

```python
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
```

- [ ] **Step 2: 조각이 돌아오는지 확인**

Task 2 Step 2의 JSON에서 `id`를 하나 꺼내 쓴다.

```bash
curl -s -b "hn_client_id=<쿠키값>" http://localhost:5000/history/<id>
```

기대 — `<footer class="result-panel">`로 시작하는 HTML이 오고, 마지막에
`<div id="backendUrls" data-origin="/static/images/origin/..." data-result="..."
data-heatmap="..." data-status="불량" ...>`이 들어 있다.
`data-origin`이 비어 있으면 `origin_save_path`가 저장 안 된 레코드다 (Task 1 이전 것).

- [ ] **Step 3: 쿠키 없이 요청하면 404인지 확인**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5000/history/<id>
```

기대 — `404`.

- [ ] **Step 4: 남의 쿠키로 요청하면 404인지 확인**

```bash
curl -s -o /dev/null -w "%{http_code}\n" -b "hn_client_id=deadbeefdeadbeefdeadbeefdeadbeef" http://localhost:5000/history/<id>
```

기대 — `404`. 200이 오면 소유자 검사가 빠진 것이다.

- [ ] **Step 5: 잘못된 id 형식이 404인지 확인**

```bash
curl -s -o /dev/null -w "%{http_code}\n" -b "hn_client_id=<쿠키값>" http://localhost:5000/history/not-an-objectid
```

기대 — `404`. 500이면 `ObjectId()` 예외 처리가 빠진 것이다.

- [ ] **Step 6: 커밋**

```bash
git add app/main.py
git commit -m "이력 기능: GET /history/<id> 결과 복원 라우트 추가"
```

---

## Task 4: 사이드바 마크업과 스타일

**Files:**
- Modify: `app/templates/finally.html` (`.system-container` 바로 앞)
- Modify: `app/templates/style.css` (파일 맨 끝)

**Interfaces:**
- Produces: DOM id `historyToggle`, `historyScrim`, `historySidebar`, `historyList` — Task 5·6이 잡는다
- Produces: CSS 클래스 `history-item`, `history-thumb`, `history-thumb-x`, `history-date`, `history-badge`(+`bad`/`good`/`err`), `history-empty`, 사이드바 열림 상태 `.open`

- [ ] **Step 1: `finally.html`에 마크업 추가**

`<body>` 다음 줄, `<div class="system-container">` **바로 앞**에 넣는다.
기존 워크스페이스 내부를 건드리지 않는 것이 요점이다.

```html
    <!-- 🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31) -->
    <button type="button" id="historyToggle" class="history-toggle" aria-label="진단 이력 열기">☰</button>
    <div id="historyScrim" class="history-scrim hide"></div>
    <aside id="historySidebar" class="history-sidebar" aria-label="진단 이력">
        <header class="history-header">진단 이력</header>
        <ul id="historyList" class="history-list"></ul>
    </aside>
    <!-- 🆕 [진단 이력 기능] 여기까지 -->
```

- [ ] **Step 2: `style.css` 맨 끝에 스타일 추가**

기존 다크 네온 테마(`#0d1527` 패널 / `#1e293b` 테두리 / `#00f2fe` 강조)를 따른다.
`.hide`는 style.css:109에 이미 있으므로 새로 만들지 않는다.

```css
/* ── 🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31) ── */
/* 좌측 진단 이력 사이드바. 오버레이 방식이라 .workspace 레이아웃을 밀지 않는다 */
.history-toggle { position: fixed; top: 16px; left: 16px; z-index: 60; width: 40px; height: 40px; background: #0d1527; color: #00f2fe; border: 1px solid #1e293b; border-radius: 6px; font-size: 18px; cursor: pointer; }
.history-toggle:hover { border-color: #00f2fe; }
.history-scrim { position: fixed; inset: 0; z-index: 50; background: rgba(0, 0, 0, 0.5); }
.history-sidebar { position: fixed; top: 0; left: 0; z-index: 55; width: 260px; height: 100vh; background: #0d1527; border-right: 1px solid #1e293b; display: flex; flex-direction: column; transform: translateX(-100%); transition: transform 0.25s ease; }
.history-sidebar.open { transform: translateX(0); }
.history-header { flex: 0 0 auto; padding: 18px 16px; font-size: 15px; color: #00f2fe; border-bottom: 1px solid #1e293b; }
/* 20건이 넘치면 여기서 스크롤된다 */
.history-list { flex: 1 1 auto; list-style: none; margin: 0; padding: 0; overflow-y: auto; }
.history-item { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-bottom: 1px solid #1e293b; cursor: pointer; }
.history-item:hover { background: #131c33; }
/* 오류 건과 파일이 사라진 건은 클릭 대상이 아니다 */
.history-item.is-error { cursor: default; opacity: 0.6; }
.history-item.is-error:hover { background: transparent; }
.history-thumb { flex: 0 0 56px; width: 56px; height: 56px; object-fit: cover; border-radius: 4px; background: #1e293b; }
.history-thumb-x { flex: 0 0 56px; width: 56px; height: 56px; border-radius: 4px; background: #1e293b; display: flex; align-items: center; justify-content: center; color: #64748b; font-size: 22px; }
.history-date { font-size: 13px; color: #a5b4fc; }
.history-badge { margin-left: auto; font-size: 12px; padding: 2px 8px; border-radius: 4px; }
.history-badge.bad { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
.history-badge.good { background: rgba(16, 185, 129, 0.15); color: #10b981; }
.history-badge.err { background: rgba(100, 116, 139, 0.15); color: #94a3b8; }
.history-empty { padding: 20px 16px; font-size: 13px; color: #64748b; }
/* ── 🆕 [진단 이력 기능] 여기까지 ── */
```

- [ ] **Step 3: 브라우저에서 버튼과 사이드바 위치 확인**

`http://localhost:5000` 새로고침. 기대 — 좌상단에 ☰ 버튼이 보이고, 사이드바는
화면 밖에 숨어 있다. 기존 업로드 존과 뷰어 패널의 위치는 그대로여야 한다.

- [ ] **Step 4: DevTools 콘솔에서 열림 상태를 강제로 확인**

아직 JS가 없으므로 콘솔에서 직접 켠다.

```javascript
document.getElementById('historySidebar').classList.add('open');
document.getElementById('historyScrim').classList.remove('hide');
```

기대 — 사이드바가 왼쪽에서 부드럽게 밀려 나오고(260px 폭), 뒤가 반투명 막으로 덮인다.
`.workspace`의 좌측 조작반이 밀려나지 않아야 한다.

- [ ] **Step 5: 행 스타일을 임시 마크업으로 확인**

같은 콘솔에서:

```javascript
document.getElementById('historyList').innerHTML =
  '<li class="history-item"><div class="history-thumb-x">✕</div>' +
  '<span class="history-date">07/31 14:22</span>' +
  '<span class="history-badge bad">불량</span></li>' +
  '<li class="history-item"><div class="history-thumb-x">✕</div>' +
  '<span class="history-date">07/30 09:13</span>' +
  '<span class="history-badge good">우수</span></li>';
```

기대 — 56px 정사각형 + 날짜 + 우측 배지가 한 줄에 정렬되고, 행에 마우스를 올리면
배경이 밝아진다. 여기서 56px이 작아 보이면 **이 단계에서 크기를 조정하고 넘어간다.**

- [ ] **Step 6: 커밋**

```bash
git add app/templates/finally.html app/templates/style.css
git commit -m "이력 기능: 사이드바 마크업과 스타일 추가"
```

---

## Task 5: 목록 조회와 렌더

**Files:**
- Modify: `app/templates/script.js` (`DOMContentLoaded` 콜백 맨 끝, 마지막 `});` 앞)

**Interfaces:**
- Consumes: `GET /history` (Task 2), DOM id·CSS 클래스 (Task 4)
- Produces: 함수 `openHistory()`, `closeHistory()`, `renderHistory(items)` — Task 6이 `closeHistory()`를 쓴다
- Produces: `restoreFromHistory(id)` 호출 지점 (Task 6에서 구현)

- [ ] **Step 1: Task 6에서 채울 자리를 임시로 막아두는 스텁과 함께 블록 작성**

`resetSystem()` 함수 **다음**, `DOMContentLoaded` 콜백을 닫는 마지막 `});` **바로 앞**에
넣는다. 파일 상단 `const` 선언부는 건드리지 않는다.

```javascript
    // ── 🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31) ──
    // 좌측 진단 이력 사이드바. 상단 const 선언부를 건드리지 않으려고
    // 필요한 요소를 이 블록 안에서 따로 잡는다 (팀 병합 충돌 최소화).
    const historyToggle = document.getElementById('historyToggle');
    const historySidebar = document.getElementById('historySidebar');
    const historyScrim = document.getElementById('historyScrim');
    const historyList = document.getElementById('historyList');

    // status 문자열 → [배지 클래스, 표시 문구]
    const HISTORY_BADGE = { '불량': ['bad', '불량'], '우수': ['good', '우수'], '오류': ['err', '오류'] };

    function closeHistory() {
        historySidebar.classList.remove('open');
        historyScrim.classList.add('hide');
    }

    /** 썸네일 자리에 들어갈 회색 X 상자 */
    function makeThumbX() {
        const box = document.createElement('div');
        box.className = 'history-thumb-x';
        box.textContent = '✕';
        return box;
    }

    function renderHistory(items) {
        historyList.innerHTML = '';

        if (!items.length) {
            const empty = document.createElement('li');
            empty.className = 'history-empty';
            empty.textContent = '아직 진단 이력이 없습니다.';
            historyList.appendChild(empty);
            return;
        }

        items.forEach(item => {
            const isError = item.status === '오류';
            const row = document.createElement('li');
            row.className = 'history-item' + (isError ? ' is-error' : '');

            if (isError || !item.thumb) {
                row.appendChild(makeThumbX());
            } else {
                const thumb = document.createElement('img');
                thumb.className = 'history-thumb';
                thumb.src = item.thumb;
                thumb.alt = '';
                // 서버에서 파일이 지워졌으면 X 상자로 갈아끼우고 클릭도 막는다
                thumb.addEventListener('error', () => {
                    thumb.replaceWith(makeThumbX());
                    row.classList.add('is-error');
                }, { once: true });
                row.appendChild(thumb);
            }

            const date = document.createElement('span');
            date.className = 'history-date';
            date.textContent = item.date;
            row.appendChild(date);

            const badge = document.createElement('span');
            const badgeSpec = HISTORY_BADGE[item.status] || HISTORY_BADGE['오류'];
            badge.className = 'history-badge ' + badgeSpec[0];
            badge.textContent = badgeSpec[1];
            row.appendChild(badge);

            if (!isError) {
                row.addEventListener('click', () => restoreFromHistory(item.id));
            }
            historyList.appendChild(row);
        });
    }

    function openHistory() {
        historySidebar.classList.add('open');
        historyScrim.classList.remove('hide');
        // 열 때마다 새로 받는다 — 방금 끝낸 진단이 바로 목록에 보인다
        fetch('/history')
            .then(res => res.json())
            .then(data => renderHistory(data.items || []))
            .catch(() => renderHistory([]));
    }

    // Task 6에서 실제 복원 로직으로 교체한다
    function restoreFromHistory(id) {
        console.log('restore', id);
    }

    historyToggle.addEventListener('click', () => {
        if (historySidebar.classList.contains('open')) closeHistory();
        else openHistory();
    });
    historyScrim.addEventListener('click', closeHistory);
    // ── 🆕 [진단 이력 기능] 여기까지 ──
```

- [ ] **Step 2: 열고 닫기 확인**

새로고침 후 ☰ 클릭. 기대 — 사이드바가 열리고 Task 1에서 돌린 진단이 목록에 뜬다
(56px 썸네일 + `MM/DD HH:MM` + 배지). ☰ 다시 클릭 또는 어두운 막 클릭 → 닫힌다.

- [ ] **Step 3: 새 진단이 즉시 목록에 붙는지 확인**

사이드바를 닫고 사진 1장을 새로 진단한 뒤 다시 ☰ 클릭.
기대 — 방금 건이 목록 **최상단**에 있다.

- [ ] **Step 4: 빈 목록 문구 확인**

DevTools → Application → Cookies에서 `hn_client_id`를 삭제하고 새로고침한 뒤 ☰ 클릭.
기대 — "아직 진단 이력이 없습니다."가 뜬다.

(새로고침 시 `/`가 새 쿠키를 발급하므로, 이후 진단은 새 `client_id`로 쌓인다.)

- [ ] **Step 5: 파일이 없을 때 X로 바뀌는지 확인**

`app/static/images/origin/`에서 목록에 보이는 사진 파일 하나를 지우고 ☰ 를 다시 연다.
기대 — 그 행만 회색 X 상자로 바뀌고 흐려지며, 클릭해도 반응이 없다.

- [ ] **Step 6: 오류 건 표시 확인**

DB에 오류 문서를 직접 하나 넣는다 (`<쿠키값>`을 현재 쿠키로 교체).

```bash
python -c "from pymongo import MongoClient; from datetime import datetime; MongoClient('mongodb://localhost:27017/')['apartment_inspection_db']['inspection_logs'].insert_one({'client_id':'<쿠키값>','status':'오류','create_at':datetime.now()}); print('inserted')"
```

☰ 를 다시 연다. 기대 — X 썸네일 + 회색 "오류" 배지, 클릭 무반응.

- [ ] **Step 7: 커밋**

```bash
git add app/templates/script.js
git commit -m "이력 기능: 사이드바 목록 조회와 렌더 추가"
```

---

## Task 6: 클릭 시 과거 결과 복원

**Files:**
- Modify: `app/templates/script.js` (Task 5가 만든 `restoreFromHistory` 스텁 교체)

**Interfaces:**
- Consumes: `GET /history/<id>` (Task 3), `closeHistory()` (Task 5)
- Consumes: 기존 `injectBackendResult(htmlContent)` (script.js:242) — **수정하지 않고 그대로 호출**
- Consumes: 기존 외부 스코프 변수 `uploadZone`, `logZone`, `laserLine`, `progressBar`, `progressText`

- [ ] **Step 1: 스텁을 실제 구현으로 교체**

Task 5에서 넣은 아래 스텁을

```javascript
    // Task 6에서 실제 복원 로직으로 교체한다
    function restoreFromHistory(id) {
        console.log('restore', id);
    }
```

아래로 바꾼다.

```javascript
    /** 저장된 진단 1건을 화면에 되살린다.
     *  서버가 /predict와 똑같은 조각을 주므로 기존 injectBackendResult()를 그대로 쓴다
     *  — 슬라이더·레이어 탭·fitCompareWrap()이 거기서 전부 세팅된다. */
    function restoreFromHistory(id) {
        fetch('/history/' + id)
            .then(res => {
                if (!res.ok) throw new Error('이력을 불러오지 못했습니다. 이미지가 삭제되었을 수 있습니다.');
                return res.text();
            })
            .then(htmlResult => {
                closeHistory();
                // btnDiagnose 클릭 핸들러(script.js:195-201)와 같은 화면 전환
                uploadZone.classList.add('hide');
                logZone.classList.remove('hide');
                laserLine.classList.add('hide');
                progressBar.style.width = '100%';
                progressText.textContent = '[ANALYZING... 100%]';
                injectBackendResult(htmlResult);
            })
            .catch(err => alert(err.message));
    }
```

- [ ] **Step 2: 불량 건 복원 확인**

☰ → 목록에서 **불량** 항목 클릭.

기대 —
1. 사이드바가 닫힌다
2. 업로드 존이 사라지고 로그 존이 뜬다
3. 뷰어에 before/after 슬라이더가 나타나고 분할선을 끌면 원본↔히트맵이 비교된다
4. 상단 `히트맵` / `결함 박스` 탭이 보이고 전환된다
5. 하단에 판정·확률 막대·추론 속도가 뜬다
6. 버튼이 `다시 진단하기`로 바뀐다

- [ ] **Step 3: 우수 건 복원 확인**

목록에서 **우수** 항목 클릭.
기대 — EXCELLENT 스탬프 이미지 **한 장만** 뜬다. 슬라이더와 탭은 안 보인다.

- [ ] **Step 4: 복원 후 새 진단이 정상인지 확인**

복원된 상태에서 `다시 진단하기`를 누르고 사진을 새로 올려 진단한다.
기대 — 평소와 똑같이 동작한다. 복원이 기존 상태 머신을 망가뜨리지 않아야 한다.

- [ ] **Step 5: 404 처리 확인**

DevTools 콘솔에서 없는 id로 직접 호출한다.

```javascript
restoreFromHistory('000000000000000000000000');
```

기대 — "이력을 불러오지 못했습니다..." alert가 뜨고 화면은 그대로 살아 있다.

- [ ] **Step 6: 커밋**

```bash
git add app/templates/script.js
git commit -m "이력 기능: 항목 클릭 시 과거 결과 복원 추가"
```

---

## Task 7: 문서 갱신

**Files:**
- Modify: `docs/PROJECT_MAP.md`, `docs/FRONTEND_GUIDE.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: Task 1~6의 최종 라우트·DOM id·DB 필드

- [ ] **Step 1: `docs/PROJECT_MAP.md` 라우트 표에 2줄 추가**

"### 라우트 (전부 `main.py`)" 표의 `/script.js` 행 아래에 넣는다.

```markdown
| `/history` | GET | 쿠키 주인의 최근 진단 20건을 JSON으로 반환 (오류 건 포함) |
| `/history/<id>` | GET | 저장된 진단 1건을 `/predict`와 **동일한 조각 HTML**로 반환. 남의 기록·오류 건은 404 |
```

- [ ] **Step 2: `docs/PROJECT_MAP.md`의 주요 DOM id 목록에 4개 추가**

"### 주요 DOM id (script.js가 잡는 것들)" 블록 끝에 한 줄 덧붙인다.

```markdown
`historyToggle` `historySidebar` `historyScrim` `historyList`
```

- [ ] **Step 3: `docs/PROJECT_MAP.md`의 외부 의존 항목에 DB 스키마 변경 반영**

"- **MongoDB**: DB `apartment_inspection_db`, 컬렉션 `inspection_logs`." 로 시작하는
줄 다음에 넣는다.

```markdown
  이력 기능용으로 `client_id`(쿠키), `origin_file_name`, `origin_save_path`가 함께 저장된다.
  이 세 필드가 없는 구 레코드는 이력 목록에 뜨지 않는다.
```

- [ ] **Step 4: `docs/FRONTEND_GUIDE.md`에 절 추가**

"## before/after 슬라이더 + 레이어 탭" 절 **앞**에 넣는다.

```markdown
## 진단 이력 (`/history`, `/history/<id>`)

`GET /history/<id>`는 `/predict`와 **완전히 같은 형식의 `f_result.html` 조각**을 반환합니다.
프론트는 그 응답을 기존 `injectBackendResult()`에 그대로 넘기면 되고, 슬라이더·레이어 탭
로직은 손댈 필요가 없습니다.

단 `peak_x` / `peak_y`는 항상 `None`입니다 — `f_result.html`이 쓰지 않아 DB에 저장하지
않기 때문입니다. 나중에 이 값을 화면에 표시하려면 `main.py`의 `log_document`에 저장부터
추가해야 합니다.

사용자 구분은 `hn_client_id` 쿠키(2일)로만 합니다. 로그인이 없으므로 쿠키를 지우면
이력이 사라집니다.
```

- [ ] **Step 5: `CLAUDE.md` 라우팅 표에 1줄 추가**

"| 라우트 추가 · 업로드 제한 · MongoDB 스키마 | `app/main.py` |" 행 아래에 넣는다.

```markdown
| 진단 이력 사이드바 (쿠키·목록·복원) | `app/main.py`의 `/history*` 라우트 + [docs/superpowers/specs/2026-07-31-history-sidebar-design.md](docs/superpowers/specs/2026-07-31-history-sidebar-design.md) |
```

- [ ] **Step 6: 문서와 코드가 어긋나지 않는지 확인**

`docs/PROJECT_MAP.md`에 적은 라우트 2개와 DOM id 4개가 실제 코드에 있는지 대조한다.

```bash
grep -n "history" app/main.py app/templates/script.js app/templates/finally.html
```

기대 — 문서에 적은 이름이 전부 코드에 존재한다.

- [ ] **Step 7: 커밋**

```bash
git add docs/PROJECT_MAP.md docs/FRONTEND_GUIDE.md CLAUDE.md
git commit -m "이력 기능: 문서 갱신 (라우트·DOM id·DB 스키마)"
```

---

## 최종 확인 (전체 Task 완료 후)

서버를 껐다 새로 띄우고 아래를 순서대로 확인한다.

- [ ] 진단 21건을 쌓은 뒤 ☰ → 목록이 **20건에서 끊기고** 스크롤된다
- [ ] 불량 건 복원 → 슬라이더·탭 정상
- [ ] 우수 건 복원 → 스탬프 한 장, 슬라이더 없음
- [ ] 쿠키 삭제 → 새로고침 → 빈 목록 문구
- [ ] 시크릿 창에서 앞서 쓴 `/history/<id>` 직접 요청 → 404
- [ ] `Stop-Service MongoDB` → ☰ → 빈 목록, 화면 살아 있음 → `Start-Service MongoDB`
- [ ] `git diff main --stat` 으로 변경 파일이 7개(`main.py`, `finally.html`, `style.css`, `script.js`, 문서 3개)인지 확인
- [ ] `git diff main -- app/main.py` 로 `predict()` 안에서 바뀐 것이 `log_document`의 3줄뿐인지 확인
