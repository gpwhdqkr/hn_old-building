# 진단 이력 사이드바 — 설계

작성 2026-07-31 · 대상 브랜치 `claude/backend-frontend-location-docs-426d09`

---

## 1. 배경과 목표

현재 서비스는 진단 결과를 한 번 보여주고 끝난다. 페이지를 새로 고치면 방금 본 결과를
다시 볼 방법이 없다. MongoDB(`inspection_logs`)에 기록은 쌓이고 있지만 조회 경로가 없다.

**목표** — 화면 좌측에 평소 숨어 있는 사이드바를 두고, 햄버거 버튼으로 열면
자기가 최근에 돌린 진단 목록이 뜬다. 항목을 누르면 그때의 결과 화면이 그대로 복원된다.

목록 한 행에 보이는 것: **56x56 썸네일(업로드한 원본 사진) · `MM/DD HH:MM` · 우수/불량/오류 배지.**
연도는 표시하지 않는다.

---

## 2. 전제와 제약

- **로그인·세션이 없다.** 사용자 구분은 쿠키 하나로만 한다. 쿠키를 지우거나 시크릿 창을
  쓰면 이력은 빈 목록이 된다. 서버에는 남지만 되찾을 방법이 없다 — 의도된 동작이다.
- **기존 레코드는 목록에 뜨지 않는다.** `client_id`도 원본 경로도 없기 때문이다.
  마이그레이션은 하지 않는다. 이 기능은 배포 시점부터 새로 쌓인다.
- **이미지는 서빙 PC의 로컬 파일이다** (`app/static/images/`, gitignore 대상).
  파일이 지워지면 DB 레코드만 남아 썸네일이 깨진다.
- **개발 PC에서 종단 검증이 안 된다.** MongoDB와 모델 가중치가 서빙 PC(192.168.0.22)에만
  있다. 5절 검증 항목 참고.
- **팀원과 같은 파일을 동시에 고친다.** 병합 충돌을 줄이는 것이 설계 제약이다 — 6절.

---

## 3. 데이터

### 3.1 MongoDB — `log_document`에 필드 3개 추가

[app/main.py:191](../../../app/main.py) 의 `log_document`에 아래를 더한다.

| 필드 | 값 | 용도 |
|---|---|---|
| `client_id` | 쿠키 `hn_client_id` 값 (없으면 `None`) | 이력 필터 |
| `origin_file_name` | `unique_filename` | — |
| `origin_save_path` | `file_path` (예: `static\images\origin\ab12.jpg`) | 썸네일 · 슬라이더 before |

`peak_x` / `peak_y`는 **저장하지 않는다.** `f_result.html`이 이 값을 쓰지 않아
결과 화면 복원에 필요가 없다.

확률 %도 저장하지 않는다. 이미 있는 `defect_probability`와 `status`로 계산된다.

### 3.2 쿠키

| 항목 | 값 |
|---|---|
| 이름 | `hn_client_id` |
| 값 | `uuid4().hex` |
| 만료 | `max_age=172800` (2일) |
| 속성 | `httponly=True`, `samesite='Lax'` |

`GET /`에서 쿠키가 없으면 발급한다. `/predict`는 쿠키를 읽어 저장하고, 쿠키가 없는
요청이면 그 응답에서 발급한 뒤 그 값으로 저장한다.

쿠키가 2일이므로 **이력도 사실상 이틀치**다. 사흘 뒤 접속하면 새 `client_id`가 발급되어
빈 목록이 된다.

---

## 4. 구현

### 4.1 백엔드 — 라우트 2개 (`app/main.py`)

**`GET /history`** → JSON

```json
{"items": [
  {"id": "68a1...", "date": "07/31 14:22", "status": "불량",
   "thumb": "/static/images/origin/ab12.jpg"}
]}
```

- 필터 `{"client_id": <쿠키값>}`, 정렬 `create_at` 내림차순, **20건**
- 쿠키가 없으면 빈 목록
- `status`가 `"오류"`인 문서도 **포함**한다 (프론트가 X 썸네일로 표시)
- `origin_save_path`가 없으면 `thumb`은 `null`
- DB 예외는 삼키고 빈 목록을 반환한다. 이력이 안 보이는 것과 화면이 죽는 것은 다르다
  — 기존 `insert_one` 격리 방침([main.py:208](../../../app/main.py))과 같은 태도다

**`GET /history/<id>`** → `f_result.html` 조각 (`/predict` 응답과 동일한 형식)

- `ObjectId(id)`로 조회. 형식이 잘못됐으면 404
- 문서의 `client_id`가 쿠키와 다르면 **404** (남의 기록 열람 차단)
- `status == "오류"`면 404 (복원할 결과 이미지가 없다)
- 렌더 인자는 DB 값에서 재구성한다:

  | 템플릿 변수 | 출처 |
  |---|---|
  | `user_image_url` | `origin_save_path` → URL 변환 |
  | `cam_image_url` | `save_path` → URL 변환 |
  | `heatmap_image_url` | `heatmap_save_path` → URL 변환 (없으면 `None`) |
  | `ai_result` | `status` |
  | `probability_percent` | `defect_probability` + `status`로 계산 |
  | `inference_ms` | `inference_time_ms` |
  | `peak_x`, `peak_y` | `None` 고정 (템플릿이 쓰지 않음) |

**공용 헬퍼 2개를 추출한다.** `predict()`와 `/history/<id>`가 같은 계산을 하므로,
복사해 두면 임계값이 바뀔 때 한쪽만 고쳐지는 종류의 버그가 난다.

- `to_web_path(path)` — 역슬래시 → 슬래시, 앞에 `/` 부착 ([main.py:169-173](../../../app/main.py)에서 이동)
- `to_probability_percent(defect_probability, is_defect)` — ([main.py:176-181](../../../app/main.py)에서 이동)

`predict()`는 이 헬퍼를 쓰도록 바꾼다. 동작은 동일하다.

### 4.2 프론트 — 마크업 (`app/templates/finally.html`)

`.system-container` **바로 앞**(= `<body>` 최상단)에 한 덩어리로 넣는다.
기존 워크스페이스 내부를 건드리지 않아야 팀원 변경과 겹치지 않는다.

```html
<button id="historyToggle">☰</button>
<div id="historyScrim"></div>
<aside id="historySidebar">
  <header>진단 이력</header>
  <ul id="historyList"></ul>
</aside>
```

### 4.3 프론트 — 스타일 (`app/templates/style.css`)

새 규칙은 **파일 맨 끝에 한 블록으로** 몰아 넣는다.

- `#historySidebar` — `position: fixed; left: 0; top: 0; height: 100vh; width: 260px;`
  기본 `transform: translateX(-100%)`, `.open`에서 `translateX(0)`. **오버레이 방식**이라
  기존 `.workspace` 레이아웃이 밀리지 않는다
- `#historyList` — `overflow-y: auto`. 20건이면 스크롤된다
- 행 — 56x56 썸네일(`object-fit: cover`) + 날짜 + 배지. 배지 색은 기존 테마를 따른다
  (불량 `#ef4444`, 우수 `#10b981`, 오류 회색)
- 배경·테두리는 기존 다크 네온 계열(`#0d1527` / `#1e293b`)을 쓴다
- `#historyScrim` — 열렸을 때만 보이는 반투명 막. 클릭하면 닫힌다

### 4.4 프론트 — 동작 (`app/templates/script.js`)

새 코드는 `DOMContentLoaded` 콜백 **맨 끝에 한 블록으로** 넣는다. 상단 `const` 선언부에
끼워 넣지 않고, 블록 안에서 자체적으로 `getElementById`를 한다.

- 햄버거 클릭 → 사이드바 열기 + `fetch('/history')` → 행 렌더.
  **열 때마다 새로 받는다** — 방금 끝낸 진단이 바로 목록에 보인다
- 행 클릭 → `fetch('/history/' + id)` → `res.text()` → `restoreFromHistory(html)`
- `restoreFromHistory(html)`:
  1. 사이드바를 닫는다
  2. `uploadZone.classList.add('hide')`, `logZone.classList.remove('hide')`
     — `btnDiagnose` 핸들러([script.js:195-201](../../../app/templates/script.js))와 같은 화면 전환
  3. **기존 `injectBackendResult(html)`를 그대로 호출한다**

  `injectBackendResult()`는 조각을 주입하고 `#backendUrls`를 읽어 슬라이더·탭·
  `fitCompareWrap()`까지 세팅한다. 진행률 애니메이션과 무관하게 동작하므로
  **이 함수도, 슬라이더 로직도 수정하지 않는다.**
- 오류 행은 클릭 이벤트를 걸지 않는다

### 4.5 X 대체 표시

두 경우에 회색 사각형 안에 X를 그린다. 둘 다 클릭이 안 먹는다.

1. `status == "오류"`인 행
2. 썸네일 `<img>`가 `onerror`를 낸 행 (파일이 지워졌거나 `thumb`이 `null`)

목록이 비었을 때는 "아직 진단 이력이 없습니다"를 표시한다.

---

## 5. 검증

**개발 PC에서 가능한 것**

- 가짜 JSON을 넣어 사이드바 열림/닫힘, 행 렌더, 56px 썸네일, 스크롤, X 대체 표시 확인
- 존재하지 않는 이미지 경로로 `onerror` 경로 확인
- 빈 목록 문구 확인

**서빙 PC(192.168.0.22)에서만 가능한 것**

- 진단 → 사이드바 열기 → 방금 항목이 목록 최상단에 보이는지
- 항목 클릭 → 불량 건에서 슬라이더·히트맵/박스 탭이 정상 복원되는지
- 항목 클릭 → 우수 건에서 스탬프 이미지 한 장만 뜨는지 (슬라이더 없음)
- 쿠키를 지운 뒤 목록이 비는지
- 다른 브라우저에서 남의 `id`로 `/history/<id>` 직접 요청 시 404인지
- MongoDB를 끈 상태에서 사이드바를 열어도 화면이 죽지 않는지

자동 테스트는 만들지 않는다. 저장소에 테스트 하네스가 없고, 검증에 필요한 DB와 모델이
개발 PC에 없다.

---

## 6. 주석 규약 — 병합 충돌 대비

팀원이 같은 파일을 동시에 고치고 있다. 두 가지로 대응한다.

### 6.1 새 코드는 파일 끝(또는 기존 구조 바깥)에 몰아 넣는다

| 파일 | 넣는 위치 |
|---|---|
| `main.py` | `if __name__ == '__main__':` **직전** |
| `style.css` | 파일 맨 끝 |
| `script.js` | `DOMContentLoaded` 콜백 맨 끝 |
| `finally.html` | `.system-container` 바로 앞 |

기존 함수 안이나 선언부 중간에 끼워 넣지 않는다. 겹치는 줄이 없으면 git이 알아서 병합한다.

### 6.2 마커 주석

새로 추가한 덩어리는 시작과 끝을 감싼다.

```python
# ── 🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31) ──────────────────
...
# ── 🆕 [진단 이력 기능] 여기까지 ────────────────────────────────────
```

```html
<!-- 🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31) -->
...
<!-- 🆕 [진단 이력 기능] 여기까지 -->
```

```css
/* ── 🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31) ── */
```

```javascript
// ── 🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31) ──
```

기존 줄을 **고친** 자리는 감쌀 수 없으므로 줄 끝에 인라인으로 표시한다.

```python
"origin_save_path": file_path,   # 🆕 [진단 이력 기능] 추가된 줄 (2026-07-31)
```

기존 줄을 고치는 곳은 세 군데뿐이다 — `log_document` 딕셔너리(필드 3개 추가),
`predict()`의 URL·확률 계산부(헬퍼 호출로 교체), `home()`(쿠키 발급).

---

## 7. 변경 파일 요약

| 파일 | 변경 |
|---|---|
| `app/main.py` | 쿠키 발급·읽기, `log_document` 필드 3개, 헬퍼 2개 추출, `/history`·`/history/<id>` |
| `app/templates/finally.html` | 햄버거 버튼 · 사이드바 · 스크림 마크업 |
| `app/templates/style.css` | 사이드바 스타일 (파일 끝 블록) |
| `app/templates/script.js` | 목록 조회·렌더·복원 (콜백 끝 블록) |
| `docs/PROJECT_MAP.md` | 라우트 표에 `/history` 2개, DB 스키마 항목 갱신 |
| `docs/FRONTEND_GUIDE.md` | `/history/<id>`가 `/predict`와 같은 조각을 반환한다는 사실 추가 |
| `CLAUDE.md` | 라우팅 표에 이력 기능 한 줄 |

`app/templates/f_result.html`, `app/ai_engine.py`, `app/cv_processor.py`는 **건드리지 않는다.**
