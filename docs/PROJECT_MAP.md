# 프로젝트 구조 지도 — 백엔드 / 프론트엔드 위치

작업 시작 전에 이 파일만 읽으면 어디를 열어야 하는지 알 수 있도록 만든 색인이다.
파일 탐색에 쓰는 토큰을 줄이는 것이 목적이므로, **구조가 바뀌면 이 파일도 같이 고친다.**

최종 갱신: 2026-07-31

---

## 한 줄 요약

- **웹 서비스(백엔드 + 프론트) 전부 `app/` 안에 있다.**
- **프론트는 `app/templates/` 안에 HTML·CSS·JS가 전부 같이 들어 있다** (`static/` 폴더 아님 — 주의).
- `src/`, `model/`, `data/`는 웹 서비스가 아니라 **모델 학습·전처리 쪽**이다. 웹 작업 시 볼 필요 없다.

---

## 백엔드 (Flask)

| 파일 | 역할 |
|---|---|
| [app/main.py](../app/main.py) | **진입점.** Flask 앱, 라우트, 업로드 방어가드, MongoDB 저장, 템플릿 렌더링 |
| [app/ai_engine.py](../app/ai_engine.py) | 추론 엔진. `ApartmentClassifier`, `LayerCAM`, 임계값 판정 |
| [app/cv_processor.py](../app/cv_processor.py) | OpenCV 시각화. 결함 박스 / 히트맵 오버레이 / EXCELLENT 스탬프 |

### 라우트 (전부 `main.py`)

| 경로 | 메서드 | 하는 일 |
|---|---|---|
| `/` | GET | `templates/finally.html` 렌더 (메인 화면) |
| `/predict` | POST | `house_image` 업로드 → 추론 → `templates/f_result.html` **조각 HTML** 반환 |
| `/style.css` | GET | `templates/style.css`를 강제 서빙 (아래 "정적 파일" 항목 참고) |
| `/script.js` | GET | `templates/script.js`를 강제 서빙 |
| `/history` | GET | 쿠키 주인의 최근 진단 20건을 JSON으로 반환 (오류 건 포함) |
| `/history/<id>` | GET | 저장된 진단 1건을 `/predict`와 **동일한 조각 HTML**로 반환. 남의 기록·오류 건은 404 |

서버 기동: `python app/main.py` → `0.0.0.0:5000`, `debug=True`.

### 백엔드 주요 상수 (튜닝 시 여기부터)

| 상수 | 위치 | 기본값 |
|---|---|---|
| `OPERATING_THRESHOLD` | `ai_engine.py:29` | 0.50 (`MODEL_THRESHOLD` 환경변수로 조정) |
| `INPUT_SIZE` / `EVAL_RESIZE_SHORT` | `ai_engine.py:23-24` | 448 / 512 (224 아님) |
| `BOX_THRESHOLD` | `cv_processor.py:11` | 0.5 (박스 민감도) |
| `RELATIVE_THRESHOLD_RATIO` | `cv_processor.py:16` | 0.7 |
| `HEATMAP_ALPHA` | `cv_processor.py:27` | 0.45 |
| `MONGO_URI` | `main.py:48` | `mongodb://localhost:27017/` (환경변수 우선) |
| 모델 가중치 경로 | `main.py:36` | `model/best_convnext_tiny_binclf_v3_finetuned_v3cta.pth` |

### 외부 의존

- **모델 가중치**: `model/*.pth`는 `.gitignore` 대상이라 저장소에 없다. 서빙 PC에 파일이 실제로 있어야 기동된다.
- **MongoDB**: DB `apartment_inspection_db`, 컬렉션 `inspection_logs`. 저장 실패해도 화면 출력은 계속되도록 try/except로 격리돼 있다 (`main.py:215-219`).
  이력 기능용으로 `client_id`(쿠키 `hn_client_id`, 2일), `origin_file_name`, `origin_save_path`가 함께 저장된다.
  이 세 필드가 없는 구 레코드는 이력 목록에 뜨지 않는다.
  ⚠️ MongoDB가 죽으면 `MongoClient`의 기본 `serverSelectionTimeoutMS`(30초) 때문에 `/history`와 `insert_one`이
  30초씩 블로킹된 뒤에야 넘어간다. 고치려면 `main.py:49`에 `serverSelectionTimeoutMS`를 주면 된다.
- **배포 위치**: 웹서비스는 별도 PC(192.168.0.22)에서 구동된다. DB도 그쪽 localhost라 개발 PC에서는 종단 검증 불가.

---

## 프론트엔드

**전부 `app/templates/` 안에 있다.** CSS·JS도 `static/`이 아니라 `templates/` 안에 같이 들어 있고,
`main.py`의 `/style.css`, `/script.js` 라우트가 이를 억지로 서빙해 준다 (`main.py:24-32`).
→ **`app/static/`에서 CSS/JS를 찾지 말 것.** `app/static/images/`는 런타임 업로드·결과 이미지 저장소일 뿐이고 `.gitignore` 대상이다.

### 현재 사용 중인 파일

| 파일 | 역할 |
|---|---|
| [app/templates/finally.html](../app/templates/finally.html) | **메인 화면 셸.** 업로드 존, 진행 로그, 뷰어 패널, before/after 비교 슬라이더, 레이어 탭 |
| [app/templates/script.js](../app/templates/script.js) | **모든 프론트 로직.** 드래그앤드롭, `fetch('/predict')`, 진행률 애니메이션, 결과 HTML 주입, 비교 슬라이더, 히트맵/박스 탭 전환 |
| [app/templates/style.css](../app/templates/style.css) | 전체 스타일 (다크 네온 테마). 줄마다 한글 주석 있음 |
| [app/templates/f_result.html](../app/templates/f_result.html) | **결과 조각(fragment).** 전체 페이지가 아니라 `/predict` 응답으로 오는 부분 HTML |

### 동작 흐름 (SPA 유사 구조 — 페이지 이동 없음)

1. 사용자가 `finally.html`에서 파일 선택/드롭 → `script.js`가 미리보기 표시
2. "진단 시작" 클릭 → `fetch('/predict', {FormData: house_image})`
3. 서버가 `f_result.html` **조각**을 문자열로 반환
4. `script.js`의 `injectBackendResult()`가 `#dynamicResult`에 `innerHTML`로 주입
5. 조각 안의 `<div id="backendUrls" data-origin/data-result/data-heatmap/data-status>`(`f_result.html:52`)를 읽어 이미지 URL을 뽑고 비교 슬라이더를 세팅

> 방어가드 위반 시 서버는 `<script>alert(...)</script>` 문자열을 반환하고,
> `script.js:226-230`이 이를 감지해 `alert()`로 띄운 뒤 초기화한다. JSON API가 아니다.

### 주요 DOM id (script.js가 잡는 것들)

`uploadZone` `uploadBox` `fileInput` `uploadText` `btnDiagnose` `systemStatus` `logZone`
`progressBar` `progressText` `laserLine` `imageViewport` `previewImg` `resultImg` `gridBg`
`dynamicResult` `compareWrap` `compareAfter` `compareDivider` `beforeImg` `afterImg` `afterTag`
`layerTabs` `tabHeatmap` `tabBox` `backendUrls`
`historyToggle` `historySidebar` `historyScrim` `historyList`

### 템플릿 변수 계약

`/predict`가 `f_result.html`로 넘기는 템플릿 변수의 상세 규격은
[docs/FRONTEND_GUIDE.md](FRONTEND_GUIDE.md)에 있다. **프론트 작업 시 이 파일을 같이 읽는다.**
변수 6종 + before/after 슬라이더·레이어 탭 구현 주의사항까지 여기 하나로 통합돼 있다.
(구버전 사본이던 `app/FRONTEND_GUIDE.md`는 2026-07-31에 삭제했다.)

---

## 쓰지 않는(레거시) 파일 — 열지 말 것

| 파일 | 상태 |
|---|---|
| `app/finally.py` | 정적 서빙 전용 미니 서버. 현재 진입점은 `main.py` |
| `app/scan.py` | `scan.html`만 띄우던 초기 실험 서버 |
| `app/templates/index.html` | 구 메인 화면. 어떤 라우트도 렌더하지 않음 |
| `app/templates/result.html` | 구 결과 페이지. 현재는 `f_result.html` 사용 |
| `app/templates/scan.html` | `scan.py` 전용 |
| `app/templates/isnull.html` | 빈 파일 |

---

## 웹 서비스가 아닌 영역 (웹 작업 시 무시)

| 경로 | 내용 |
|---|---|
| `src/` | 전처리·학습·평가 스크립트 |
| `src/v3/` | **현재 서비스 모델(ConvNeXt-Tiny binclf_v3)의 학습/평가/LayerCAM 코드** |
| `src/binclf/` | 이원화(우수/불량) 실험 — efficientnet / mobilenet / resnet |
| `src/legacy_ml/`, `src/v2/` | 구버전 파이프라인 |
| `model/` | 학습 가중치 (`.pth`, 대부분 gitignore) |
| `data/` | 원본·전처리 데이터 |
| `docs/` | 사용설명서, 멘토 피드백, 작업 기록 |
| `runpod_test/`, `ppt_data/`, `test_results/` | 실험·산출물 보관 |

---

## 문서 색인 (저장소 내 md 전부)

| 파일 | 내용 |
|---|---|
| [CLAUDE.md](../CLAUDE.md) | 루트 길잡이. "하려는 일 → 읽을 곳" 라우팅 표. 세션 시작 시 자동 로드 |
| **`docs/PROJECT_MAP.md`** | **이 파일.** 구조 상세 지도 — 라우팅으로 부족할 때 |
| [docs/FRONTEND_GUIDE.md](FRONTEND_GUIDE.md) | `/predict` 응답 변수 6종, 슬라이더/탭 구현 규약. **웹 작업 필독** |
| [docs/노후건물_데이터수집_모델링결과.md](노후건물_데이터수집_모델링결과.md) | AI-Hub 데이터 수집 명세(37,285장)와 전체 실험 결과 정리. 보고서용 |
| [docs/ResNet_사용설명서.md](ResNet_사용설명서.md) | `src/resnet/` 학습·평가 5스크립트 사용법 (런팟 기준). 구버전 실험 |
| [src/v3/README.md](../src/v3/README.md) | **현재 서비스 모델(binclf_v3) 학습 파이프라인.** 재학습 시 필독 |
| [src/v2/README.md](../src/v2/README.md) | v2 파이프라인(binclf_v2 + midclf_v2). v3의 전신 |
| [src/binclf/README.md](../src/binclf/README.md) | 최초 이원화 실험(3클래스 → 2클래스 병합). 라벨 정의 근거 |
| [runpod_test/README.md](../runpod_test/README.md) | RunPod GPU 대여·반납 테스트 템플릿 |
| [test_results/summary_all_tests.md](../test_results/summary_all_tests.md) | 전 실험 test 성능 비교표 (동일 5,558장 기준) |

md가 아닌 참고 자료: `docs/EfficientNet_사용설명서.pdf`, `docs/resnet_작업순서.txt`,
`docs/김민권 기록0.txt`, `docs/mento_feedback/second feedback.txt`.

---

## 작업 유형별 열어야 할 파일

| 하려는 일 | 열 파일 |
|---|---|
| 화면 레이아웃/문구 수정 | `app/templates/finally.html`, `style.css` |
| 진단 이력 사이드바 수정 | `app/main.py`의 `/history*` 라우트 + `script.js`·`style.css`의 `🆕 [진단 이력 기능]` 블록 |
| 업로드·진행률·슬라이더 동작 수정 | `app/templates/script.js` |
| 결과 표시 항목 변경 | `app/templates/f_result.html` (+ 필요 시 `main.py`의 `render_template` 인자) |
| 라우트 추가/업로드 제한 변경 | `app/main.py` |
| 판정 임계값·전처리 변경 | `app/ai_engine.py` |
| 박스/히트맵 그리는 방식 변경 | `app/cv_processor.py` |
| DB 스키마 변경 | `app/main.py`의 `log_document` (195~213행) |
| 모델 재학습 | `src/v3/` |
