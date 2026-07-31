# 프론트 연동 가이드 — `/predict` 응답 데이터 6종

백엔드가 ConvNeXt-Tiny 이원화 모델(binclf_v3)로 전환되면서, `/predict` POST 처리 후
`result.html` 렌더링 시 아래 변수들이 템플릿 컨텍스트로 내려갑니다.
Jinja2에서 `{{ 변수명 }}`으로 바로 사용하면 됩니다.

## 요청 (기존과 동일)

- `POST /predict`, `multipart/form-data`
- 필드명 `house_image` — JPG/JPEG/PNG, 50MB 이하, 가로·세로 1024px 이하
- 검증 실패 시 기존과 동일하게 `<script>alert(...)` 응답이 옵니다

## 응답 템플릿 변수

| 변수 | 타입 | 설명 |
|---|---|---|
| `ai_result` | str | **① 판정 결과.** `"우수"` 또는 `"불량"`. 에러 시 `"분류 실패 (...)"` 문자열 |
| `probability_percent` | float \| None | **② 판정 확률 %.** 판정된 클래스의 확률 (0~100, 소수 1자리). 예: 불량 판정 + `87.3` → "87.3% 확률로 불량". 에러 시 `None` |
| `cam_image_url` | str | **③ 판정 근거 이미지 URL.** 불량 → 결함 근사 영역 빨간 박스("Defect Area"), 우수 → EXCELLENT 스탬프. 에러 시 원본 URL이 그대로 옴 |
| `heatmap_image_url` | str \| None | **⑥ LayerCAM 히트맵 오버레이 URL.** 불량 판정에서만 생성됩니다(JET 컬러맵 + 분석 영역 흰 테두리). 박스 이미지와 **크기·좌표계가 완전히 동일**해서 같은 자리에 겹쳐 놓고 탭으로 교체해도 화면이 튀지 않습니다. **우수 판정·에러·히트맵 생성 실패 시 `None`** — 이때는 히트맵 탭을 감추고 `cam_image_url` 한 장만 쓰세요 |
| `peak_x`, `peak_y` | int \| None | **④ 확인 요망 지점 좌표.** AI가 가장 강하게 반응한 지점, 원본 이미지 픽셀 좌표계. 이미지에는 그려지지 않고 데이터로만 제공 — 프론트에서 툴팁/핀 등으로 자유롭게 활용. **우수 판정 또는 에러 시 `None`** |
| `inference_ms` | int \| None | **⑤ 추론 속도.** 이미지 1장 AI 연산 시간(밀리초). 에러 시 `None` |
| `user_image_url` | str | 업로드 원본 이미지 URL (기존 변수) |

## 사용 예시 (Jinja2)

```html
<h2>판정: {{ ai_result }}</h2>

{% if probability_percent is not none %}
  <p>{{ probability_percent }}% 확률로 {{ ai_result }}</p>
{% endif %}

<img src="{{ cam_image_url }}" alt="AI 판정 근거 (결함 박스)">

{% if heatmap_image_url %}
  <img src="{{ heatmap_image_url }}" alt="AI 판정 근거 (LayerCAM 히트맵)">
{% endif %}

{% if peak_x is not none %}
  <p>⚠ 중점 확인 지점: ({{ peak_x }}, {{ peak_y }}) — 근사 위치입니다</p>
{% endif %}

{% if inference_ms is not none %}
  <small>AI 분석 시간: {{ inference_ms }}ms</small>
{% endif %}
```

## 진단 이력 (`/history`, `/history/<id>`)

좌측 사이드바에서 지난 진단을 다시 열 수 있습니다. 사용자 구분은 **`hn_client_id` 쿠키(2일)** 하나로만 합니다
— 로그인이 없으므로 쿠키를 지우면 이력이 사라집니다.

- `GET /history` → `{"items":[{"id","date","status","thumb"}]}`. 최근 **20건**, `date`는 `MM/DD HH:MM`(연도 없음),
  `status`는 `"우수"`/`"불량"`/`"오류"`, `thumb`은 원본 사진 URL(없으면 `null`).
  쿠키가 없거나 DB 장애면 빈 목록을 돌려줍니다 — 화면이 죽지 않는 것이 우선입니다.
- `GET /history/<id>` → **`/predict`와 완전히 같은 형식의 `f_result.html` 조각**. 그래서 프론트는 이 응답을
  기존 `injectBackendResult()`에 그대로 넘기면 되고, 슬라이더·레이어 탭 로직은 손댈 필요가 없습니다.
  남의 기록·오류 건·확률 없는 건은 404입니다.

**복원 시 반드시 지킬 것 — 이전 레이어를 먼저 걷어내세요.**
`injectBackendResult()`는 히트맵이 있을 때 슬라이더·탭을 **켜기만 하고, 없을 때 끄지는 않습니다.**
정상 진단 흐름에서는 파일을 새로 고를 때 `renderPreview()`가 미리 감춰 주지만, 이력 복원 경로에는 그 단계가
없습니다. 그래서 불량 건을 복원한 뒤 우수 건을 복원하면 **직전 불량 건의 슬라이더와 탭이 화면에 남습니다.**
`restoreFromHistory()`가 `injectBackendResult()` 호출 직전에 `compareWrap`과 `layerTabs`에 `hide`를 붙이는
이유가 이것입니다. 복원 경로를 새로 만들 때도 같은 정리가 필요합니다.

`peak_x` / `peak_y`는 복원 시 항상 `None`입니다 — `f_result.html`이 쓰지 않아 DB에 저장하지 않기 때문입니다.
나중에 이 값을 화면에 표시하려면 `main.py`의 `log_document`에 저장부터 추가해야 합니다.

## before/after 슬라이더 + 레이어 탭 (현재 `finally.html` 구현)

불량 판정 화면은 **원본(before) ↔ 판정 근거(after)** 를 좌우로 겹쳐 놓고 분할선을 끌어
비교하는 방식입니다. after 레이어는 탭으로 교체합니다.

**슬라이더는 불량 판정 전용입니다.** `heatmap_image_url`이 있을 때만 켜지고, 우수 판정이나
히트맵 생성 실패는 기존과 똑같이 `cam_image_url` 한 장(`resultImg`)만 그대로 보여줍니다 —
비교할 근거 레이어가 없기 때문입니다.

- **before** = `user_image_url`, **after** = `heatmap_image_url`(기본) 또는 `cam_image_url`(박스 탭)
- 탭: `히트맵` / `결함 박스` — 불량일 때만 노출, 기본 선택은 히트맵
- 우수 판정: 슬라이더·탭 모두 숨김. EXCELLENT 스탬프 이미지 한 장만 표시 (기존 동작 그대로)
- 슬라이더의 after 이미지는 `resultImg`가 아니라 별도의 `afterImg`입니다. `resultImg`는
  원래 위치·용도 그대로 남아 있어 우수 경로가 이전과 완전히 동일하게 동작합니다
- 분할선은 CSS 변수 `--split` 하나로 `clip-path`와 divider가 함께 움직입니다 (`style.css`의 `.compare-*`)
- 비교 상자 크기는 `script.js`의 `fitCompareWrap()`이 원본 종횡비에 맞춰 계산합니다.
  뷰포트를 그대로 쓰면 세로 사진에서 분할선이 이미지 밖으로 새기 때문입니다

구현 시 주의 — `.image-viewport img`에 걸린 `z-index: 1`이 before 이미지에도 적용되므로
after 레이어(`.compare-after`)에 **`z-index: 2`가 반드시 있어야** 합니다. 없으면 after가
before 뒤로 깔려서 "탭을 눌러도 아무 변화가 없는" 증상이 납니다.

## 표시할 때 반드시 지켜야 할 것

1. **박스는 근사 영역입니다.** 빨간 박스는 detection 모델의 정밀 경계가 아니라,
   분류 모델의 판단 근거 히트맵(LayerCAM)을 이진화해 만든 근사 영역입니다.
   `peak_x/peak_y`도 마찬가지로 근사치입니다. UI 문구에 "근사 영역/위치" 또는
   "이 부근을 확인하세요" 톤을 명시해 주세요.
2. **모델은 이미지 중앙 영역만 분석합니다.** 전처리(중앙 크롭) 특성상 원본의
   가장자리 결함은 박스가 안 잡힐 수 있습니다. 촬영 가이드에 "결함부를 화면
   중앙에 두고 촬영"을 안내하면 좋습니다.
3. **확률은 참고 지표입니다.** 캘리브레이션되지 않은 softmax 값이라 100.0%처럼
   극단값이 자주 나옵니다. 판정의 근거이지 정밀 계측값이 아니라는 톤으로 표기 권장.
4. **에러 처리.** `ai_result`에 `"실패"`가 포함되면 나머지 값(`probability_percent`,
   `peak_x/peak_y`, `inference_ms`)은 `None`입니다. `is not none` 가드 필수.
5. **판정 기준.** 우수/불량 판정은 불량 확률을 운영 임계값(기본 0.50,
   `MODEL_THRESHOLD` 환경변수로 조정)과 비교한 결과입니다. argmax가 아니므로
   프론트에서 확률을 보고 재판정하지 마세요 — `ai_result`가 최종 판정입니다.

## 이미지 안내 (범례 만들 때)

- 빨간 사각 박스 + "Defect Area" = AI가 불량 근거로 강하게 본 근사 영역
  (여러 개일 수 있음, 없을 수도 있음 — 반응이 약하게 분산된 경우).
- 히트맵은 JET 컬러 = **빨강(강한 반응) → 노랑 → 초록 → 파랑(약한 반응)**.
  흰 사각 테두리 + "LayerCAM Heatmap" 문구 안쪽이 모델이 실제로 본 중앙 크롭 영역이고,
  테두리 밖은 원본 그대로입니다 (색이 안 칠해진 게 정상).
- 우수 판정은 초록 "EXCELLENT (GOOD)" 스탬프 이미지가 옵니다.
- 박스 개수/민감도는 백엔드 `cv_processor.py`의 `BOX_THRESHOLD`(기본 0.5)로
  조정합니다 — 자잘한 박스가 많으면 0.6~0.7로 상향 요청.
- 히트맵 진하기는 같은 파일의 `HEATMAP_ALPHA`(기본 0.45)로 조정합니다 —
  낮추면 원본 질감이, 올리면 반응 분포가 잘 보입니다.

## 참고 수치 (CPU 로컬 실측, 2026-07-29)

- 우수 판정(히트맵 연산 생략): 약 280~340ms
- 불량 판정(내부 히트맵 연산 포함): 약 1,450ms
- `inference_ms`는 모델 연산만 재는 값입니다. 히트맵·박스 이미지 인코딩(각 1장 추가 저장)은
  이 수치 밖이라 응답 체감 시간은 조금 더 깁니다 — 히트맵 추가로 늘어난 몫은 수십 ms 수준.
- GPU 인스턴스에서는 크게 단축됩니다. 로딩 스피너를 붙일 거면 불량 케이스 기준으로.
