# 프론트 연동 가이드 — `/predict` 응답 데이터 5종

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
| `cam_image_url` | str | **③ 판단 근거 이미지 URL.** 불량 → LayerCAM 히트맵 오버레이(+마커), 우수 → EXCELLENT 스탬프. 에러 시 원본 URL이 그대로 옴 |
| `peak_x`, `peak_y` | int \| None | **④ 확인 요망 지점 좌표.** 원본 이미지 픽셀 좌표계. **우수 판정 또는 에러 시 `None`** |
| `inference_ms` | int \| None | **⑤ 추론 속도.** 이미지 1장 AI 연산 시간(밀리초). 에러 시 `None` |
| `user_image_url` | str | 업로드 원본 이미지 URL (기존 변수) |

## 사용 예시 (Jinja2)

```html
<h2>판정: {{ ai_result }}</h2>

{% if probability_percent is not none %}
  <p>{{ probability_percent }}% 확률로 {{ ai_result }}</p>
{% endif %}

<img src="{{ cam_image_url }}" alt="AI 판단 근거">

{% if peak_x is not none %}
  <p>⚠ 이 지점 확인 요망: ({{ peak_x }}, {{ peak_y }}) — 근사 위치입니다</p>
{% endif %}

{% if inference_ms is not none %}
  <small>AI 분석 시간: {{ inference_ms }}ms</small>
{% endif %}
```

## 표시할 때 반드시 지켜야 할 것

1. **위치 마커는 근사치입니다.** `peak_x/peak_y`(및 오버레이의 "Check Here" 마커)는
   히트맵이 가장 강하게 반응한 한 점이지, 결함의 정확한 경계 박스가 아닙니다.
   UI 문구에 "근사 위치" 또는 "이 지점 주변을 확인하세요"처럼 명시해 주세요.
2. **히트맵은 이미지 중앙 영역에만 깔립니다.** 모델은 원본 전체가 아니라 중앙
   정사각 영역(전처리 크롭)만 보고 판정합니다. 오버레이가 가장자리를 안 덮는 것은
   정상이며, 색이 있는 영역이 곧 "AI가 분석한 영역"입니다.
3. **확률은 참고 지표입니다.** 캘리브레이션되지 않은 softmax 값이라 100.0%처럼
   극단값이 자주 나옵니다. 판정의 근거이지 정밀 계측값이 아니라는 톤으로 표기 권장.
4. **에러 처리.** `ai_result`에 `"실패"`가 포함되면 나머지 값(`probability_percent`,
   `peak_x/peak_y`, `inference_ms`)은 `None`입니다. `is not none` 가드 필수.
5. **judgment 기준.** 우수/불량 판정은 불량 확률을 운영 임계값(기본 0.50,
   `MODEL_THRESHOLD` 환경변수로 조정)과 비교한 결과입니다. argmax가 아니므로
   프론트에서 확률을 보고 재판정하지 마세요 — `ai_result`가 최종 판정입니다.

## 이미지 색상 안내 (범례 만들 때)

- 오버레이는 JET 컬러맵: **빨강·노랑 = 모델이 불량 근거로 강하게 본 곳**,
  파랑 = 거의 참고하지 않은 곳.
- 빨간 원 + "Check Here" = 히트맵 최대 반응 지점 (근사).
- 우수 판정은 초록 "EXCELLENT (GOOD)" 스탬프 이미지가 옵니다.

## 참고 수치 (CPU 로컬 실측, 2026-07-29)

- 우수 판정(히트맵 생략): 약 340ms
- 불량 판정(히트맵 포함): 약 1,500ms
- GPU 인스턴스에서는 크게 단축됩니다. 로딩 스피너를 붙일 거면 불량 케이스 기준으로.
