# CLAUDE.md — 이 저장소의 길잡이

노후 건물(아파트) 균열 진단 서비스. **Flask 웹 서비스 + ConvNeXt-Tiny 이원화 분류 모델(binclf_v3)**.

이 파일은 **어디로 가야 하는지만** 알려준다. 상세 내용은 각 문서에 있으니
필요한 것만 골라 읽어라. 무작정 `grep`/전체 탐색부터 하지 말 것.

---

## 하려는 일 → 읽을 곳

| 하려는 일 | 먼저 읽을 파일 |
|---|---|
| **구조 파악 · 어디에 뭐가 있는지 모를 때** | **[docs/PROJECT_MAP.md](docs/PROJECT_MAP.md)** ← 항상 여기서 시작 |
| 화면 레이아웃 · 문구 · 스타일 수정 | `app/templates/finally.html`, `app/templates/style.css` |
| 업로드 · 진행률 · 비교 슬라이더 등 프론트 동작 | `app/templates/script.js` |
| 결과 화면에 표시할 항목 변경 | `app/templates/f_result.html` + [docs/FRONTEND_GUIDE.md](docs/FRONTEND_GUIDE.md) |
| **프론트↔백 데이터 규약 (변수 6종)** | [docs/FRONTEND_GUIDE.md](docs/FRONTEND_GUIDE.md) |
| 라우트 추가 · 업로드 제한 · MongoDB 스키마 | `app/main.py` |
| **진단 이력 사이드바** (쿠키 · 목록 · 과거 결과 복원) | `app/main.py`의 `/history*` + [docs/superpowers/specs/2026-07-31-history-sidebar-design.md](docs/superpowers/specs/2026-07-31-history-sidebar-design.md) |
| 판정 임계값 · 전처리 · LayerCAM | `app/ai_engine.py` |
| 결함 박스 · 히트맵 그리는 방식 | `app/cv_processor.py` |
| **v3 모델 학습 · 재학습 · 평가 · 임계값 튜닝** | **[src/v3/README.md](src/v3/README.md)** |
| 데이터 수집 명세 · 실험 결과 (보고서용) | [docs/노후건물_데이터수집_모델링결과.md](docs/노후건물_데이터수집_모델링결과.md) |
| 전 실험 성능 비교표 | [test_results/summary_all_tests.md](test_results/summary_all_tests.md) |
| RunPod GPU 대여/반납 | [runpod_test/README.md](runpod_test/README.md) |
| 구버전 계보 확인이 필요할 때만 | `src/v2/README.md`, `src/binclf/README.md`, `docs/ResNet_사용설명서.md` |

---

## 먼저 알아야 할 3가지

1. **웹 서비스는 전부 `app/` 안에 있다.**
   진입점은 `app/main.py` (`python app/main.py` → `0.0.0.0:5000`).

2. **프론트의 HTML·CSS·JS가 전부 `app/templates/` 안에 있다. `app/static/`이 아니다.**
   `main.py`의 `/style.css`, `/script.js` 라우트가 `templates/`에서 강제로 서빙한다.
   `app/static/images/`는 런타임 업로드·결과 이미지 저장소일 뿐이고 gitignore 대상이다.

3. **`src/`, `model/`, `data/`는 학습 쪽이다.** 웹 작업 중이면 열 필요 없다.
   현재 서비스에 쓰는 코드 계보는 `src/v3/` 하나뿐이고, 나머지는 지난 실험이다.

## 실행 전 확인

- **모델 가중치**: `model/best_convnext_tiny_binclf_v3_finetuned_v3cta.pth`.
  `model/*.pth`는 gitignore라 저장소에 없다 — 서빙 PC에 실물이 있어야 기동된다.
- **MongoDB**: `apartment_inspection_db` / `inspection_logs`.
  `MONGO_URI` 환경변수로 주소 변경. DB가 죽어도 화면 출력은 계속되도록 격리돼 있다.
- **웹 서비스는 별도 PC(192.168.0.22)에서 구동**된다. DB도 그쪽 localhost라
  개발 PC에서는 종단 검증이 안 된다.

## 문서를 고칠 때

구조가 바뀌면 `docs/PROJECT_MAP.md`와 이 파일도 같이 고쳐라.
새 md는 흩뿌리지 말고 `docs/`에 두고, 위 표에 한 줄 추가한다.
