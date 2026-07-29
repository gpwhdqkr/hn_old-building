# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# v3 이원화 test 평가 (3단계): 클래스별 F1/혼동행렬/추론속도 3층 + 샘플별 예측 CSV.
#
# 실행 (런팟 = 리눅스):
#   ARCH=resnet50 RUN_NAME=v3r50a python src/v3/evaluate_binclf_v3.py
#   EVAL_TARGET=baseline 으로 지정하면 1단계 모델을 평가
# (PowerShell):
#   $env:ARCH="resnet50"; $env:RUN_NAME="v3r50a"; python src/v3/evaluate_binclf_v3.py
#
# 결과 (test_results/binclf_v2/<ARCH>/<RUN_NAME>/):
#   evaluate_<EVAL_TARGET>_result.txt        — 화면 출력 전문 (기존 binclf 포맷 유지)
#   test_confusion_matrix_<EVAL_TARGET>.csv
#   predictions_<EVAL_TARGET>.csv            — 샘플별 확률/예측 (v2 신규)
#
# [주의] 학습 때와 USE_CLAHE / EVAL_RESIZE_MODE가 다르면 경고가 출력됩니다.
# 이원화 test는 불량이 약 91%라 accuracy에 속지 말 것 — 우수 F1과 macro F1 기준.
# ============================================================

from pipeline_v3 import run_evaluate

if __name__ == "__main__":
    run_evaluate("binclf")
