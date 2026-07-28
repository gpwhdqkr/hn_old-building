# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# v2 결함종류 7클래스 test 평가: softmax argmax 예측, 7x7 혼동행렬,
# 추론속도 3층, 샘플별 7클래스 확률 CSV.
#
# 실행 (런팟 = 리눅스):
#   ARCH=resnet50 RUN_NAME=v2r50a python src/v2/evaluate_midclf_v2.py
# (PowerShell):
#   $env:ARCH="resnet50"; $env:RUN_NAME="v2r50a"; python src/v2/evaluate_midclf_v2.py
#
# 결과 (test_results/midclf_v2/<ARCH>/<RUN_NAME>/):
#   evaluate_<EVAL_TARGET>_result.txt
#   test_confusion_matrix_<EVAL_TARGET>.csv  — 7x7, middle_name 라벨
#   predictions_<EVAL_TARGET>.csv            — prob_C ~ prob_W 확률 전부 포함
# ============================================================

from pipeline_v2 import run_evaluate

if __name__ == "__main__":
    run_evaluate("midclf")
