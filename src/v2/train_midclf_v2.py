# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# v2 결함종류 7클래스(middle_id: C/P/X/F/T/L/W) 1단계 baseline 학습.
# 추론은 softmax 최고 확률 클래스를 선택하는 방식 (evaluate에서 argmax).
#
# 실행 (런팟 = 리눅스):
#   ARCH=resnet50 RUN_NAME=v2r50a python src/v2/train_midclf_v2.py
# (PowerShell — 로컬 스모크):
#   $env:DATA_DIR="D:/hn_old-building"; $env:RUN_NAME="smoke"
#   $env:LIMIT_PER_SPLIT="70"; $env:BATCH_SIZE="8"; $env:NUM_WORKERS="2"; $env:NUM_EPOCHS="1"
#   python src/v2/train_midclf_v2.py
#
# [환경변수] train_binclf_v2.py와 동일 — src/v2/README.md 참고
#
# 결과:
#   model/best_<ARCH>_midclf_v2_baseline_<RUN_NAME>.pth
#   test_results/midclf_v2/<ARCH>/<RUN_NAME>/train_history_baseline.csv
#
# 다음 단계 (같은 ARCH/RUN_NAME 필수):
#   ARCH=resnet50 RUN_NAME=v2r50a python src/v2/fine_tune_midclf_v2.py
# ============================================================

from pipeline_v2 import run_train

if __name__ == "__main__":
    run_train("midclf")
