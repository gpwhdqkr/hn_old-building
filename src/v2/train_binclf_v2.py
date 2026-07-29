# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# v2 이원화(우수 vs 불량) 1단계 baseline 학습 (head만, ImageNet 가중치).
#
# 실행 (런팟 = 리눅스):
#   ARCH=resnet50 RUN_NAME=v2r50a python src/v2/train_binclf_v2.py
# (PowerShell — 로컬 스모크):
#   $env:DATA_DIR="D:/hn_old-building"; $env:RUN_NAME="smoke"
#   $env:LIMIT_PER_SPLIT="64"; $env:BATCH_SIZE="8"; $env:NUM_WORKERS="2"; $env:NUM_EPOCHS="1"
#   python src/v2/train_binclf_v2.py
#
# [환경변수] ARCH(resnet50) RUN_NAME(default) NUM_EPOCHS(5) LR_HEAD(1e-3)
#   BATCH_SIZE(64) SEED(42) USE_AMP(1) USE_CLAHE(0) EVAL_RESIZE_MODE(centercrop)
#   DATA_DIR / LIMIT_PER_SPLIT(스모크 전용) — 상세는 src/v2/README.md
#
# 결과:
#   model/best_<ARCH>_binclf_v2_baseline_<RUN_NAME>.pth
#   test_results/binclf_v2/<ARCH>/<RUN_NAME>/train_history_baseline.csv
#
# 다음 단계 (같은 ARCH/RUN_NAME 필수):
#   ARCH=resnet50 RUN_NAME=v2r50a python src/v2/fine_tune_binclf_v2.py
# ============================================================

from pipeline_v2 import run_train

if __name__ == "__main__":
    run_train("binclf")
