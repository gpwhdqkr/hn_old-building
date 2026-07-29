# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# v2 결함종류 7클래스 2단계 파인튜닝.
# 반드시 train_midclf_v2.py를 같은 ARCH/RUN_NAME으로 먼저 실행해야 합니다.
#
# 실행 (런팟 = 리눅스):
#   ARCH=resnet50 RUN_NAME=v2r50a NUM_EPOCHS=15 python src/v2/fine_tune_midclf_v2.py
# (PowerShell):
#   $env:ARCH="resnet50"; $env:RUN_NAME="v2r50a"; python src/v2/fine_tune_midclf_v2.py
#
# 결과:
#   model/best_<ARCH>_midclf_v2_finetuned_<RUN_NAME>.pth
#   test_results/midclf_v2/<ARCH>/<RUN_NAME>/train_history_finetuned.csv
#
# 다음 단계:
#   ARCH=resnet50 RUN_NAME=v2r50a python src/v2/evaluate_midclf_v2.py
# ============================================================

from pipeline_v2 import run_fine_tune

if __name__ == "__main__":
    run_fine_tune("midclf")
