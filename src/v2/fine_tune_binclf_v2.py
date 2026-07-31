# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# v2 이원화 2단계 파인튜닝 (부분 unfreeze + 차등 lr + cosine + AMP).
# 반드시 train_binclf_v2.py를 같은 ARCH/RUN_NAME으로 먼저 실행해야 합니다.
#
# 실행 (런팟 = 리눅스):
#   ARCH=resnet50 RUN_NAME=v2r50a NUM_EPOCHS=15 python src/v2/fine_tune_binclf_v2.py
# (PowerShell):
#   $env:ARCH="resnet50"; $env:RUN_NAME="v2r50a"; python src/v2/fine_tune_binclf_v2.py
#
# [환경변수] NUM_EPOCHS(15) PATIENCE(3) LR_BACKBONE(1e-5) LR_HEAD_FINETUNE(1e-4)
#   USE_SCHEDULER(1=cosine) USE_AMP(1) — 나머지는 train과 동일, README 참고
#
# 결과:
#   model/best_<ARCH>_binclf_v2_finetuned_<RUN_NAME>.pth
#   (validation macro F1이 baseline보다 좋아진 시점만 갱신. 한 번도 안 좋아지면
#    baseline 상태 그대로 저장되어 있음 — 기존 binclf 방식과 동일)
#   test_results/binclf_v2/<ARCH>/<RUN_NAME>/train_history_finetuned.csv
#
# 다음 단계:
#   ARCH=resnet50 RUN_NAME=v2r50a python src/v2/evaluate_binclf_v2.py
# ============================================================

from pipeline_v2 import run_fine_tune

if __name__ == "__main__":
    run_fine_tune("binclf")
