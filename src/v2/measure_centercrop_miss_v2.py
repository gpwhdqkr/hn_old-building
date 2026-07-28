# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# 학습 전 1회 실행하는 사전 측정: val/test의 결정적 파이프라인
# (Resize 짧은 변 512 → CenterCrop 448)이 불량 이미지의 결함 annotation
# (polygon/bbox 모두)을 얼마나 잘라먹는지 기하 계산으로 실측합니다
# (이미지 파일을 열지 않음 — 수 초).
#
#   python src/v2/measure_centercrop_miss_v2.py
#   (PowerShell) $env:DATA_DIR="D:/hn_old-building"; python src/v2/measure_centercrop_miss_v2.py
#
# [판정 기준 — 실행 전에 고정해 둔 규칙]
#   완전 누락률 > 2%  또는  max coverage < 0.5 비율 > 10%
#     → EVAL_RESIZE_MODE=letterbox 를 기본값으로 채택 (README에 기록)
#   그 미만 → centercrop 유지 (기존 실험과 비교 연속성 우선)
#
# 결과: test_results/v2_checks/centercrop_miss_report.txt (+ 화면 동일 출력)
# ============================================================

from collections import Counter, defaultdict

import pandas as pd

from preprocess_v2 import (
    EVAL_RESIZE_SHORT,
    INPUT_SIZE,
    image_spec_path,
    load_defect_bboxes,
    metadata_path,
    test_results_root,
)

COMPLETE_MISS_LIMIT = 0.02   # 완전 누락률 허용 상한
LOW_COVERAGE_LIMIT = 0.10    # max coverage < 0.5 비율 허용 상한


def main():
    result_lines = []

    def record(text=""):
        print(text)
        result_lines.append(str(text))

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    spec = pd.read_csv(image_spec_path, encoding="utf-8-sig")

    target = metadata[
        metadata["split"].isin(["validation", "test"])
        & (metadata["model_label"] == 1)
    ].merge(spec, on=["source_data_id", "image_relpath"])

    record("===== CenterCrop 결함 누락률 사전 측정 =====")
    record(f"대상 : val/test 불량 이미지 {len(target)}장")
    record(f"파이프라인 : Resize(짧은 변 {EVAL_RESIZE_SHORT}) → CenterCrop({INPUT_SIZE})")

    # 결함 bbox (캐시 좌표계, polygon/bbox 타입 모두 포함 — preprocess_v2와 동일 로직)
    defect_bboxes, degenerate_count = load_defect_bboxes()
    record(f"결함 bbox 인덱스 : {sum(len(b) for b in defect_bboxes.values())}개 "
           f"(퇴화/빈 좌표 제외 {degenerate_count}개)")

    max_coverage = defaultdict(float)
    seen_ids = set()
    letterbox_short_sides = []
    missing_annotation_count = 0

    for row in target.itertuples(index=False):
        boxes = defect_bboxes.get(row.source_data_id)
        if not boxes:
            missing_annotation_count += 1
            continue

        cache_width, cache_height = row.cache_width, row.cache_height

        # Resize(짧은 변 512) 시뮬레이션
        resize_factor = EVAL_RESIZE_SHORT / min(cache_width, cache_height)
        resized_width = cache_width * resize_factor
        resized_height = cache_height * resize_factor

        # 중앙 448x448 크롭 창
        crop_left = (resized_width - INPUT_SIZE) / 2
        crop_top = (resized_height - INPUT_SIZE) / 2
        crop_right = crop_left + INPUT_SIZE
        crop_bottom = crop_top + INPUT_SIZE

        letterbox_factor = INPUT_SIZE / max(cache_width, cache_height)

        for x0, y0, x1, y1 in boxes:
            bx0, by0 = x0 * resize_factor, y0 * resize_factor
            bx1, by1 = x1 * resize_factor, y1 * resize_factor

            inter_width = max(0.0, min(bx1, crop_right) - max(bx0, crop_left))
            inter_height = max(0.0, min(by1, crop_bottom) - max(by0, crop_top))
            bbox_area = max(1e-9, (bx1 - bx0) * (by1 - by0))
            coverage = (inter_width * inter_height) / bbox_area

            max_coverage[row.source_data_id] = max(
                max_coverage[row.source_data_id], coverage
            )
            letterbox_short_sides.append(min(x1 - x0, y1 - y0) * letterbox_factor)

        seen_ids.add(row.source_data_id)

    if missing_annotation_count:
        record(f"[경고] 결함 bbox가 없는 불량 이미지 {missing_annotation_count}장 "
               "(annotations 누락?)")

    coverages = [max_coverage[source_id] for source_id in seen_ids]
    total = len(coverages)

    complete_miss = sum(1 for value in coverages if value == 0.0)
    below_030 = sum(1 for value in coverages if value < 0.30)
    below_050 = sum(1 for value in coverages if value < 0.50)

    record("")
    record("----- 이미지 단위 max coverage (결함 bbox 중 가장 잘 보이는 것 기준) -----")
    record(f"완전 누락 (coverage = 0)   : {complete_miss}장 ({complete_miss / total:.2%})")
    record(f"coverage < 0.30            : {below_030}장 ({below_030 / total:.2%})")
    record(f"coverage < 0.50            : {below_050}장 ({below_050 / total:.2%})")

    # 해상도 그룹별 분해 (세로로 긴 이미지가 위험군)
    record("")
    record("----- 캐시 해상도별 완전 누락 -----")
    size_counter = Counter()
    size_miss_counter = Counter()
    for row in target.itertuples(index=False):
        if row.source_data_id not in seen_ids:
            continue
        key = (row.cache_width, row.cache_height)
        size_counter[key] += 1
        if max_coverage[row.source_data_id] == 0.0:
            size_miss_counter[key] += 1
    for size, count in size_counter.most_common():
        miss = size_miss_counter.get(size, 0)
        record(f"{size[0]}x{size[1]} : {count}장 중 완전 누락 {miss}장 ({miss / count:.2%})")

    if letterbox_short_sides:
        letterbox_short_sides.sort()
        median_value = letterbox_short_sides[len(letterbox_short_sides) // 2]
        record("")
        record(f"[참고] letterbox 전환 시 결함 bbox 짧은 변 중앙값 : {median_value:.1f}px "
               f"(too small하면 letterbox의 축소 부작용 주의)")

    # ---- 판정 ----
    complete_miss_rate = complete_miss / total
    low_coverage_rate = below_050 / total

    record("")
    record("===== 판정 =====")
    record(f"기준 : 완전 누락 > {COMPLETE_MISS_LIMIT:.0%} 또는 "
           f"coverage<0.5 비율 > {LOW_COVERAGE_LIMIT:.0%} 이면 letterbox 권장")

    if complete_miss_rate > COMPLETE_MISS_LIMIT or low_coverage_rate > LOW_COVERAGE_LIMIT:
        record("결론 : EVAL_RESIZE_MODE=letterbox 를 기본값으로 사용하세요.")
    else:
        record("결론 : centercrop 유지 (결함 누락이 허용 범위 이내).")

    output_dir = test_results_root / "v2_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "centercrop_miss_report.txt"
    report_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8-sig")
    print("\n리포트 저장 :", report_path)


if __name__ == "__main__":
    main()
