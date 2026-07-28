# -*- coding: utf-8 -*-
"""폴리곤 → bbox(min/max) 통합 변환 스크립트 (전처리 v2).

annotations_v2.jsonl의 annotation 77,432개를 전부 bbox 하나로 통일한
평탄 CSV를 생성한다. 소비 코드가 폴리곤/타입 분기 없이 bbox 4열만 읽으면 된다.
  - Type="polygon" (45,790개): 평탄화 폴리곤 [x1,y1,...]의 min/max로 bbox 유도
  - Type="bbox"    (31,642개): 원본 bbox([x0,y0,x1,y1] 코너 좌표) 그대로 사용
    ※ 과거 전처리 ②가 polygon 필드만 저장해 이 41%가 유실됐던 버그의 재발 방지를
      위해, 두 타입 모두 처리됐는지 개수 검증을 함께 수행한다.

산출물 (기존 산출물은 절대 덮어쓰지 않는다 - 이미 있으면 즉시 에러):
  1) data/processed/annotations_bbox_v2.csv        - annotation 1개 = 1행 (utf-8-sig)
     컬럼: source_data_id, ann_index, class_id, type, x0, y0, x1, y1
     좌표는 원본 픽셀 좌표계. 캐시(processed_images_v2) 좌표가 필요하면
     image_spec_v2.csv의 scale을 곱할 것 (폴리곤 좌표와 동일 규칙).
  2) data/processed/annotations_bbox_v2_report.txt - 전수 검증 리포트

전수 검증 항목:
  - 폴리곤형: 기존 bbox 필드 vs min/max 재계산 일치 여부
  - 코너 순서(x0<=x1, y0<=y1), 퇴화 박스(폭/높이 0), 음수 좌표
  - image_spec_v2.csv 실측 해상도 대비 범위 초과 (명목 Resolution은 신뢰 금지)
  - 타입별 개수가 annotations_v2.jsonl과 일치하는지

실행:
    python src/convert_polygons_to_bbox_v2.py
경로가 다르면:
    (PowerShell) $env:DATA_DIR="D:/hn_old-building"; python src/convert_polygons_to_bbox_v2.py
"""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

data_dir = Path(os.environ.get("DATA_DIR", r"D:\hn_old-building"))

annotations_path = data_dir / "data" / "processed" / "annotations_v2.jsonl"
image_spec_path = data_dir / "data" / "processed" / "image_spec_v2.csv"
out_csv_path = data_dir / "data" / "processed" / "annotations_bbox_v2.csv"
out_report_path = data_dir / "data" / "processed" / "annotations_bbox_v2_report.txt"


def main():
    for path in (out_csv_path, out_report_path):
        if path.exists():
            sys.exit(f"[에러] 산출물이 이미 존재합니다 (덮어쓰기 금지): {path}")
    if not annotations_path.exists():
        sys.exit(f"[에러] 입력이 없습니다: {annotations_path}")

    # 실측 해상도 (명목 Resolution 필드는 전부 [1440,1080]이라 신뢰 금지)
    spec = {}
    with open(image_spec_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            spec[row["source_data_id"]] = (int(row["orig_width"]), int(row["orig_height"]))

    type_counts = Counter()
    n_minmax_mismatch = 0      # 폴리곤형: 기존 bbox 필드 != min/max 재계산
    n_bad_corner = 0           # x0>x1 또는 y0>y1
    n_degenerate = 0           # 폭 또는 높이 0
    n_negative = 0             # 음수 좌표
    n_out_of_bounds = 0        # 실측 해상도 초과 (x1>W 또는 y1>H)
    n_no_spec = 0              # image_spec_v2에 없는 이미지
    n_no_coords = 0            # polygon도 bbox도 없는 행
    rows = []

    with open(annotations_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ann_type = rec.get("type", "")
            type_counts[ann_type] += 1
            polygon = rec.get("polygon") or []
            bbox = rec.get("bbox") or []

            if polygon:
                xs = polygon[0::2]
                ys = polygon[1::2]
                minmax = [min(xs), min(ys), max(xs), max(ys)]
                if bbox and bbox != minmax:
                    n_minmax_mismatch += 1
                bbox = minmax
            if not bbox:
                n_no_coords += 1
                continue

            x0, y0, x1, y1 = bbox
            if x0 > x1 or y0 > y1:
                n_bad_corner += 1
            if x0 == x1 or y0 == y1:
                n_degenerate += 1
            if min(bbox) < 0:
                n_negative += 1
            wh = spec.get(rec["source_data_id"])
            if wh is None:
                n_no_spec += 1
            elif x1 > wh[0] or y1 > wh[1]:
                n_out_of_bounds += 1

            rows.append({
                "source_data_id": rec["source_data_id"],
                "ann_index": rec["ann_index"],
                "class_id": rec["class_id"],
                "type": ann_type,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            })

    with open(out_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source_data_id", "ann_index", "class_id", "type",
                           "x0", "y0", "x1", "y1"])
        writer.writeheader()
        writer.writerows(rows)

    n_total = sum(type_counts.values())
    report = [
        "annotations_bbox_v2 변환/검증 리포트",
        f"입력: {annotations_path}",
        f"출력: {out_csv_path}",
        "",
        f"annotation 총 {n_total}개 -> bbox 행 {len(rows)}개",
        f"  타입별: {dict(type_counts)}",
        f"  polygon형 min/max 재계산 vs 기존 bbox 필드 불일치: {n_minmax_mismatch}",
        f"  좌표 없는 행(제외됨): {n_no_coords}",
        "",
        "bbox 전수 검증 (원본 픽셀 좌표계):",
        f"  코너 순서 오류(x0>x1 or y0>y1): {n_bad_corner}",
        f"  퇴화 박스(폭 또는 높이 0): {n_degenerate}",
        f"  음수 좌표: {n_negative}",
        f"  실측 해상도 초과(image_spec_v2 기준): {n_out_of_bounds}",
        f"  image_spec_v2 미등재 이미지의 annotation: {n_no_spec}",
        "",
        "좌표는 원본 픽셀 좌표계 그대로임. 캐시(processed_images_v2) 좌표가",
        "필요하면 image_spec_v2.csv의 scale을 곱해서 사용할 것.",
    ]
    out_report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
