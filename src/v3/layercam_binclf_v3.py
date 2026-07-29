# -*- coding: utf-8 -*-
# ============================================================
# [사용법]
# 학습된 binclf_v3 체크포인트에 LayerCAM을 적용해
#   1) 불량 히트맵이 실제 결함 위치를 가리키는지 채점 (pointing game + 에너지 비율)
#   2) 히트맵 오버레이 이미지 저장 (정답 bbox 초록 박스와 함께 육안 확인)
# 을 수행한다. 재학습 불필요 - forward/backward hook만 사용.
#
# 실행 (같은 ARCH/RUN_NAME으로 학습이 끝나 있어야 함):
#   ARCH=resnet50 RUN_NAME=v3r50a python src/v3/layercam_binclf_v3.py
# (PowerShell 스모크):
#   $env:DATA_DIR="D:/hn_old-building"; $env:ARCH="resnet50"; $env:RUN_NAME="smoke"
#   $env:LIMIT_IMAGES="40"; python src/v3/layercam_binclf_v3.py
#
# [환경변수]
#   ARCH / RUN_NAME / EVAL_TARGET(finetuned) : 대상 체크포인트 선택
#   LIMIT_IMAGES : 채점할 불량 test 이미지 수 (0=전체). 스모크/미리보기용
#   N_OVERLAYS   : 저장할 오버레이 이미지 수 (기본 8)
#   DATA_DIR 등 나머지는 preprocess_v3와 동일
#
# [채점 방식]
# - 대상: test split의 불량(model_label=1) 이미지 중 GT bbox가 있는 것
# - 히트맵: 불량 클래스(index 1)에 대한 LayerCAM
#   (여러 층의 ReLU(grad)*act 맵을 448로 업샘플 후 min-max 정규화해 평균 융합
#    - 얕은 층이 섞여 Grad-CAM보다 세밀함. 균열처럼 가는 결함 대응)
# - pointing game: 히트맵 최대점이 GT bbox(centercrop 좌표계로 변환) 안이면 hit
# - 에너지 비율: 히트맵 총합 중 GT bbox 내부 비율 (박스 면적 비율과 비교해 해석)
# - GT bbox는 캐시 좌표계 -> Resize(512)/CenterCrop(448) 좌표계로 변환하며,
#   크롭 밖으로 완전히 나간 박스는 제외 (전부 나가면 그 이미지는 채점 제외)
#
# [주의]
# - EVAL_RESIZE_MODE=centercrop 전용 (letterbox는 미지원 - 즉시 에러)
# - LayerCAM은 "모델이 주목한 영역"이지 정답 박스가 아니다. 채점 결과가
#   나쁘면 위치 표시 용도로는 detection 모델이 필요하다는 신호로 해석할 것.
#
# 결과 (test_results/binclf_v3/<ARCH>/<RUN_NAME>_layercam/):
#   layercam_report.txt        : 채점 요약
#   layercam_scores.csv        : 이미지별 점수 (슬라이스 분석용)
#   overlay_<source_id>.jpg    : 히트맵 오버레이 + GT 박스 (N_OVERLAYS장)
# ============================================================

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from model_factory_v3 import create_model
from pipeline_v3 import ARCH, RUN_NAME, checkpoint_path, get_device
from preprocess_v3 import (
    EVAL_RESIZE_SHORT,
    INPUT_SIZE,
    eval_resize_mode,
    load_defect_bboxes,
    load_split_dataframes,
    processed_images_dir,
    test_results_root,
    image_mean,
    image_std,
)

EVALUATION_TARGET = os.environ.get("EVAL_TARGET", "finetuned")
LIMIT_IMAGES = int(os.environ.get("LIMIT_IMAGES", 0))
N_OVERLAYS = int(os.environ.get("N_OVERLAYS", 8))

# 백본별 LayerCAM 추출 층 (얕은 층 -> 깊은 층, 다중 스케일 융합)
def _target_modules(model, arch):
    if arch in ("resnet50", "resnet18"):
        return [model.layer2, model.layer3, model.layer4]
    if arch == "mobilenet_v2":
        return [model.features[6], model.features[13], model.features[18]]
    # efficientnet_b0/b2, convnext_tiny: features 후반 3개 블록
    return [model.features[-3], model.features[-2], model.features[-1]]


class LayerCAM:
    """여러 층의 LayerCAM 맵을 뽑아 정규화-평균으로 융합한다."""

    def __init__(self, model, modules):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.handles = []
        for index, module in enumerate(modules):
            self.handles.append(module.register_forward_hook(self._save_activation(index)))
            self.handles.append(module.register_full_backward_hook(self._save_gradient(index)))

    def _save_activation(self, index):
        def hook(module, inputs, output):
            self.activations[index] = output.detach()
        return hook

    def _save_gradient(self, index):
        def hook(module, grad_inputs, grad_outputs):
            self.gradients[index] = grad_outputs[0].detach()
        return hook

    def compute(self, input_tensor, class_index):
        """입력 1장(batch=1)에 대한 융합 히트맵 [INPUT_SIZE, INPUT_SIZE] (0~1)과
        해당 클래스 확률을 반환한다."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probability = torch.softmax(logits, dim=1)[0, class_index].item()
        logits[0, class_index].backward()

        fused = None
        n_maps = 0
        for index in self.activations:
            activation = self.activations[index]          # [1, C, h, w]
            gradient = self.gradients[index]              # [1, C, h, w]
            cam = torch.relu((torch.relu(gradient) * activation).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
            )[0, 0]
            value_range = cam.max() - cam.min()
            if value_range > 0:
                cam = (cam - cam.min()) / value_range
                fused = cam if fused is None else fused + cam
                n_maps += 1

        if fused is None or n_maps == 0:
            fused = torch.zeros(INPUT_SIZE, INPUT_SIZE)
        else:
            fused = fused / n_maps
        return fused.cpu().numpy(), probability

    def close(self):
        for handle in self.handles:
            handle.remove()


def map_bbox_to_crop(bbox, cache_width, cache_height):
    """캐시 좌표 bbox -> Resize(512 짧은 변)/CenterCrop(448) 좌표. 완전 이탈 시 None.

    torchvision center_crop과 동일한 오프셋 공식(round((크기-448)/2))을 쓴다.
    """
    ratio = EVAL_RESIZE_SHORT / min(cache_width, cache_height)
    resized_width = round(cache_width * ratio)
    resized_height = round(cache_height * ratio)
    offset_x = int(round((resized_width - INPUT_SIZE) / 2.0))
    offset_y = int(round((resized_height - INPUT_SIZE) / 2.0))

    x0 = bbox[0] * ratio - offset_x
    y0 = bbox[1] * ratio - offset_y
    x1 = bbox[2] * ratio - offset_x
    y1 = bbox[3] * ratio - offset_y

    x0, x1 = max(0.0, x0), min(float(INPUT_SIZE), x1)
    y0, y1 = max(0.0, y0), min(float(INPUT_SIZE), y1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (x0, y0, x1, y1)


def make_eval_input(pil_image):
    """평가 경로와 동일한 결정적 전처리 (Resize 512 -> CenterCrop 448 -> 정규화)."""
    from torchvision.transforms.v2 import functional as TF
    image = TF.resize(pil_image, EVAL_RESIZE_SHORT, antialias=True)
    image = TF.center_crop(image, [INPUT_SIZE, INPUT_SIZE])
    crop_pil = image  # 오버레이용 (정규화 전)
    tensor = TF.to_image(image)
    tensor = TF.to_dtype(tensor, torch.float32, scale=True)
    tensor = TF.normalize(tensor, mean=image_mean, std=image_std)
    return tensor.unsqueeze(0), crop_pil


def heatmap_to_color(cam):
    """0~1 히트맵 -> 간단 jet 컬러맵 RGB 배열."""
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(cam - 0.75) * 4.0, 0, 1)
    green = np.clip(1.5 - np.abs(cam - 0.5) * 4.0, 0, 1)
    blue = np.clip(1.5 - np.abs(cam - 0.25) * 4.0, 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def save_overlay(crop_pil, cam, boxes, hit, probability, output_path):
    heat = Image.fromarray(heatmap_to_color(cam))
    overlay = Image.blend(crop_pil.convert("RGB"), heat, alpha=0.45)
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        draw.rectangle(box, outline=(0, 255, 0), width=3)
    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
    marker = 6
    draw.ellipse(
        [peak_x - marker, peak_y - marker, peak_x + marker, peak_y + marker],
        outline=(255, 255, 255), width=3,
    )
    draw.text((6, 6), f"hit={hit} p(bad)={probability:.3f}", fill=(255, 255, 255))
    overlay.save(output_path, quality=90)


def main():
    if eval_resize_mode != "centercrop":
        sys.exit("[에러] LayerCAM 채점은 EVAL_RESIZE_MODE=centercrop 전용입니다")

    device = get_device()
    model_path = checkpoint_path("binclf", EVALUATION_TARGET)
    if not model_path.exists():
        sys.exit(f"[에러] 체크포인트가 없습니다: {model_path}\n"
                 f"먼저 같은 ARCH/RUN_NAME으로 학습을 실행하세요.")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = create_model(ARCH, 2, use_pretrained_weights=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_data = load_split_dataframes("binclf")
    defect_bboxes, _ = load_defect_bboxes()

    targets = test_data[test_data["model_label"] == 1].reset_index(drop=True)
    if LIMIT_IMAGES > 0:
        targets = targets.head(LIMIT_IMAGES)

    output_dir = test_results_root / "binclf_v3" / ARCH / f"{RUN_NAME}_layercam"
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_engine = LayerCAM(model, _target_modules(model, ARCH))

    rows = []
    n_skipped_no_box = 0
    n_overlay_saved = 0

    for _, row in targets.iterrows():
        source_id = row["source_data_id"]
        raw_boxes = defect_bboxes.get(source_id)
        if not raw_boxes:
            n_skipped_no_box += 1
            continue

        with Image.open(processed_images_dir / row["image_relpath"]) as image_file:
            pil_image = image_file.convert("RGB")
        cache_width, cache_height = pil_image.size

        boxes = [
            mapped for mapped in (
                map_bbox_to_crop(box, cache_width, cache_height) for box in raw_boxes
            ) if mapped is not None
        ]
        if not boxes:
            n_skipped_no_box += 1  # 결함이 전부 centercrop 밖 (실측상 극소수)
            continue

        input_tensor, crop_pil = make_eval_input(pil_image)
        cam, probability = cam_engine.compute(input_tensor.to(device), class_index=1)

        peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
        hit = any(x0 <= peak_x <= x1 and y0 <= peak_y <= y1 for x0, y0, x1, y1 in boxes)

        total_energy = float(cam.sum())
        box_mask = np.zeros_like(cam, dtype=bool)
        for x0, y0, x1, y1 in boxes:
            box_mask[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
        energy_ratio = float(cam[box_mask].sum() / total_energy) if total_energy > 0 else 0.0
        box_area_ratio = float(box_mask.mean())

        rows.append({
            "source_data_id": source_id,
            "prob_불량": round(probability, 6),
            "pred_correct": int(probability >= 0.5),
            "peak_x": int(peak_x), "peak_y": int(peak_y),
            "hit": int(hit),
            "energy_in_bbox": round(energy_ratio, 4),
            "bbox_area_ratio": round(box_area_ratio, 4),
            "n_boxes": len(boxes),
        })

        if n_overlay_saved < N_OVERLAYS:
            save_overlay(crop_pil, cam, boxes, bool(hit), probability,
                         output_dir / f"overlay_{source_id}.jpg")
            n_overlay_saved += 1

    cam_engine.close()

    if not rows:
        sys.exit("[에러] 채점 가능한 이미지가 없습니다 (GT bbox 유무 확인)")

    scores = pd.DataFrame(rows)
    scores.to_csv(output_dir / "layercam_scores.csv", index=False, encoding="utf-8-sig")

    pointing = scores["hit"].mean()
    energy = scores["energy_in_bbox"].mean()
    area = scores["bbox_area_ratio"].mean()
    correct_mask = scores["pred_correct"] == 1
    pointing_when_correct = scores.loc[correct_mask, "hit"].mean() if correct_mask.any() else float("nan")

    report = [
        "LayerCAM 위치 채점 리포트 (binclf_v3)",
        f"모델 : {model_path.name} / 장치 : {device.type}",
        f"대상 : test 불량 이미지 {len(scores)}장 채점"
        f" (bbox 없음/크롭 밖 제외 {n_skipped_no_box}장"
        f"{', LIMIT_IMAGES=' + str(LIMIT_IMAGES) if LIMIT_IMAGES > 0 else ''})",
        "",
        f"pointing game (히트맵 피크가 GT bbox 안) : {pointing:.4f}",
        f"  - 불량 정분류(p>=0.5)에서만           : {pointing_when_correct:.4f}",
        f"에너지 집중도 (히트맵 총합 중 bbox 내부) : {energy:.4f}",
        f"  - 참조: bbox 면적 비율(무작위 기대치)  : {area:.4f}",
        "",
        "해석 기준: pointing game이 bbox 면적 비율보다 크게 높을수록",
        "히트맵이 결함을 실제로 가리키는 것. 상세는 layercam_scores.csv",
        "(metadata와 join하면 그룹/원등급별 슬라이스 채점 가능).",
    ]
    (output_dir / "layercam_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print("\n오버레이 저장 :", output_dir, f"({n_overlay_saved}장)")


if __name__ == "__main__":
    main()
