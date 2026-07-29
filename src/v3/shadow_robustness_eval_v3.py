# -*- coding: utf-8 -*-
"""음영 강건성 측정: 학습된 binclf_v3 모델을 합성 음영 조건별로 test 평가.

학습 없음, 측정 전용. 서비스에서 들어올 음영/저조도 사진에 대한 강건성을
합성 조건(그림자 다각형, 감마 저조도)으로 근사 측정한다.
조건×이미지별 고정 시드(crc32)라 재현 가능.

실행 (환경변수는 다른 v3 스크립트와 동일 규약):
    DATA_DIR=... ARCH=convnext_tiny RUN_NAME=v3cta python src/v3/shadow_robustness_eval_v3.py
    (PowerShell) $env:DATA_DIR="D:/hn_old-building"; python src/v3/shadow_robustness_eval_v3.py

결과: DATA_DIR/test_results/binclf_v3/<ARCH>/<RUN_NAME>_shadoweval/shadow_robustness_report.txt

2026-07-29 convnext_tiny v3cta 측정 결과: clean 0.9501 → 최악(강한 그림자+저조도)
macro F1 0.9309 (-0.019). 기존 증강만으로 충분히 강건 → 음영 증강 재학습 불채택.
"""
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_factory_v3 import create_model  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parents[2])))
ARCH = os.environ.get("ARCH", "convnext_tiny")
RUN_NAME = os.environ.get("RUN_NAME", "v3cta")

CKPT = DATA_DIR / "model" / f"best_{ARCH}_binclf_v3_finetuned_{RUN_NAME}.pth"
CACHE = DATA_DIR / "data" / "processed_images_v2"
OUT_DIR = DATA_DIR / "test_results" / "binclf_v3" / ARCH / f"{RUN_NAME}_shadoweval"

INPUT_SIZE = 448
EVAL_RESIZE_SHORT = 512
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))

_post = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def add_polygon_shadows(image, rng, n_polys, factor, blur_radius=25):
    """랜덤 다각형 그림자: 마스크에 다각형을 그리고 블러 후 밝기 곱연산."""
    width, height = image.size
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)
    for _ in range(n_polys):
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)
        n_pts = rng.integers(4, 7)
        radius = rng.uniform(0.25, 0.55) * min(width, height)
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
        pts = [
            (cx + np.cos(a) * radius * rng.uniform(0.6, 1.0),
             cy + np.sin(a) * radius * rng.uniform(0.6, 1.0))
            for a in angles
        ]
        draw.polygon(pts, fill=int(255 * factor))
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    arr = np.asarray(image, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    return Image.fromarray(np.clip(arr * m, 0, 255).astype(np.uint8))


def add_lowlight(image, scale=0.75, gamma=1.8):
    """전역 저조도: 밝기 스케일 + 감마 (어두운 실내/역광 근사)."""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.power(arr * scale, gamma)
    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))


CONDITION_NAMES = ["clean", "shadow_mild", "shadow_strong", "lowlight", "shadow_lowlight"]


def apply_condition(name, image, rng):
    if name == "shadow_mild":
        return add_polygon_shadows(image, rng, 1, 0.55)
    if name == "shadow_strong":
        return add_polygon_shadows(image, rng, 2, 0.35)
    if name == "lowlight":
        return add_lowlight(image)
    if name == "shadow_lowlight":
        return add_lowlight(add_polygon_shadows(image, rng, 2, 0.35))
    return image


class ShadowTestDataset(Dataset):
    def __init__(self, dataframe, condition_name):
        self.df = dataframe
        self.condition_name = condition_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(CACHE / row["image_relpath"]) as f:
            image = f.convert("RGB")
        rng = np.random.default_rng((zlib.crc32(self.condition_name.encode()) + index) % (2**32))
        image = apply_condition(self.condition_name, image, rng)
        image = TF.resize(image, EVAL_RESIZE_SHORT, antialias=True)
        image = TF.center_crop(image, [INPUT_SIZE, INPUT_SIZE])
        return _post(image), int(row["model_label"])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")
    print(f"모델: {CKPT.name}")

    metadata = pd.read_csv(
        DATA_DIR / "data" / "processed" / "metadata_split_v3.csv",
        encoding="utf-8-sig",
    )
    test_df = metadata[metadata["split"] == "test"].reset_index(drop=True)
    print(f"test 이미지: {len(test_df)}장 (우수 {int((test_df.model_label == 0).sum())} / 불량 {int((test_df.model_label == 1).sum())})")

    model = create_model(ARCH, 2, use_pretrained_weights=False)
    checkpoint = torch.load(CKPT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    lines = [
        f"===== 음영 강건성 측정 ({ARCH} {RUN_NAME} finetuned) =====",
        f"test {len(test_df)}장, 조건×이미지 고정 시드(crc32), Resize{EVAL_RESIZE_SHORT}-CenterCrop{INPUT_SIZE}",
        "",
    ]
    results = {}
    for name in CONDITION_NAMES:
        loader = DataLoader(
            ShadowTestDataset(test_df, name), batch_size=BATCH_SIZE,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        preds, labels = [], []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    logits = model(images)
                preds.extend(logits.argmax(1).cpu().tolist())
                labels.extend(targets.tolist())
        f1_per_class = f1_score(labels, preds, average=None)
        macro = f1_score(labels, preds, average="macro")
        accuracy = accuracy_score(labels, preds)
        results[name] = macro
        line = (
            f"{name:16s} | 우수 F1 {f1_per_class[0]:.4f} | 불량 F1 {f1_per_class[1]:.4f}"
            f" | Macro F1 {macro:.4f} | Acc {accuracy:.4f}"
        )
        print(line, flush=True)
        lines.append(line)

    lines.append("")
    lines.append("clean 대비 Macro F1 하락:")
    for name, macro in results.items():
        if name != "clean":
            lines.append(f"  {name:16s} : {macro - results['clean']:+.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = OUT_DIR / "shadow_robustness_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"\n리포트 저장: {report}")


if __name__ == "__main__":
    main()
