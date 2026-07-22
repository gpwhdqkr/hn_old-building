from pathlib import Path
from PIL import Image

source_dir = Path(r"D:\hn_old-building_raw\raw\images")
save_dir = Path(r"D:\hn_old-building_raw\processed_images")

for image_path in source_dir.rglob("*"):
    if image_path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png"
    }:
        continue

    relative_path = image_path.relative_to(source_dir)
    saved_path = save_dir / relative_path

    saved_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((224,224))
        image.save(saved_path)