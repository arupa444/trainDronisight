import json
from pathlib import Path


def build_coco(annotations: dict, class_names: list) -> dict:
    """Build a COCO dict. annotations: {image_filename: Annotation}."""
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    images, anns = [], []
    ann_id = 1
    for img_id, (fname, ann) in enumerate(sorted(annotations.items()), start=1):
        images.append({"id": img_id, "file_name": fname,
                       "width": ann.width, "height": ann.height})
        for b in ann.boxes:
            if b.name not in class_names:
                continue
            anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_names.index(b.name),
                "bbox": [b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin],
                "area": (b.xmax - b.xmin) * (b.ymax - b.ymin),
                "iscrowd": 0,
            })
            ann_id += 1
    return {"images": images, "annotations": anns, "categories": categories}


def write_coco(path, coco: dict):
    Path(path).write_text(json.dumps(coco))
