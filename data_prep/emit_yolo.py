from pathlib import Path
import yaml
from shared.labels import to_yolo_line


def write_label_file(path, boxes, img_w, img_h, class_names):
    """Write YOLO .txt lines for the boxes that belong to class_names."""
    lines = [to_yolo_line(b, img_w, img_h, class_names)
             for b in boxes if b.name in class_names]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(root, version, class_names):
    """Write a YOLO data.yaml pointing at the orig/ or clahe/ image variant."""
    root = Path(root)
    data = {
        "path": str(root),
        "train": f"images/train/{version}",
        "val": f"images/val/{version}",
        "test": f"images/test/{version}",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    out = root / f"data_{version}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(out)
