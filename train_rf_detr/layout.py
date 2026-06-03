import os
import shutil
from pathlib import Path

# rfdetr uses "valid" not "val"
_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


def build_rfdetr_view(subset_db: Path, version: str, dest: Path) -> Path:
    """Create dataset/{train,valid,test}/_annotations.coco.json + image symlinks
    from our COCO db, without copying image bytes."""
    subset_db, dest = Path(subset_db), Path(dest)
    for split, rf_split in _SPLIT_MAP.items():
        ann = subset_db / "annotations" / f"instances_{split}_{version}.json"
        if not ann.exists():
            continue
        out_dir = dest / rf_split
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(ann, out_dir / "_annotations.coco.json")
        img_src = subset_db / "images" / split / version
        for img in img_src.glob("*.jpg"):
            if img.name.startswith("._"):
                continue
            link = out_dir / img.name
            if not link.exists():
                os.symlink(img.resolve(), link)
    return dest
