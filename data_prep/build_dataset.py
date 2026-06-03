"""Orchestrate the full data-prep into both DBs.

Usage:
    python -m data_prep.build_dataset --subset pole
    python -m data_prep.build_dataset --subset components
    python -m data_prep.build_dataset --subset all [--no-balance]
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd

from shared import config
from shared.labels import parse_voc
from data_prep.collect import collect_samples
from data_prep.grouping import assign_groups
from data_prep.split import grouped_split
from data_prep.balance import select_balanced, sample_weights
from data_prep.profile_images import profile_array
from data_prep.preprocess import load_oriented_bgr, clahe_params_from_profile, apply_clahe
from data_prep.emit_yolo import write_label_file, write_data_yaml
from data_prep.emit_coco import build_coco, write_coco


def sample_class_list(subset: str):
    return config.POLE_CLASSES if subset == "pole" else config.COMPONENT_CLASSES


def output_key(source: str, stem: str) -> str:
    """Unique per-image key namespaced by source folder (DJI counters reset per card)."""
    return f"{source}_{stem}"


def dataset_version_hash(image_keys) -> str:
    h = hashlib.sha256()
    for k in sorted(image_keys):
        h.update(k.encode())
    return h.hexdigest()[:12]


def clean_appledouble(root) -> int:
    """Delete macOS AppleDouble (._*) sidecar files under root. Returns count removed."""
    removed = 0
    for p in Path(root).rglob("._*"):
        if p.is_file():
            p.unlink()
            removed += 1
    return removed


def _save(bgr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def build_subset(subset: str, balance: bool):
    class_names = sample_class_list(subset)
    samples = collect_samples(config.SOURCE_DIRS)

    # parse + keep only images that contain >=1 class for this subset
    parsed = {}
    skipped_bad_xml = 0
    for s in samples:
        try:
            ann = parse_voc(s.xml)
        except Exception:
            skipped_bad_xml += 1
            continue
        if any(b.name in class_names for b in ann.boxes):
            parsed[s.image] = (s, ann)

    # groups (per source) -> items
    by_source = {}
    for img in parsed:
        by_source.setdefault(parsed[img][0].source, []).append(img.name)
    name_to_group = {}
    for source, names in by_source.items():
        name_to_group.update(assign_groups(names, source, config.GROUP_TIME_GAP_S))

    items = [{"image": img, "name": img.name,
              "key": output_key(parsed[img][0].source, img.stem),
              "group": name_to_group[img.name],
              "source": parsed[img][0].source,
              "classes": [b.name for b in parsed[img][1].boxes if b.name in class_names]}
             for img in parsed]

    if balance and config.BALANCE_CAP_ENABLED:
        items = select_balanced(items, class_names, enabled=True, seed=config.SEED)

    split = grouped_split(items, config.SPLIT_RATIOS, config.SEED)
    version_hash = dataset_version_hash([it["key"] for it in items])

    # write both DBs
    coco_per_split = {"train": {}, "val": {}, "test": {}}
    manifest_rows = []
    skipped_dim = 0
    for split_name, split_items in split.items():
        for it in split_items:
            s, ann = parsed[it["image"]]
            bgr = load_oriented_bgr(s.image)
            h, w = bgr.shape[:2]
            if (w, h) != (ann.width, ann.height):
                skipped_dim += 1
                continue
            prof = profile_array(bgr)
            clip, grid = clahe_params_from_profile(prof)
            clahe_img = apply_clahe(bgr, clip, grid)

            key = output_key(it["source"], it["image"].stem)
            for db_root in (config.YOLO_DB, config.COCO_DB):
                base = db_root / subset / "images" / split_name
                _save(bgr, base / "orig" / f"{key}.jpg")
                _save(clahe_img, base / "clahe" / f"{key}.jpg")

            # YOLO labels (shared by orig/clahe)
            lbl = config.YOLO_DB / subset / "labels" / split_name / f"{key}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            write_label_file(lbl, ann.boxes, ann.width, ann.height, class_names)

            coco_per_split[split_name][f"{key}.jpg"] = ann
            manifest_rows.append({"name": key, "source": it["source"],
                                  "group": it["group"], "split": split_name,
                                  "subset": subset, **prof, "clahe_clip": clip})

        # YOLO data.yaml (orig + clahe)
        for version in ("orig", "clahe"):
            write_data_yaml(config.YOLO_DB / subset, version, class_names)
        # COCO json (orig + clahe; boxes identical, only image dir differs)
        coco = build_coco(coco_per_split[split_name], class_names)
        for version in ("orig", "clahe"):
            ann_dir = config.COCO_DB / subset / "annotations"
            ann_dir.mkdir(parents=True, exist_ok=True)
            write_coco(ann_dir / f"instances_{split_name}_{version}.json", coco)

    # manifests + version + sampling weights
    df = pd.DataFrame(manifest_rows)
    (config.YOLO_DB / subset).mkdir(parents=True, exist_ok=True)
    df.to_csv(config.YOLO_DB / subset / "manifest.csv", index=False)
    weights = sample_weights(items, class_names)
    pd.DataFrame({"name": [it["key"] for it in items], "weight": weights}) \
        .to_csv(config.YOLO_DB / subset / "sample_weights.csv", index=False)
    meta = {"subset": subset, "version_hash": version_hash,
            "n_images": len(items), "balanced": bool(balance and config.BALANCE_CAP_ENABLED),
            "class_names": class_names}
    for db in (config.YOLO_DB, config.COCO_DB):
        (db / subset).mkdir(parents=True, exist_ok=True)
        (db / subset / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    # self-clean macOS AppleDouble sidecars so future builds leave clean DBs
    n = clean_appledouble(config.YOLO_DB / subset) + clean_appledouble(config.COCO_DB / subset)
    print(f"[{subset}] {len(items)} images, version {version_hash}")
    print(f"[{subset}] skipped {skipped_dim} images: EXIF/XML dimension mismatch")
    print(f"[{subset}] skipped {skipped_bad_xml} unparseable XML files")
    print(f"[{subset}] removed {n} AppleDouble sidecar files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["pole", "components", "all"], required=True)
    ap.add_argument("--no-balance", action="store_true")
    args = ap.parse_args()
    subsets = ["pole", "components"] if args.subset == "all" else [args.subset]
    for sub in subsets:
        build_subset(sub, balance=not args.no_balance)


if __name__ == "__main__":
    main()
