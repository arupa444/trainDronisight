"""Build the two thermal-cascade YOLO datasets, reproducibly, from the 4-annotator
labelImg folders. Self-contained port of the trainDronisight data-prep design,
scoped to two single-class subsets:

  transformer : single-class `transformer`, FULL FRAME (the pole analog).
  wire        : single-class `wire`, cropped to each transformer box + pad
                (== the cascade inference: detect transformer -> crop -> detect wire).

Each dataset gets the pole-style layout: images|labels / {train,val,test} /
{orig,clahe}, data_orig.yaml + data_clahe.yaml, manifest.csv, dataset_meta.json.
CLAHE is applied to the FULL frame then sliced, so crop pixels match inference.

Usage (from repo root, repo venv active):
    python -m thermal.data_prep.build --subset both
"""
import argparse
import json
import os
from pathlib import Path

import cv2

from thermal.data_prep.labels import parse_voc, to_yolo_line
from thermal.data_prep.imaging import (load_oriented_bgr, profile_array,
                                       clahe_params_from_profile, apply_clahe)
from thermal.data_prep.assemble import (collect_samples, merge_by_image_identity,
                                        assign_groups, grouped_split, make_anchor_crops)

SEED = 1337
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}
GROUP_TIME_GAP_S = 60
WIRE_CROP_PAD = 0.15
WIRE_MIN_VISIBLE = 0.30

SSD = Path(os.environ.get("DRONISIGHT_DATA", "/Volumes/dronisight"))
_THERMAL = SSD / "thermal"
THERMAL_DIRS = [
    _THERMAL / "JUNE 11 MEM 1" / "_T_collected",
    _THERMAL / "Jun 11 Mem 2 ",          # trailing space is part of the real name
    _THERMAL / "JUNE 11 MEM 3" / "_T_collected",
    _THERMAL / "JUNE 11 MEM 4",
]

SUBSETS = {
    "transformer": {"out": SSD / "yolo_thermal_transformer", "classes": ["transformer"],
                    "crop": False},
    "wire": {"out": SSD / "Yolo_thermal_wire", "classes": ["wire"], "crop": True},
}


def _save(bgr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _write_yaml(out: Path, version: str, class_names):
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    (out / f"data_{version}.yaml").write_text(
        f"path: {out}\ntrain: images/train/{version}\nval: images/val/{version}\n"
        f"test: images/test/{version}\nnames:\n{names}\n")


def build_subset(name: str):
    cfg = SUBSETS[name]
    classes, out, is_crop = cfg["classes"], cfg["out"], cfg["crop"]
    keep = set(classes)

    # collect + parse + keep images with >=1 in-subset class
    parsed = {}
    bad = 0
    for s in collect_samples(THERMAL_DIRS):
        try:
            ann = parse_voc(s.xml)
        except Exception:
            bad += 1
            continue
        if any(b.name in keep for b in ann.boxes) or (is_crop and any(b.name == "transformer" for b in ann.boxes)):
            parsed[s.image] = (s, ann)

    parsed, merge_stats = merge_by_image_identity(parsed)

    # groups per source
    by_source = {}
    for img, (s, _) in parsed.items():
        by_source.setdefault(s.source, []).append(img.name)
    name_to_group = {}
    for source, names in by_source.items():
        name_to_group.update(assign_groups(names, source, GROUP_TIME_GAP_S))

    # items (full-frame image, or one per transformer crop)
    items = []
    crop_ann = {}
    for img, (s, ann) in parsed.items():
        base = f"{s.source}_{img.stem}"
        grp = name_to_group[img.name]
        if is_crop:
            for ci, (cbox, cann) in enumerate(make_anchor_crops(
                    ann, classes, ["transformer"], WIRE_CROP_PAD, WIRE_MIN_VISIBLE)):
                key = f"{base}_c{ci}"
                crop_ann[key] = (cbox, cann)
                items.append({"image": img, "key": key, "group": grp, "source": s.source})
        else:
            if any(b.name in keep for b in ann.boxes):
                items.append({"image": img, "key": base, "group": grp, "source": s.source})

    split = grouped_split(items, SPLIT_RATIOS, SEED)

    manifest = ["key,source,split"]
    n_written = skipped_dim = 0
    box_count = 0
    for split_name, split_items in split.items():
        for it in split_items:
            s, full_ann = parsed[it["image"]]
            bgr_full = load_oriented_bgr(s.image)
            h, w = bgr_full.shape[:2]
            if (w, h) != (full_ann.width, full_ann.height):
                skipped_dim += 1
                continue
            clip, grid = clahe_params_from_profile(profile_array(bgr_full))
            clahe_full = apply_clahe(bgr_full, clip, grid)
            if is_crop:
                x0, y0, x1, y1 = crop_ann[it["key"]][0]
                orig_img, clahe_img = bgr_full[y0:y1, x0:x1], clahe_full[y0:y1, x0:x1]
                ann = crop_ann[it["key"]][1]
            else:
                orig_img, clahe_img, ann = bgr_full, clahe_full, full_ann
            for variant, im in (("orig", orig_img), ("clahe", clahe_img)):
                _save(im, out / "images" / split_name / variant / f"{it['key']}.jpg")
                lines = [to_yolo_line(b, ann.width, ann.height, classes)
                         for b in ann.boxes if b.name in keep]
                lf = out / "labels" / split_name / variant / f"{it['key']}.txt"
                lf.parent.mkdir(parents=True, exist_ok=True)
                lf.write_text("\n".join(lines) + ("\n" if lines else ""))
            box_count += sum(1 for b in ann.boxes if b.name in keep)
            manifest.append(f"{it['key']},{it['source']},{split_name}")
            n_written += 1

    for v in ("orig", "clahe"):
        _write_yaml(out, v, classes)
    (out / "manifest.csv").write_text("\n".join(manifest) + "\n")
    meta = {"subset": name, "classes": classes, "crop_aligned": is_crop,
            "n_written_items": n_written, "n_boxes": box_count,
            "split_counts": {k: len(v) for k, v in split.items()},
            "skipped_dim_mismatch": skipped_dim, "skipped_bad_xml": bad,
            "merge": merge_stats}
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] {n_written} items, {box_count} boxes -> {out}")
    print(f"[{name}] merge: {merge_stats['unique_images']} unique from "
          f"{merge_stats['input_copies']} copies "
          f"({merge_stats['images_spanning_multiple_folders']} spanned >1 folder)")
    print(f"[{name}] split: {meta['split_counts']}  skipped_dim={skipped_dim} bad_xml={bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["transformer", "wire", "both"], default="both")
    args = ap.parse_args()
    todo = ["transformer", "wire"] if args.subset == "both" else [args.subset]
    for s in todo:
        build_subset(s)


if __name__ == "__main__":
    main()
