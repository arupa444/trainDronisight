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
from shared.labels import parse_voc, Annotation
from data_prep.collect import collect_samples
from data_prep.grouping import assign_groups
from data_prep.split import grouped_split
from data_prep.balance import select_balanced, sample_weights
from data_prep.dedup import drop_duplicate_annotations
from data_prep.merge_annotations import merge_by_image_identity, resolve_cross_class_conflicts
from data_prep.oversample import plan_oversample, augment_image
from data_prep.profile_images import profile_array
from data_prep.preprocess import load_oriented_bgr, clahe_params_from_profile, apply_clahe
from data_prep.emit_yolo import write_label_file, write_data_yaml
from data_prep.emit_coco import build_coco, write_coco


def sample_class_list(subset: str):
    return config.SUBSET_CLASSES[subset]


def output_key(source: str, stem: str) -> str:
    """Unique per-image key namespaced by source folder (DJI counters reset per card)."""
    return f"{source}_{stem}"


def yolo_label_paths(subset: str, split: str, key: str):
    """YOLO label paths that MIRROR the image variant dirs (labels/<split>/<orig|clahe>/).

    Ultralytics resolves an image's label by substituting the last '/images/' with
    '/labels/' in its path, so labels must live under the same <split>/<variant>
    nesting as the images. Boxes are identical across orig/clahe, so the same label
    is written into both variant dirs.
    """
    return [config.YOLO_DB / subset / "labels" / split / v / f"{key}.txt"
            for v in ("orig", "clahe")]


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


def _write_sample(subset, split_name, key, bgr, clahe_img, ann, class_names,
                  coco_per_split, manifest_rows, source, group, prof, clip, augmented=False):
    """Write one image (orig + clahe) to both DBs, its mirrored YOLO labels, and record
    it in the per-split COCO map + manifest."""
    for db_root in (config.YOLO_DB, config.COCO_DB):
        base = db_root / subset / "images" / split_name
        _save(bgr, base / "orig" / f"{key}.jpg")
        _save(clahe_img, base / "clahe" / f"{key}.jpg")
    for lbl in yolo_label_paths(subset, split_name, key):
        lbl.parent.mkdir(parents=True, exist_ok=True)
        write_label_file(lbl, ann.boxes, ann.width, ann.height, class_names)
    coco_per_split[split_name][f"{key}.jpg"] = ann
    manifest_rows.append({"name": key, "source": source, "group": group,
                          "split": split_name, "subset": subset, "augmented": augmented,
                          **prof, "clahe_clip": clip})


def build_subset(subset: str, balance: bool):
    class_names = sample_class_list(subset)
    # Balance modes:
    #   target (e.g. classification=400): split raw, then on TRAIN cap each class DOWN to the
    #     target and augment under-target classes UP to it; val/test stay raw.
    #   below_1000: keep all, oversample TRAIN up to the max class count.
    #   else (pole/above): legacy down-cap toward the rarest kept class.
    target = config.BALANCE_TARGET.get(subset)
    do_target = target is not None
    do_balance = (balance and config.BALANCE_CAP_ENABLED
                  and not do_target and subset != "component_below_1000")
    do_oversample = subset == "component_below_1000"
    samples = collect_samples(config.SUBSET_SOURCE_DIRS.get(subset, config.SOURCE_DIRS))

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

    # dedup re-annotated images shared across configured folder pairs (e.g. mem7 / mem 7.1)
    parsed, n_dedup = drop_duplicate_annotations(parsed, config.DEDUP_PAIRS)

    # Collapse byte-identical copies of one physical image into ONE entry holding the
    # UNION of all copies' boxes (keyed on content hash). Critical for the per-annotator
    # 6th-june data (same photo in several member folders, each with only partial labels)
    # and for the mem7/mem7.1 byte-identical overlap; a pure hashing no-op on disjoint
    # captures. Runs BEFORE grouping/splitting so a photo can never leak across splits.
    merge_stats = None
    if config.MERGE_CROSS_FOLDER.get(subset, True):
        parsed, merge_stats = merge_by_image_identity(parsed)

    # Condition-conflict resolution (component_classification): when members gave the SAME
    # object different condition labels, defect beats normal and defect-vs-defect is dropped
    # as ambiguous. Drop any image left with no in-subset boxes afterwards.
    cond_conflict_stats = None
    if config.RESOLVE_CONDITION_CONFLICTS.get(subset):
        n_over = n_drop = 0
        empties = []
        for img, (s, ann) in parsed.items():
            rb, no, nd = resolve_cross_class_conflicts(ann.boxes)
            n_over += no
            n_drop += nd
            if any(b.name in class_names for b in rb):
                parsed[img] = (s, Annotation(ann.width, ann.height, rb))
            else:
                empties.append(img)
        for img in empties:
            del parsed[img]
        cond_conflict_stats = {"normal_overridden_by_defect": n_over,
                               "ambiguous_objects_dropped": n_drop,
                               "images_emptied_and_dropped": len(empties)}

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

    if do_balance:
        items = select_balanced(items, class_names, enabled=True, seed=config.SEED)

    split = grouped_split(items, config.SPLIT_RATIOS, config.SEED)
    version_hash = dataset_version_hash([it["key"] for it in items])

    # write both DBs
    coco_per_split = {"train": {}, "val": {}, "test": {}}
    manifest_rows = []
    skipped_dim = 0
    n_aug = 0
    for split_name, split_items in split.items():
        if do_target and split_name == "train":
            # cap each class DOWN to the target by dropping excess train images (val/test stay raw)
            split_items = select_balanced(split_items, class_names, enabled=True,
                                          seed=config.SEED, cap=target)
        written = []
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
            _write_sample(subset, split_name, key, bgr, clahe_img, ann, class_names,
                          coco_per_split, manifest_rows, it["source"], it["group"], prof, clip)
            written.append(it)

        # oversample TRAIN with bbox-aware augmentation: below_1000 -> equalize to the max class
        # count (target=None); component_classification -> bring under-target classes up to target.
        if (do_oversample or do_target) and split_name == "train" and written:
            for j, idx in enumerate(plan_oversample(written, class_names, seed=config.SEED, target=target)):
                it = written[idx]
                s, ann = parsed[it["image"]]
                bgr = load_oriented_bgr(s.image)
                hh, ww = bgr.shape[:2]
                if (ww, hh) != (ann.width, ann.height):
                    continue
                sub_boxes = [b for b in ann.boxes if b.name in class_names]
                a_img, a_boxes = augment_image(bgr, sub_boxes, seed_n=config.SEED + j + 1)
                if not a_boxes:
                    continue
                ah, aw = a_img.shape[:2]
                prof = profile_array(a_img)
                clip, grid = clahe_params_from_profile(prof)
                a_clahe = apply_clahe(a_img, clip, grid)
                key = f"{output_key(it['source'], it['image'].stem)}_aug{j}"
                _write_sample(subset, "train", key, a_img, a_clahe,
                              Annotation(aw, ah, a_boxes), class_names,
                              coco_per_split, manifest_rows, it["source"], it["group"],
                              prof, clip, augmented=True)
                n_aug += 1

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
    balance_mode = ("target" if do_target else "cap_rarest" if do_balance
                    else "oversample_max" if do_oversample else "none")
    # n_images counts UNIQUE SOURCE images (post-merge/resolve, pre train-cap/augmentation);
    # n_written_images is what actually landed on disk (after the train cap + augmented copies).
    meta = {"subset": subset, "version_hash": version_hash,
            "n_unique_source_images": len(items), "n_images": len(items),
            "n_written_images": len(manifest_rows),
            "balance_mode": balance_mode,
            "balance_target": target, "oversampled_train": n_aug,
            "skipped_dim_mismatch": skipped_dim, "skipped_bad_xml": skipped_bad_xml,
            "cross_folder_merge": merge_stats,
            "condition_conflict_resolution": cond_conflict_stats,
            "class_names": class_names}
    for db in (config.YOLO_DB, config.COCO_DB):
        (db / subset).mkdir(parents=True, exist_ok=True)
        (db / subset / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    # self-clean macOS AppleDouble sidecars so future builds leave clean DBs
    n = clean_appledouble(config.YOLO_DB / subset) + clean_appledouble(config.COCO_DB / subset)
    print(f"[{subset}] {len(items)} images (+{n_aug} augmented train), version {version_hash}")
    if merge_stats:
        print(f"[{subset}] cross-folder merge: {merge_stats['unique_images']} unique images "
              f"from {merge_stats['input_copies']} member copies "
              f"({merge_stats['images_spanning_multiple_folders']} spanned >1 folder, "
              f"{merge_stats['duplicate_copies_collapsed']} copies collapsed, "
              f"+{merge_stats['boxes_added_by_union']} boxes unioned, "
              f"{merge_stats['overlapping_boxes_removed']} dup boxes removed)")
    if cond_conflict_stats:
        print(f"[{subset}] condition conflicts: "
              f"{cond_conflict_stats['normal_overridden_by_defect']} normal->defect overrides, "
              f"{cond_conflict_stats['ambiguous_objects_dropped']} ambiguous objects dropped, "
              f"{cond_conflict_stats['images_emptied_and_dropped']} images emptied")
    print(f"[{subset}] dedup dropped {n_dedup} duplicate-annotation images")
    print(f"[{subset}] skipped {skipped_dim} images: EXIF/XML dimension mismatch")
    print(f"[{subset}] skipped {skipped_bad_xml} unparseable XML files")
    print(f"[{subset}] removed {n} AppleDouble sidecar files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=config.SUBSETS + ["all"], required=True)
    ap.add_argument("--no-balance", action="store_true")
    args = ap.parse_args()
    subsets = list(config.SUBSETS) if args.subset == "all" else [args.subset]
    for sub in subsets:
        build_subset(sub, balance=not args.no_balance)


if __name__ == "__main__":
    main()
