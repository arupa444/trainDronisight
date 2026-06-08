"""Validate a built DB. Usage: python -m data_prep.verify_dataset --subset components"""
import argparse
from pathlib import Path

import pandas as pd

from shared import config


def assert_no_group_leakage(df: pd.DataFrame):
    spans = df.groupby("group")["split"].nunique()
    bad = spans[spans > 1]
    assert bad.empty, f"capture groups span multiple splits: {list(bad.index)}"


def class_counts_from_manifest(df: pd.DataFrame) -> dict:
    return df["split"].value_counts().to_dict()


def assert_no_image_content_leakage(subset: str):
    """No physical image (by byte content) may appear in more than one split.

    For the per-annotator subsets (6th-june condition) the same photo lives in several
    member folders; the build merges them, and this re-checks the WRITTEN DB to prove a
    photo didn't slip into both train and val/test. Scoped to merge subsets so the mem*
    builds don't pay the hashing cost. Augmented (`_augN`) copies are train-only by design
    and derive from a train image, so they can't cross splits -- still hashed for safety."""
    from data_prep.merge_annotations import image_content_hash

    seen = {}  # content hash -> split
    collisions = []
    base = config.YOLO_DB / subset / "images"
    for split in ("train", "val", "test"):
        d = base / split / "orig"
        if not d.is_dir():
            continue
        for img in d.glob("*.jpg"):
            if img.name.startswith("._"):
                continue
            h = image_content_hash(img)
            prev = seen.get(h)
            if prev is not None and prev != split:
                collisions.append((img.name, prev, split))
            else:
                seen[h] = split
    assert not collisions, f"image content leaks across splits: {collisions[:5]}"
    print(f"OK no image-content leakage across splits ({len(seen)} unique images).")


def find_invalid_labels(labels_dir) -> list:
    """Return paths of YOLO label files that are malformed. Skips AppleDouble (._*) sidecars."""
    bad = []
    for txt in Path(labels_dir).rglob("*.txt"):
        if txt.name.startswith("._"):
            continue
        ok = True
        for ln in txt.read_text().splitlines():
            vals = ln.split()
            if len(vals) != 5:
                ok = False; break
            try:
                if not all(0.0 <= float(v) <= 1.0 for v in vals[1:]):
                    ok = False; break
            except ValueError:
                ok = False; break
        if not ok:
            bad.append(str(txt))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=config.SUBSETS, required=True)
    args = ap.parse_args()
    manifest = config.YOLO_DB / args.subset / "manifest.csv"
    df = pd.read_csv(manifest)
    assert_no_group_leakage(df)
    print("OK no group leakage. Split sizes:", class_counts_from_manifest(df))
    if config.MERGE_CROSS_FOLDER.get(args.subset, True):
        assert_no_image_content_leakage(args.subset)
    # box validity: every YOLO label coord in [0,1]
    bad = find_invalid_labels(config.YOLO_DB / args.subset / "labels")
    assert not bad, f"invalid YOLO labels: {bad[:5]}"
    print("OK all YOLO labels valid.")


if __name__ == "__main__":
    main()
