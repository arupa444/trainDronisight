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
    ap.add_argument("--subset", choices=["pole", "components"], required=True)
    args = ap.parse_args()
    manifest = config.YOLO_DB / args.subset / "manifest.csv"
    df = pd.read_csv(manifest)
    assert_no_group_leakage(df)
    print("OK no leakage. Split sizes:", class_counts_from_manifest(df))
    # box validity: every YOLO label coord in [0,1]
    bad = find_invalid_labels(config.YOLO_DB / args.subset / "labels")
    assert not bad, f"invalid YOLO labels: {bad[:5]}"
    print("OK all YOLO labels valid.")


if __name__ == "__main__":
    main()
