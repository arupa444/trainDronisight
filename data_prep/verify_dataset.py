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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["pole", "components"], required=True)
    args = ap.parse_args()
    manifest = config.YOLO_DB / args.subset / "manifest.csv"
    df = pd.read_csv(manifest)
    assert_no_group_leakage(df)
    print("OK no leakage. Split sizes:", class_counts_from_manifest(df))
    # box validity: every YOLO label coord in [0,1]
    for txt in (config.YOLO_DB / args.subset / "labels").rglob("*.txt"):
        for line in txt.read_text().split():
            pass  # presence check; full numeric check below
    bad = []
    for txt in (config.YOLO_DB / args.subset / "labels").rglob("*.txt"):
        for ln in txt.read_text().splitlines():
            vals = ln.split()
            if len(vals) != 5 or not all(0.0 <= float(v) <= 1.0 for v in vals[1:]):
                bad.append(str(txt))
                break
    assert not bad, f"invalid YOLO labels: {bad[:5]}"
    print("OK all YOLO labels valid.")


if __name__ == "__main__":
    main()
