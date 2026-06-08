"""End-to-end build_subset: the orchestration the unit tests never exercise.

Builds a tiny on-disk fixture (two source folders, one byte-identical cross-folder pair
carrying DIFFERENT classes) and asserts the written DBs are correct: both DBs created, the
cross-folder copy is MERGED to one entry with the UNION of boxes, YOLO and COCO splits agree,
and every written image has both variants + a label.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from shared import config
from data_prep import build_dataset


W, H = 64, 48


def _voc(width, height, objs):
    body = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
        f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>"
        for (n, x0, y0, x1, y1) in objs)
    return f"<annotation><size><width>{width}</width><height>{height}</height>" \
           f"<depth>3</depth></size>{body}</annotation>"


def _write_img(path, val=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (np.zeros((H, W, 3), np.uint8) + val)
    cv2.imwrite(str(path), img)


@pytest.fixture
def tiny_db(tmp_path, monkeypatch):
    src_a = tmp_path / "memA"
    src_b = tmp_path / "memB"
    src_a.mkdir(); src_b.mkdir()

    # 3 distinct images in A, each its own (untimed) capture group
    for i, cls in enumerate(["wire", "h_insulator", "v_insulator"]):
        stem = f"IMG_{i}"
        _write_img(src_a / f"{stem}.JPG", val=30 + i * 40)
        (src_a / f"{stem}.xml").write_text(_voc(W, H, [(cls, 2, 2, 30, 30)]))

    # a byte-identical photo appears in BOTH folders with DIFFERENT classes ->
    # must merge to ONE entry holding the union {wire, crossarm_stright}
    shared_img = src_a / "SHARED.JPG"
    _write_img(shared_img, val=200)
    shutil.copy(shared_img, src_b / "SHARED.JPG")               # byte-identical
    (src_a / "SHARED.xml").write_text(_voc(W, H, [("wire", 5, 5, 40, 40)]))
    (src_b / "SHARED.xml").write_text(_voc(W, H, [("crossarm_stright", 6, 6, 41, 41)]))

    monkeypatch.setattr(config, "YOLO_DB", tmp_path / "yolo_db")
    monkeypatch.setattr(config, "COCO_DB", tmp_path / "coco_db")
    monkeypatch.setattr(config, "SUBSET_SOURCE_DIRS",
                        {**config.SUBSET_SOURCE_DIRS, "component_above_1000": [src_a, src_b]})
    return tmp_path


def test_build_subset_writes_both_dbs_and_merges(tiny_db):
    build_dataset.build_subset("component_above_1000", balance=False)
    sub = "component_above_1000"
    ydb = config.YOLO_DB / sub
    cdb = config.COCO_DB / sub

    # both DBs + manifest + meta exist
    assert (ydb / "manifest.csv").exists()
    meta = json.loads((ydb / "dataset_meta.json").read_text())
    assert (cdb / "dataset_meta.json").exists()

    # the byte-identical cross-folder pair collapsed to one entry
    assert meta["cross_folder_merge"]["duplicate_copies_collapsed"] == 1
    assert meta["cross_folder_merge"]["unique_images"] == 4   # 3 distinct + 1 shared

    # every written YOLO image has both variants and a mirrored label
    df = pd.read_csv(ydb / "manifest.csv")
    for _, row in df.iterrows():
        split, key = row["split"], row["name"]
        for variant in ("orig", "clahe"):
            assert (ydb / "images" / split / variant / f"{key}.jpg").exists()
            assert (ydb / "labels" / split / variant / f"{key}.txt").exists()

    # the merged SHARED image carries the UNION of both members' classes
    shared_rows = df[df["name"].str.contains("SHARED")]
    assert len(shared_rows) == 1                               # not duplicated across folders
    split = shared_rows.iloc[0]["split"]
    key = shared_rows.iloc[0]["name"]
    label = (ydb / "labels" / split / "orig" / f"{key}.txt").read_text().strip().splitlines()
    classes = config.COMPONENT_ABOVE_CLASSES
    idxs = sorted(int(ln.split()[0]) for ln in label)
    assert idxs == sorted([classes.index("wire"), classes.index("crossarm_stright")])


def test_build_subset_coco_and_yolo_splits_agree(tiny_db):
    build_dataset.build_subset("component_above_1000", balance=False)
    sub = "component_above_1000"
    for split in ("train", "val", "test"):
        ydir = config.YOLO_DB / sub / "images" / split / "orig"
        yolo_stems = {p.stem for p in ydir.glob("*.jpg")} if ydir.is_dir() else set()
        coco_json = config.COCO_DB / sub / "annotations" / f"instances_{split}_orig.json"
        coco_stems = set()
        if coco_json.exists():
            coco = json.loads(coco_json.read_text())
            coco_stems = {Path(im["file_name"]).stem for im in coco["images"]}
        assert yolo_stems == coco_stems, f"{split}: YOLO vs COCO image sets differ"


def test_no_image_content_leakage_after_build(tiny_db):
    from data_prep.verify_dataset import assert_no_image_content_leakage
    build_dataset.build_subset("component_above_1000", balance=False)
    assert_no_image_content_leakage("component_above_1000")  # must not raise
