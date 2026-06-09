"""End-to-end build_subset: the orchestration the unit tests never exercise.

Builds tiny on-disk fixtures and asserts the written DBs are correct for the two crop modes:
  * `component`  (anchor mode) — crops to the pole box; the byte-identical cross-folder pair MERGES.
  * `cond_*`     (self mode)   — crops to each component's own box.
Checks both DBs created, crops smaller than the frame, YOLO↔COCO splits agree, labels valid,
no image-content leakage across splits.
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
    cv2.imwrite(str(path), np.zeros((H, W, 3), np.uint8) + val)


@pytest.fixture
def comp_db(tmp_path, monkeypatch):
    """mem-style fixture for a component specialist (`comp_insulator`, anchor/pole-crop build)."""
    src_a, src_b = tmp_path / "memA", tmp_path / "memB"
    src_a.mkdir(); src_b.mkdir()
    # 3 distinct images, each with a POLE box (sub-region) + an insulator on it
    for i, cls in enumerate(["h_insulator", "v_insulator", "h_insulator"]):
        _write_img(src_a / f"IMG_{i}.JPG", val=30 + i * 40)
        (src_a / f"IMG_{i}.xml").write_text(_voc(W, H, [("pole", 0, 0, 40, 40), (cls, 2, 2, 30, 30)]))
    # byte-identical photo in BOTH folders, different classes -> must MERGE to union {h_insulator, v_insulator}
    _write_img(src_a / "SHARED.JPG", val=200)
    shutil.copy(src_a / "SHARED.JPG", src_b / "SHARED.JPG")
    (src_a / "SHARED.xml").write_text(_voc(W, H, [("pole", 0, 0, 42, 42), ("h_insulator", 5, 5, 40, 40)]))
    (src_b / "SHARED.xml").write_text(_voc(W, H, [("pole", 0, 0, 42, 42), ("v_insulator", 6, 6, 41, 41)]))
    monkeypatch.setattr(config, "YOLO_DB", tmp_path / "yolo_db")
    monkeypatch.setattr(config, "COCO_DB", tmp_path / "coco_db")
    monkeypatch.setattr(config, "SUBSET_SOURCE_DIRS",
                        {**config.SUBSET_SOURCE_DIRS, "comp_insulator": [src_a, src_b]})
    return tmp_path


def test_component_build_crops_merges_and_syncs_dbs(comp_db):
    build_dataset.build_subset("comp_insulator", balance=False)
    ydb, cdb = config.YOLO_DB / "comp_insulator", config.COCO_DB / "comp_insulator"
    assert (ydb / "manifest.csv").exists() and (cdb / "dataset_meta.json").exists()
    meta = json.loads((ydb / "dataset_meta.json").read_text())
    assert meta["crop_aligned"] is True and meta["crop_mode"] == "anchor"
    assert meta["cross_folder_merge"]["duplicate_copies_collapsed"] == 1
    assert meta["cross_folder_merge"]["unique_images"] == 4          # 3 distinct + 1 merged shared

    df = pd.read_csv(ydb / "manifest.csv")
    for _, row in df.iterrows():                                     # both variants + mirrored label
        for v in ("orig", "clahe"):
            assert (ydb / "images" / row["split"] / v / f"{row['name']}.jpg").exists()
            assert (ydb / "labels" / row["split"] / v / f"{row['name']}.txt").exists()

    # every written image is a CROP (smaller than the 64x48 frame)
    for p in (ydb / "images").rglob("*.jpg"):
        h, w = cv2.imread(str(p)).shape[:2]
        assert w < W or h < H

    # the merged SHARED crop carries the UNION of both members' classes (exclude augmented copies)
    shared = df[df["name"].str.contains("SHARED") & ~df["augmented"].astype(bool)]
    assert len(shared) == 1
    lbl = (ydb / "labels" / shared.iloc[0]["split"] / "orig" / f"{shared.iloc[0]['name']}.txt").read_text().split("\n")
    idxs = sorted(int(l.split()[0]) for l in lbl if l.strip())
    assert idxs == sorted([config.COMP_INSULATOR_CLASSES.index("h_insulator"),
                           config.COMP_INSULATOR_CLASSES.index("v_insulator")])

    # YOLO and COCO splits agree; no image-content leakage
    from data_prep.verify_dataset import assert_no_image_content_leakage, find_invalid_labels
    for split in ("train", "val", "test"):
        ydir = ydb / "images" / split / "orig"
        ystems = {p.stem for p in ydir.glob("*.jpg")} if ydir.is_dir() else set()
        cj = cdb / "annotations" / f"instances_{split}_orig.json"
        cstems = {Path(im["file_name"]).stem for im in json.loads(cj.read_text())["images"]} if cj.exists() else set()
        assert ystems == cstems
    assert not find_invalid_labels(ydb / "labels")
    assert_no_image_content_leakage("comp_insulator")


def test_cond_self_build_crops_to_each_component(tmp_path, monkeypatch):
    """6th-june-style fixture for a `cond_*` self-mode build (crop to each condition box)."""
    src = tmp_path / "6thMemX"; src.mkdir()
    for i in range(3):
        _write_img(src / f"C_{i}.JPG", val=40 + i * 30)
        (src / f"C_{i}.xml").write_text(_voc(W, H, [
            ("v_insulator_normal", 4, 4, 24, 24), ("v_insulator_broken", 36, 30, 56, 46)]))
    monkeypatch.setattr(config, "YOLO_DB", tmp_path / "yolo_db")
    monkeypatch.setattr(config, "COCO_DB", tmp_path / "coco_db")
    monkeypatch.setattr(config, "SUBSET_SOURCE_DIRS",
                        {**config.SUBSET_SOURCE_DIRS, "cond_v_insulator": [src]})
    build_dataset.build_subset("cond_v_insulator", balance=False)
    ydb = config.YOLO_DB / "cond_v_insulator"
    meta = json.loads((ydb / "dataset_meta.json").read_text())
    assert meta["crop_aligned"] is True and meta["crop_mode"] == "self"
    imgs = list((ydb / "images").rglob("*.jpg"))
    assert imgs and all(cv2.imread(str(p)).shape[1] < W for p in imgs)   # each is a component crop
    from data_prep.verify_dataset import find_invalid_labels, assert_no_image_content_leakage
    assert not find_invalid_labels(ydb / "labels")
    assert_no_image_content_leakage("cond_v_insulator")
