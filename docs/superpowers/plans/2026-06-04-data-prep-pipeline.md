# Data-Prep Pipeline Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the raw, inconsistently-annotated `mem2`–`mem8` drone imagery into two clean, self-contained training DBs (`yolo_train_db` in YOLO format, `RF_DETR_Faster_RCNN_train_db` in COCO format), each with `pole` and `components` sub-datasets, `orig`+`clahe` image versions, leakage-safe grouped splits, and balanced labels.

**Architecture:** Pure-Python, heavily unit-tested core (`shared/labels.py`) that treats the name-based VOC XML as source of truth; a `data_prep/` package of small single-responsibility modules (collect → group → split → balance → profile → preprocess → emit → verify) orchestrated by one CLI. Source `mem*` folders are never mutated.

**Tech Stack:** Python 3.11+ (via `uv`), `opencv-python`, `numpy`, `pillow`, `scikit-learn`, `pandas`, `pyyaml`, `pytest`. (`torch`/`ultralytics`/`rfdetr` arrive in Plan 2.)

---

## File Structure

```
trainDronisight/
├── pyproject.toml
├── shared/
│   ├── __init__.py
│   ├── config.py          # paths, class lists, cap, split ratios, seed
│   ├── device.py          # select_device(): CUDA → MPS → CPU (used in Plan 2; built here)
│   └── labels.py          # class normalization/merge, VOC parse, VOC→YOLO
├── data_prep/
│   ├── __init__.py
│   ├── collect.py         # pair images with XML across mem2–mem8
│   ├── grouping.py        # capture-sequence group keys (leakage units)
│   ├── split.py           # grouped, location-stratified 80/15/5
│   ├── balance.py         # greedy cap selection + inverse-freq weights
│   ├── profile_images.py  # exposure/clip/haze/sharpness stats
│   ├── preprocess.py      # EXIF orient + adaptive CLAHE; orig + clahe
│   ├── emit_yolo.py       # write YOLO DB
│   ├── emit_coco.py       # write COCO DB
│   ├── build_dataset.py   # CLI orchestrator + dataset version hash + manifest
│   └── verify_dataset.py  # assertions + spot-render
└── tests/
    ├── conftest.py
    ├── fixtures/           # tiny XML + synthetic images
    └── test_*.py
```

**Canonical class indices (fixed, used everywhere):**
- Pole sub-dataset: `["pole"]` → `pole=0`
- Components sub-dataset: `["wire", "h_insulator", "v_insulator", "crossarm_stright"]` → `wire=0, h_insulator=1, v_insulator=2, crossarm_stright=3`

---

### Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `shared/__init__.py`, `data_prep/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "train-dronisight"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "opencv-python>=4.9",
    "numpy>=1.26",
    "pillow>=10.2",
    "scikit-learn>=1.4",
    "pandas>=2.2",
    "pyyaml>=6.0",
    "lxml>=5.1",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the venv and install (uv, per project rule)**

Run:
```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
```
Expected: venv created, packages installed, `train-dronisight` editable install succeeds.

- [ ] **Step 3: Create empty package markers**

`shared/__init__.py`, `data_prep/__init__.py`, `tests/__init__.py` → empty files.

- [ ] **Step 4: Create `tests/conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 5: Verify pytest runs (no tests yet)**

Run: `pytest -q`
Expected: `no tests ran` (exit 0 or 5), no import errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml shared/ data_prep/ tests/
git commit -m "chore: scaffold data-prep project"
```

---

### Task 1: `shared/config.py` — central constants

**Files:**
- Create: `shared/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from shared import config

def test_class_sets_are_canonical():
    assert config.POLE_CLASSES == ["pole"]
    assert config.COMPONENT_CLASSES == ["wire", "h_insulator", "v_insulator", "crossarm_stright"]

def test_split_ratios_sum_to_one():
    assert abs(sum(config.SPLIT_RATIOS.values()) - 1.0) < 1e-9
    assert set(config.SPLIT_RATIOS) == {"train", "val", "test"}

def test_source_dirs_are_mem2_to_mem8():
    assert [p.name for p in config.SOURCE_DIRS] == [f"mem{i}" for i in range(2, 9)]

def test_seed_is_fixed():
    assert isinstance(config.SEED, int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`.

- [ ] **Step 3: Write `shared/config.py`**

```python
from pathlib import Path

SEED = 1337

# Source data (never modified)
SSD_ROOT = Path("/Volumes/dronisight")
SOURCE_DIRS = [SSD_ROOT / f"mem{i}" for i in range(2, 9)]  # mem1 has no labels

# Output DBs
YOLO_DB = SSD_ROOT / "yolo_train_db"
COCO_DB = SSD_ROOT / "RF_DETR_Faster_RCNN_train_db"

# Class policy (>1000-instance classes only; others ignored, never deleted)
POLE_CLASSES = ["pole"]
COMPONENT_CLASSES = ["wire", "h_insulator", "v_insulator", "crossarm_stright"]

# Split
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}

# Grouping: new capture-sequence group when consecutive frames differ by > this (seconds)
GROUP_TIME_GAP_S = 60

# Balancing
BALANCE_CAP_ENABLED = True  # cap to lowest kept-class count per sub-dataset

# CLAHE defaults (per-image params still come from the profile)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add shared/config.py tests/test_config.py
git commit -m "feat: add shared config constants"
```

---

### Task 2: `shared/device.py` — CUDA → MPS → CPU selector

**Files:**
- Create: `shared/device.py`
- Test: `tests/test_device.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_device.py
from unittest import mock
from shared import device

def test_prefers_cuda():
    with mock.patch.object(device, "_cuda_available", return_value=True), \
         mock.patch.object(device, "_mps_available", return_value=True):
        assert device.select_device() == "cuda"

def test_falls_back_to_mps():
    with mock.patch.object(device, "_cuda_available", return_value=False), \
         mock.patch.object(device, "_mps_available", return_value=True):
        assert device.select_device() == "mps"

def test_falls_back_to_cpu():
    with mock.patch.object(device, "_cuda_available", return_value=False), \
         mock.patch.object(device, "_mps_available", return_value=False):
        assert device.select_device() == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `shared/device.py`**

```python
"""Device selection with priority CUDA -> MPS -> CPU.

torch is imported lazily so data_prep (Plan 1) has no torch dependency.
"""


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def select_device() -> str:
    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_device.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add shared/device.py tests/test_device.py
git commit -m "feat: add CUDA->MPS->CPU device selector"
```

---

### Task 3: `shared/labels.py` — class normalization & merge

**Files:**
- Create: `shared/labels.py`
- Test: `tests/test_labels_normalize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labels_normalize.py
from shared.labels import normalize_class_name

def test_merges_all_crossarm_variants():
    for raw in ["crossarm_stright", "crossarm_Stright", "crossarmStright"]:
        assert normalize_class_name(raw) == "crossarm_stright"

def test_passes_through_kept_classes():
    for raw in ["pole", "wire", "h_insulator", "v_insulator"]:
        assert normalize_class_name(raw) == raw

def test_is_case_insensitive():
    assert normalize_class_name("H_Insulator") == "h_insulator"
    assert normalize_class_name(" WIRE ") == "wire"

def test_excluded_classes_return_none():
    for raw in ["rust", "om_crossarm", "top_crossarm", "vegetation"]:
        assert normalize_class_name(raw) is None

def test_unknown_returns_none():
    assert normalize_class_name("banana") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_labels_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the normalizer in `shared/labels.py`**

```python
"""Label source-of-truth = name-based VOC XML. Index-based YOLO .txt files are ignored."""

# All raw spellings seen in the data, mapped to a canonical kept class.
# Excluded classes (rust, om_crossarm, top_crossarm, vegetation) are intentionally
# NOT listed -> they normalize to None (ignored, never deleted from source).
_CANONICAL = {
    "pole": "pole",
    "wire": "wire",
    "h_insulator": "h_insulator",
    "v_insulator": "v_insulator",
    "crossarm_stright": "crossarm_stright",
    "crossarm_stright ": "crossarm_stright",
    "crossarmstright": "crossarm_stright",  # the mem5 stray 10th class
}


def normalize_class_name(raw: str):
    """Return canonical kept-class name, or None if the class is excluded/unknown."""
    if raw is None:
        return None
    key = raw.strip().lower()
    return _CANONICAL.get(key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_labels_normalize.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add shared/labels.py tests/test_labels_normalize.py
git commit -m "feat: add class name normalization and crossarm merge"
```

---

### Task 4: `shared/labels.py` — VOC XML parsing

**Files:**
- Modify: `shared/labels.py`
- Test: `tests/test_labels_parse.py`, `tests/fixtures/sample.xml`

- [ ] **Step 1: Create the fixture `tests/fixtures/sample.xml`**

```xml
<annotation>
  <size><width>4096</width><height>3072</height><depth>3</depth></size>
  <object><name>pole</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>220</ymax></bndbox></object>
  <object><name>crossarmStright</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>50</xmax><ymax>15</ymax></bndbox></object>
  <object><name>rust</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>
</annotation>
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_labels_parse.py
from pathlib import Path
from shared.labels import parse_voc

FIX = Path(__file__).parent / "fixtures" / "sample.xml"

def test_parses_size():
    ann = parse_voc(FIX)
    assert (ann.width, ann.height) == (4096, 3072)

def test_keeps_only_kept_classes_and_normalizes():
    ann = parse_voc(FIX)
    names = [b.name for b in ann.boxes]
    assert names == ["pole", "crossarm_stright"]  # rust excluded

def test_box_coords_are_ints():
    ann = parse_voc(FIX)
    pole = ann.boxes[0]
    assert (pole.xmin, pole.ymin, pole.xmax, pole.ymax) == (10, 20, 110, 220)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_labels_parse.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_voc'`.

- [ ] **Step 4: Add parsing to `shared/labels.py`**

```python
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass
class Box:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class Annotation:
    width: int
    height: int
    boxes: list  # list[Box]


def parse_voc(path) -> Annotation:
    """Parse a VOC XML, normalizing names and dropping excluded/invalid boxes."""
    root = ET.parse(str(path)).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        name = normalize_class_name(obj.findtext("name"))
        if name is None:
            continue
        bb = obj.find("bndbox")
        xmin = int(float(bb.findtext("xmin")))
        ymin = int(float(bb.findtext("ymin")))
        xmax = int(float(bb.findtext("xmax")))
        ymax = int(float(bb.findtext("ymax")))
        if xmax <= xmin or ymax <= ymin:
            continue  # drop degenerate boxes
        boxes.append(Box(name, xmin, ymin, xmax, ymax))
    return Annotation(width, height, boxes)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_labels_parse.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add shared/labels.py tests/test_labels_parse.py tests/fixtures/sample.xml
git commit -m "feat: parse VOC XML into normalized annotations"
```

---

### Task 5: `shared/labels.py` — VOC→YOLO conversion

**Files:**
- Modify: `shared/labels.py`
- Test: `tests/test_labels_yolo.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labels_yolo.py
import pytest
from shared.labels import Box, to_yolo_line

def test_converts_center_normalized():
    # box 10..110 x, 20..220 y on 100x100? use explicit dims
    b = Box("wire", xmin=0, ymin=0, xmax=50, ymax=100)
    line = to_yolo_line(b, img_w=100, img_h=200, class_names=["wire"])
    cls, xc, yc, w, h = line.split()
    assert cls == "0"
    assert float(xc) == pytest.approx(0.25)   # (0+50)/2 / 100
    assert float(yc) == pytest.approx(0.25)   # (0+100)/2 / 200
    assert float(w) == pytest.approx(0.5)
    assert float(h) == pytest.approx(0.5)

def test_uses_class_index_from_list():
    b = Box("crossarm_stright", 0, 0, 10, 10)
    names = ["wire", "h_insulator", "v_insulator", "crossarm_stright"]
    assert to_yolo_line(b, 100, 100, names).startswith("3 ")

def test_raises_if_class_not_in_list():
    b = Box("pole", 0, 0, 10, 10)
    with pytest.raises(ValueError):
        to_yolo_line(b, 100, 100, ["wire"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_labels_yolo.py -v`
Expected: FAIL with `ImportError: cannot import name 'to_yolo_line'`.

- [ ] **Step 3: Add conversion to `shared/labels.py`**

```python
def to_yolo_line(box: Box, img_w: int, img_h: int, class_names: list) -> str:
    """Format one box as a YOLO label line: '<cls> <xc> <yc> <w> <h>' (normalized)."""
    if box.name not in class_names:
        raise ValueError(f"{box.name!r} not in {class_names}")
    cls = class_names.index(box.name)
    xc = ((box.xmin + box.xmax) / 2) / img_w
    yc = ((box.ymin + box.ymax) / 2) / img_h
    w = (box.xmax - box.xmin) / img_w
    h = (box.ymax - box.ymin) / img_h
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_labels_yolo.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add shared/labels.py tests/test_labels_yolo.py
git commit -m "feat: convert VOC boxes to YOLO label lines"
```

---

### Task 6: `data_prep/collect.py` — pair images with XML

**Files:**
- Create: `data_prep/collect.py`
- Test: `tests/test_collect.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect.py
from pathlib import Path
from data_prep.collect import collect_samples

def test_pairs_only_images_with_xml(tmp_path):
    d = tmp_path / "mem2"
    d.mkdir()
    (d / "a.JPG").write_bytes(b"x")
    (d / "a.xml").write_text("<annotation/>")
    (d / "b.JPG").write_bytes(b"x")          # no xml -> skipped
    (d / "c.xml").write_text("<annotation/>")  # no image -> skipped
    samples = collect_samples([d])
    assert [s.image.name for s in samples] == ["a.JPG"]
    assert samples[0].xml.name == "a.xml"
    assert samples[0].source == "mem2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collect.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data_prep/collect.py`**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    image: Path
    xml: Path
    source: str  # mem folder name


def collect_samples(source_dirs) -> list:
    """Find every image that has a sibling .xml across the given source dirs."""
    samples = []
    for d in source_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for img in sorted(d.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            xml = img.with_suffix(".xml")
            if xml.exists():
                samples.append(Sample(image=img, xml=xml, source=d.name))
    return samples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collect.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add data_prep/collect.py tests/test_collect.py
git commit -m "feat: collect image/XML sample pairs from source dirs"
```

---

### Task 7: `data_prep/grouping.py` — capture-sequence groups (leakage units)

**Files:**
- Create: `data_prep/grouping.py`
- Test: `tests/test_grouping.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grouping.py
from datetime import datetime
from data_prep.grouping import parse_capture_time, assign_groups

def test_parse_dji_timestamp():
    dt = parse_capture_time("DJI_20260325112518_0159_D.JPG")
    assert dt == datetime(2026, 3, 25, 11, 25, 18)

def test_parse_returns_none_for_nonmatching():
    assert parse_capture_time("random.JPG") is None

def test_groups_split_on_time_gap():
    # two frames 5s apart, then a 10-min gap, then one more -> 2 groups
    names = [
        "DJI_20260325112518_0001_D.JPG",
        "DJI_20260325112523_0002_D.JPG",
        "DJI_20260325113523_0003_D.JPG",
    ]
    groups = assign_groups(names, source="mem2", gap_seconds=60)
    assert groups[names[0]] == groups[names[1]]
    assert groups[names[1]] != groups[names[2]]

def test_group_ids_are_namespaced_by_source():
    g = assign_groups(["DJI_20260325112518_0001_D.JPG"], source="mem2", gap_seconds=60)
    assert g["DJI_20260325112518_0001_D.JPG"].startswith("mem2:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grouping.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data_prep/grouping.py`**

```python
import re
from datetime import datetime

_TS = re.compile(r"DJI_(\d{14})_")


def parse_capture_time(filename: str):
    """Extract the DJI capture timestamp (YYYYMMDDHHMMSS) from a filename."""
    m = _TS.search(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")


def assign_groups(filenames, source: str, gap_seconds: int) -> dict:
    """Group consecutive frames; a gap > gap_seconds starts a new group.

    Files with no parseable timestamp each become their own group (conservative:
    never merged, so they can't leak across splits).
    """
    timed = []
    untimed = []
    for fn in filenames:
        t = parse_capture_time(fn)
        (timed if t else untimed).append((fn, t))
    timed.sort(key=lambda x: x[1])

    groups = {}
    gid = 0
    prev = None
    for fn, t in timed:
        if prev is not None and (t - prev).total_seconds() > gap_seconds:
            gid += 1
        groups[fn] = f"{source}:{gid}"
        prev = t
    for fn, _ in untimed:
        gid += 1
        groups[fn] = f"{source}:{gid}"
    return groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_grouping.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add data_prep/grouping.py tests/test_grouping.py
git commit -m "feat: assign capture-sequence groups for leakage-safe splitting"
```

---

### Task 8: `data_prep/split.py` — grouped, location-stratified 80/15/5

**Files:**
- Create: `data_prep/split.py`
- Test: `tests/test_split.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split.py
from data_prep.split import grouped_split

def _items():
    # 10 groups per source, 2 sources, 1 item each for simplicity
    items = []
    for src in ["mem2", "mem3"]:
        for g in range(10):
            items.append({"name": f"{src}_{g}", "group": f"{src}:{g}", "source": src})
    return items

def test_no_group_spans_two_splits():
    split = grouped_split(_items(), ratios={"train": .8, "val": .15, "test": .05}, seed=1)
    group_to_split = {}
    for s in ("train", "val", "test"):
        for it in split[s]:
            group_to_split.setdefault(it["group"], s)
            assert group_to_split[it["group"]] == s

def test_is_deterministic():
    a = grouped_split(_items(), {"train": .8, "val": .15, "test": .05}, seed=7)
    b = grouped_split(_items(), {"train": .8, "val": .15, "test": .05}, seed=7)
    assert [i["name"] for i in a["train"]] == [i["name"] for i in b["train"]]

def test_every_source_appears_in_train():
    split = grouped_split(_items(), {"train": .8, "val": .15, "test": .05}, seed=1)
    srcs = {it["source"] for it in split["train"]}
    assert srcs == {"mem2", "mem3"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data_prep/split.py`**

```python
import random
from collections import defaultdict


def grouped_split(items, ratios, seed):
    """Split items into train/val/test by GROUP (never splitting a group),
    stratified per source so every location appears in train.

    items: list of dicts with keys 'group' and 'source'.
    """
    rng = random.Random(seed)
    # group -> its source (groups are source-namespaced, so unique per source)
    groups_by_source = defaultdict(list)
    members = defaultdict(list)
    for it in items:
        members[it["group"]].append(it)
    for g, its in members.items():
        groups_by_source[its[0]["source"]].append(g)

    out = {"train": [], "val": [], "test": []}
    for source, groups in groups_by_source.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        n = len(groups)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        # guarantee >=1 train group per source when possible
        n_train = max(n_train, 1) if n else 0
        buckets = {
            "train": groups[:n_train],
            "val": groups[n_train:n_train + n_val],
            "test": groups[n_train + n_val:],
        }
        for split_name, gs in buckets.items():
            for g in gs:
                out[split_name].extend(members[g])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_split.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add data_prep/split.py tests/test_split.py
git commit -m "feat: grouped location-stratified train/val/test split"
```

---

### Task 9: `data_prep/balance.py` — greedy cap + inverse-freq weights

**Files:**
- Create: `data_prep/balance.py`
- Test: `tests/test_balance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_balance.py
from data_prep.balance import cap_target, select_balanced, sample_weights

def test_cap_target_is_min_kept_count():
    counts = {"wire": 3500, "h_insulator": 3000, "crossarm_stright": 1661}
    assert cap_target(counts) == 1661

def test_select_keeps_all_when_disabled():
    items = [{"name": "a", "classes": ["wire"]}, {"name": "b", "classes": ["wire"]}]
    kept = select_balanced(items, class_names=["wire"], enabled=False, seed=1)
    assert len(kept) == 2

def test_select_respects_cap_for_overrepresented_class():
    # 5 images each with one 'wire'; cap=2 -> keep 2
    items = [{"name": f"w{i}", "classes": ["wire"]} for i in range(5)]
    kept = select_balanced(items, class_names=["wire"], enabled=True, seed=1, cap=2)
    assert sum("wire" in it["classes"] for it in kept) == 2

def test_select_prioritizes_rare_class_images():
    items = [{"name": "rareimg", "classes": ["crossarm_stright"]}] + \
            [{"name": f"w{i}", "classes": ["wire"]} for i in range(5)]
    kept = select_balanced(items, class_names=["wire", "crossarm_stright"],
                           enabled=True, seed=1, cap=1)
    assert any(it["name"] == "rareimg" for it in kept)

def test_sample_weights_are_inverse_frequency():
    items = [{"classes": ["wire", "wire"]}, {"classes": ["crossarm_stright"]}]
    w = sample_weights(items, ["wire", "crossarm_stright"])
    assert w[1] > w[0]  # the rare-class image gets a higher weight
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_balance.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data_prep/balance.py`**

```python
import random
from collections import Counter


def cap_target(class_counts: dict) -> int:
    """Stable-frequency cap = the lowest kept-class instance count."""
    return min(class_counts.values())


def _counts(items, class_names):
    c = Counter()
    for it in items:
        for cls in it["classes"]:
            if cls in class_names:
                c[cls] += 1
    return {cls: c.get(cls, 0) for cls in class_names}


def select_balanced(items, class_names, enabled, seed, cap=None):
    """Greedily select images so each class's instance count approaches `cap`,
    admitting rare-class images first and never dropping an image while it still
    feeds an under-cap class. Returns the kept subset (order-independent)."""
    if not enabled:
        return list(items)
    if cap is None:
        cap = cap_target(_counts(items, class_names))

    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    # rarer images first: fewer total kept-class instances == more "specific"
    items.sort(key=lambda it: sum(c in class_names for c in it["classes"]))

    running = {cls: 0 for cls in class_names}
    kept = []
    for it in items:
        contributes = [c for c in it["classes"] if c in class_names and running[c] < cap]
        if not contributes:
            continue
        kept.append(it)
        for c in it["classes"]:
            if c in class_names:
                running[c] += 1
    return kept


def sample_weights(items, class_names) -> list:
    """Per-image inverse-frequency weight (for 'train on all data + weighted sampling')."""
    totals = _counts(items, class_names)
    inv = {cls: (1.0 / n if n else 0.0) for cls, n in totals.items()}
    weights = []
    for it in items:
        w = sum(inv.get(c, 0.0) for c in it["classes"] if c in class_names)
        weights.append(w)
    return weights
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_balance.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add data_prep/balance.py tests/test_balance.py
git commit -m "feat: greedy class-balanced selection + inverse-freq weights"
```

---

### Task 10: `data_prep/profile_images.py` — exposure/clip/haze/sharpness

**Files:**
- Create: `data_prep/profile_images.py`
- Test: `tests/test_profile.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profile.py
import numpy as np
from data_prep.profile_images import profile_array

def test_bright_image_has_high_highlight_clip():
    img = np.full((64, 64, 3), 255, np.uint8)
    p = profile_array(img)
    assert p["highlight_clip"] > 0.9
    assert p["mean_luma"] > 240

def test_dark_image_has_high_shadow_clip():
    img = np.zeros((64, 64, 3), np.uint8)
    p = profile_array(img)
    assert p["shadow_clip"] > 0.9

def test_backlit_image_flagged():
    img = np.zeros((64, 64, 3), np.uint8)
    img[:20, :, :] = 255  # bright sky band, dark below
    p = profile_array(img)
    assert p["highlight_clip"] > 0.2
    assert p["backlit"] is True

def test_profile_keys_present():
    img = np.full((32, 32, 3), 128, np.uint8)
    p = profile_array(img)
    for k in ("mean_luma", "std_luma", "highlight_clip", "shadow_clip", "haze", "sharpness", "backlit"):
        assert k in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_profile.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data_prep/profile_images.py`**

```python
import cv2
import numpy as np


def profile_array(bgr: np.ndarray) -> dict:
    """Compute exposure/clip/haze/sharpness statistics for one BGR image."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    n = L.size

    highlight_clip = float((L >= 250).sum() / n)
    shadow_clip = float((L <= 5).sum() / n)
    mean_luma = float(L.mean())
    std_luma = float(L.std())

    # dark-channel prior as a coarse haze proxy (higher == hazier)
    dark = cv2.erode(bgr.min(axis=2), np.ones((15, 15), np.uint8))
    haze = float(dark.mean() / 255.0)

    sharpness = float(cv2.Laplacian(L, cv2.CV_32F).var())

    # backlit: meaningful blown highlights AND a dark region present
    backlit = bool(highlight_clip > 0.15 and (L < 60).mean() > 0.15)

    return {
        "mean_luma": mean_luma,
        "std_luma": std_luma,
        "highlight_clip": highlight_clip,
        "shadow_clip": shadow_clip,
        "haze": haze,
        "sharpness": sharpness,
        "backlit": backlit,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_profile.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add data_prep/profile_images.py tests/test_profile.py
git commit -m "feat: per-image exposure/haze/sharpness profiling"
```

---

### Task 11: `data_prep/preprocess.py` — EXIF orient + adaptive CLAHE

**Files:**
- Create: `data_prep/preprocess.py`
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocess.py
import numpy as np
from data_prep.preprocess import clahe_params_from_profile, apply_clahe

def test_well_exposed_image_gets_near_identity_clip():
    profile = {"backlit": False, "highlight_clip": 0.01, "shadow_clip": 0.01}
    clip, grid = clahe_params_from_profile(profile)
    assert clip <= 1.2  # near identity

def test_backlit_image_gets_stronger_clip():
    profile = {"backlit": True, "highlight_clip": 0.4, "shadow_clip": 0.3}
    clip, grid = clahe_params_from_profile(profile)
    assert clip >= 2.0

def test_apply_clahe_preserves_shape_and_dtype():
    img = np.random.randint(0, 255, (48, 64, 3), np.uint8)
    out = apply_clahe(img, clip=2.0, grid=(8, 8))
    assert out.shape == img.shape and out.dtype == np.uint8

def test_apply_clahe_increases_contrast_on_low_contrast_input():
    img = np.full((48, 64, 3), 100, np.uint8)
    img[:24] = 110  # very low contrast
    out = apply_clahe(img, clip=3.0, grid=(8, 8))
    assert out.std() >= img.std()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data_prep/preprocess.py`**

```python
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_oriented_bgr(path) -> np.ndarray:
    """Load an image honoring EXIF orientation, return BGR uint8."""
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.asarray(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def clahe_params_from_profile(profile: dict):
    """Pick (clipLimit, tileGrid) adaptively: near-identity unless backlit/clipped."""
    grid = (8, 8)
    if profile.get("backlit") or profile.get("highlight_clip", 0) > 0.2:
        clip = 3.0 if profile.get("highlight_clip", 0) > 0.35 else 2.0
    elif profile.get("shadow_clip", 0) > 0.2:
        clip = 2.0
    else:
        clip = 1.0  # effectively identity
    return clip, grid


def apply_clahe(bgr: np.ndarray, clip: float, grid) -> np.ndarray:
    """CLAHE on the LAB L-channel only; chroma untouched."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocess.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add data_prep/preprocess.py tests/test_preprocess.py
git commit -m "feat: EXIF-aware load + adaptive LAB-L CLAHE"
```

---

### Task 12: `data_prep/emit_yolo.py` — write the YOLO DB

**Files:**
- Create: `data_prep/emit_yolo.py`
- Test: `tests/test_emit_yolo.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emit_yolo.py
import yaml
from pathlib import Path
from shared.labels import Box
from data_prep.emit_yolo import write_label_file, write_data_yaml

def test_write_label_file(tmp_path):
    boxes = [Box("wire", 0, 0, 50, 100), Box("crossarm_stright", 10, 10, 20, 20)]
    out = tmp_path / "img.txt"
    write_label_file(out, boxes, 100, 200, ["wire", "h_insulator", "v_insulator", "crossarm_stright"])
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("3 ")

def test_write_data_yaml(tmp_path):
    p = write_data_yaml(tmp_path, version="clahe", class_names=["wire", "h_insulator"])
    data = yaml.safe_load(Path(p).read_text())
    assert data["names"] == {0: "wire", 1: "h_insulator"}
    assert data["train"].endswith("images/train/clahe")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_emit_yolo.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data_prep/emit_yolo.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_emit_yolo.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add data_prep/emit_yolo.py tests/test_emit_yolo.py
git commit -m "feat: emit YOLO labels and data.yaml"
```

---

### Task 13: `data_prep/emit_coco.py` — write the COCO DB

**Files:**
- Create: `data_prep/emit_coco.py`
- Test: `tests/test_emit_coco.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emit_coco.py
import json
from shared.labels import Box, Annotation
from data_prep.emit_coco import build_coco

def test_build_coco_structure():
    anns = {
        "a.jpg": Annotation(100, 200, [Box("wire", 0, 0, 50, 100)]),
        "b.jpg": Annotation(100, 100, [Box("crossarm_stright", 10, 10, 30, 40)]),
    }
    coco = build_coco(anns, class_names=["wire", "h_insulator", "v_insulator", "crossarm_stright"])
    assert {c["name"] for c in coco["categories"]} == \
        {"wire", "h_insulator", "v_insulator", "crossarm_stright"}
    assert len(coco["images"]) == 2
    # COCO bbox is [x, y, w, h]
    wire_ann = [a for a in coco["annotations"] if a["category_id"] == 0][0]
    assert wire_ann["bbox"] == [0, 0, 50, 100]

def test_category_ids_match_class_index():
    coco = build_coco({}, ["wire", "h_insulator"])
    cats = {c["name"]: c["id"] for c in coco["categories"]}
    assert cats == {"wire": 0, "h_insulator": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_emit_coco.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data_prep/emit_coco.py`**

```python
import json
from pathlib import Path


def build_coco(annotations: dict, class_names: list) -> dict:
    """Build a COCO dict. annotations: {image_filename: Annotation}."""
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    images, anns = [], []
    ann_id = 1
    for img_id, (fname, ann) in enumerate(sorted(annotations.items()), start=1):
        images.append({"id": img_id, "file_name": fname,
                       "width": ann.width, "height": ann.height})
        for b in ann.boxes:
            if b.name not in class_names:
                continue
            anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_names.index(b.name),
                "bbox": [b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin],
                "area": (b.xmax - b.xmin) * (b.ymax - b.ymin),
                "iscrowd": 0,
            })
            ann_id += 1
    return {"images": images, "annotations": anns, "categories": categories}


def write_coco(path, coco: dict):
    Path(path).write_text(json.dumps(coco))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_emit_coco.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add data_prep/emit_coco.py tests/test_emit_coco.py
git commit -m "feat: emit COCO annotations for FRCNN/RF-DETR"
```

---

### Task 14: `data_prep/build_dataset.py` — CLI orchestrator

**Files:**
- Create: `data_prep/build_dataset.py`
- Test: `tests/test_build_dataset.py`

- [ ] **Step 1: Write the failing test (orchestration logic, not full I/O)**

```python
# tests/test_build_dataset.py
from data_prep.build_dataset import sample_class_list, dataset_version_hash

def test_sample_class_list_pole_vs_components():
    from shared import config
    assert sample_class_list("pole") == config.POLE_CLASSES
    assert sample_class_list("components") == config.COMPONENT_CLASSES

def test_version_hash_is_stable_and_order_independent():
    a = dataset_version_hash(["mem2/a.JPG", "mem3/b.JPG"])
    b = dataset_version_hash(["mem3/b.JPG", "mem2/a.JPG"])
    assert a == b and len(a) == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data_prep/build_dataset.py`**

```python
"""Orchestrate the full data-prep into both DBs.

Usage:
    python -m data_prep.build_dataset --subset pole
    python -m data_prep.build_dataset --subset components
    python -m data_prep.build_dataset --subset all [--no-balance]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import pandas as pd

from shared import config
from shared.labels import parse_voc
from data_prep.collect import collect_samples
from data_prep.grouping import assign_groups
from data_prep.split import grouped_split
from data_prep.balance import select_balanced, sample_weights, cap_target
from data_prep.profile_images import profile_array
from data_prep.preprocess import load_oriented_bgr, clahe_params_from_profile, apply_clahe
from data_prep.emit_yolo import write_label_file, write_data_yaml
from data_prep.emit_coco import build_coco, write_coco


def sample_class_list(subset: str):
    return config.POLE_CLASSES if subset == "pole" else config.COMPONENT_CLASSES


def dataset_version_hash(image_keys) -> str:
    h = hashlib.sha256()
    for k in sorted(image_keys):
        h.update(k.encode())
    return h.hexdigest()[:12]


def _save(bgr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def build_subset(subset: str, balance: bool):
    class_names = sample_class_list(subset)
    samples = collect_samples(config.SOURCE_DIRS)

    # parse + keep only images that contain >=1 class for this subset
    parsed = {}
    for s in samples:
        ann = parse_voc(s.xml)
        if any(b.name in class_names for b in ann.boxes):
            parsed[s.image] = (s, ann)

    # groups (per source) -> items
    by_source = {}
    for img in parsed:
        by_source.setdefault(parsed[img][0].source, []).append(img.name)
    name_to_group = {}
    for source, names in by_source.items():
        name_to_group.update(assign_groups(names, source, config.GROUP_TIME_GAP_S))

    items = [{"image": img, "name": img.name, "group": name_to_group[img.name],
              "source": parsed[img][0].source,
              "classes": [b.name for b in parsed[img][1].boxes if b.name in class_names]}
             for img in parsed]

    if balance and config.BALANCE_CAP_ENABLED:
        items = select_balanced(items, class_names, enabled=True, seed=config.SEED)

    split = grouped_split(items, config.SPLIT_RATIOS, config.SEED)
    version_hash = dataset_version_hash([it["name"] for it in items])

    # write both DBs
    coco_per_split = {"train": {}, "val": {}, "test": {}}
    manifest_rows = []
    for split_name, split_items in split.items():
        for it in split_items:
            s, ann = parsed[it["image"]]
            bgr = load_oriented_bgr(s.image)
            prof = profile_array(bgr)
            clip, grid = clahe_params_from_profile(prof)
            clahe_img = apply_clahe(bgr, clip, grid)

            stem = it["image"].stem
            for db_root in (config.YOLO_DB, config.COCO_DB):
                base = db_root / subset / "images" / split_name
                _save(bgr, base / "orig" / f"{stem}.jpg")
                _save(clahe_img, base / "clahe" / f"{stem}.jpg")

            # YOLO labels (shared by orig/clahe)
            lbl = config.YOLO_DB / subset / "labels" / split_name / f"{stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            write_label_file(lbl, ann.boxes, ann.width, ann.height, class_names)

            coco_per_split[split_name][f"{stem}.jpg"] = ann
            manifest_rows.append({"name": it["name"], "source": it["source"],
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
    pd.DataFrame({"name": [it["name"] for it in items], "weight": weights}) \
        .to_csv(config.YOLO_DB / subset / "sample_weights.csv", index=False)
    meta = {"subset": subset, "version_hash": version_hash,
            "n_images": len(items), "balanced": bool(balance and config.BALANCE_CAP_ENABLED),
            "class_names": class_names}
    for db in (config.YOLO_DB, config.COCO_DB):
        (db / subset).mkdir(parents=True, exist_ok=True)
        (db / subset / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{subset}] {len(items)} images, version {version_hash}")


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_dataset.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add data_prep/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: orchestrate full data-prep into both DBs"
```

---

### Task 15: `data_prep/verify_dataset.py` — integrity assertions + spot-render

**Files:**
- Create: `data_prep/verify_dataset.py`
- Test: `tests/test_verify.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify.py
import pytest
from data_prep.verify_dataset import assert_no_group_leakage, class_counts_from_manifest
import pandas as pd

def test_leakage_detector_passes_clean_split():
    df = pd.DataFrame([
        {"group": "mem2:0", "split": "train"},
        {"group": "mem2:0", "split": "train"},
        {"group": "mem2:1", "split": "val"},
    ])
    assert_no_group_leakage(df)  # no raise

def test_leakage_detector_raises_on_span():
    df = pd.DataFrame([
        {"group": "mem2:0", "split": "train"},
        {"group": "mem2:0", "split": "val"},
    ])
    with pytest.raises(AssertionError):
        assert_no_group_leakage(df)

def test_class_counts():
    df = pd.DataFrame([{"split": "train"}, {"split": "val"}])
    counts = class_counts_from_manifest(df)
    assert counts == {"train": 1, "val": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_verify.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `data_prep/verify_dataset.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_verify.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add data_prep/verify_dataset.py tests/test_verify.py
git commit -m "feat: dataset integrity verification (leakage + label validity)"
```

---

### Task 16: End-to-end smoke run on real data

**Files:** none (manual validation gate)

- [ ] **Step 1: Run full test suite**

Run: `pytest -q`
Expected: all tests PASS.

- [ ] **Step 2: Build the pole subset for real**

Run: `python -m data_prep.build_dataset --subset pole`
Expected: prints `[pole] <N> images, version <hash>`; `/Volumes/dronisight/yolo_train_db/pole/` and `RF_DETR_Faster_RCNN_train_db/pole/` populated.

- [ ] **Step 3: Build the components subset for real**

Run: `python -m data_prep.build_dataset --subset components`
Expected: prints `[components] <N> images, version <hash>`.

- [ ] **Step 4: Verify both subsets**

Run:
```bash
python -m data_prep.verify_dataset --subset pole
python -m data_prep.verify_dataset --subset components
```
Expected: `OK no leakage` + `OK all YOLO labels valid` for both.

- [ ] **Step 5: Sanity-check counts**

Run: `python -c "import pandas as pd; d=pd.read_csv('/Volumes/dronisight/yolo_train_db/components/manifest.csv'); print(d['split'].value_counts()); print(d['backlit'].mean())"`
Expected: ~80/15/5 split; a nonzero backlit fraction (confirms profiling works).

- [ ] **Step 6: Commit any fixups**

```bash
git add -A && git commit -m "test: end-to-end data-prep smoke run validated"
```

---

## Self-Review Notes (completed)
- **Spec coverage:** §2 label cleanup → Tasks 3–5; §2.3 class policy/cap → Tasks 1,9; §3 preprocessing → Tasks 10–11; §4 DB layout/split → Tasks 8,12,13,14; §5 module structure → all; §6.5 leakage assertion → Task 15. Device selector (§1) built in Task 2 for Plan 2. Training/inference/notebooks are Plans 2–3 (out of scope here).
- **Placeholders:** none — every code step is complete.
- **Type consistency:** `Box`/`Annotation` (Task 4) reused identically in Tasks 5, 12, 13; `Sample` (Task 6) used in Task 14; class-name lists flow from `config` (Task 1) everywhere.
