# Colab Notebooks Implementation Plan (Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide Google Colab notebooks (CUDA) that reproduce the whole workflow — data-prep, all three trainings (YOLO26x / Faster R-CNN / RF-DETR-L), and the inference pipeline — wrapping the modules from Plans 1–2 so nothing is duplicated.

**Architecture:** Notebooks are **generated programmatically** with `nbformat` from `notebooks/build_notebooks.py`, so they're reproducible and lint-checkable rather than hand-edited JSON. Each notebook is thin: mount Drive, clone/pull the repo, `uv pip install`, then call the existing `data_prep` / `train_*` / `inference` entrypoints. A small, unit-tested `notebooks/colab_utils.py` holds the only real logic (Drive paths, dataset unzip, repo setup).

**Tech Stack:** Adds `nbformat`. Reuses everything from Plans 1–2.

**Depends on:** Plans 1 & 2.

**Why CUDA here:** RF-DETR-L training is impractical on MPS (spec §1/§6.6); Colab's free/cheap NVIDIA GPU is the intended home for it, and the same notebooks accelerate YOLO/Faster R-CNN too. The `CUDA→MPS→CPU` selector means the identical code runs locally and on Colab.

---

## File Structure

```
trainDronisight/
├── notebooks/
│   ├── __init__.py
│   ├── colab_utils.py        # Drive mount paths, zip detection, repo setup (unit-tested)
│   ├── build_notebooks.py    # generates the 5 .ipynb via nbformat
│   ├── 00_data_prep.ipynb        (generated)
│   ├── 01_train_yolo.ipynb       (generated)
│   ├── 02_train_faster_rcnn.ipynb(generated)
│   ├── 03_train_rf_detr.ipynb    (generated)
│   └── 04_inference_pipeline.ipynb(generated)
└── tests/
    └── test_colab_utils.py, test_build_notebooks.py
```

**Data-transfer model (documented in 00):** the two DBs are large (full 12 MP). On the M4 you copy the folder directly; on Colab you **zip each DB to Google Drive once**, then the notebook unzips to fast local `/content` storage. `colab_utils.ensure_dataset()` encapsulates this.

---

### Task 0: Add `nbformat` dep + package marker

**Files:**
- Modify: `pyproject.toml`
- Create: `notebooks/__init__.py`

- [ ] **Step 1: Add `nbformat>=5.10` to `pyproject.toml` dependencies**

```toml
    "nbformat>=5.10",
```

- [ ] **Step 2: Install**

Run: `source .venv/bin/activate && uv pip install -e ".[dev]"`
Expected: `nbformat` installed.

- [ ] **Step 3: Create empty `notebooks/__init__.py`**

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml notebooks/__init__.py
git commit -m "chore: add nbformat for notebook generation"
```

---

### Task 1: `notebooks/colab_utils.py` — Drive/dataset/repo helpers

**Files:**
- Create: `notebooks/colab_utils.py`
- Test: `tests/test_colab_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_colab_utils.py
from pathlib import Path
from unittest import mock
from notebooks import colab_utils

def test_drive_db_path():
    p = colab_utils.drive_db_zip("yolo_train_db", drive_root="/content/drive/MyDrive/dronisight")
    assert p == "/content/drive/MyDrive/dronisight/yolo_train_db.zip"

def test_ensure_dataset_unzips_when_missing(tmp_path):
    zip_path = tmp_path / "yolo_train_db.zip"
    zip_path.write_bytes(b"PK")  # pretend zip exists
    dest = tmp_path / "data"
    with mock.patch.object(colab_utils, "_unzip") as uz:
        out = colab_utils.ensure_dataset(str(zip_path), str(dest))
    uz.assert_called_once_with(str(zip_path), str(dest))
    assert out == str(dest)

def test_ensure_dataset_skips_when_present(tmp_path):
    dest = tmp_path / "data" / "yolo_train_db"
    dest.mkdir(parents=True)
    (dest / "marker").write_text("x")  # already unzipped
    with mock.patch.object(colab_utils, "_unzip") as uz:
        colab_utils.ensure_dataset(str(tmp_path / "yolo_train_db.zip"),
                                   str(tmp_path / "data"), expect_subdir="yolo_train_db")
    uz.assert_not_called()

def test_repo_clone_command():
    cmd = colab_utils.repo_setup_cmd("https://github.com/u/trainDronisight.git", "/content/repo")
    assert "git clone" in cmd and "/content/repo" in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_colab_utils.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `notebooks/colab_utils.py`**

```python
"""Helpers used inside the Colab notebooks. Pure/path logic is unit-tested;
the Colab-only side effects (drive.mount) live in tiny wrappers."""
import os
import zipfile
from pathlib import Path


def drive_db_zip(db_name: str, drive_root="/content/drive/MyDrive/dronisight") -> str:
    return f"{drive_root}/{db_name}.zip"


def _unzip(zip_path: str, dest: str):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def ensure_dataset(zip_path: str, dest: str, expect_subdir: str = None) -> str:
    """Unzip the DB to fast local storage if it isn't already there."""
    check = Path(dest) / expect_subdir if expect_subdir else Path(dest)
    if check.exists() and any(check.iterdir()):
        return dest
    Path(dest).mkdir(parents=True, exist_ok=True)
    _unzip(zip_path, dest)
    return dest


def repo_setup_cmd(repo_url: str, dest="/content/repo") -> str:
    return f"git clone {repo_url} {dest} || (cd {dest} && git pull)"


def mount_drive():  # pragma: no cover (Colab-only)
    from google.colab import drive
    drive.mount("/content/drive")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_colab_utils.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add notebooks/colab_utils.py tests/test_colab_utils.py
git commit -m "feat: Colab Drive/dataset/repo helpers"
```

---

### Task 2: `notebooks/build_notebooks.py` — generate the 5 notebooks

**Files:**
- Create: `notebooks/build_notebooks.py`
- Test: `tests/test_build_notebooks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_notebooks.py
import nbformat
from pathlib import Path
from notebooks.build_notebooks import build_all, NOTEBOOKS

def test_build_all_writes_valid_notebooks(tmp_path):
    paths = build_all(out_dir=tmp_path)
    assert len(paths) == len(NOTEBOOKS)
    for p in paths:
        nb = nbformat.read(p, as_version=4)   # raises if invalid
        nbformat.validate(nb)
        assert len(nb.cells) >= 3

def test_each_notebook_has_a_gpu_check_and_install():
    from notebooks.build_notebooks import NOTEBOOKS
    for spec in NOTEBOOKS.values():
        joined = "\n".join(spec)
        assert "nvidia-smi" in joined or "torch.cuda" in joined
        assert "uv pip install" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_notebooks.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `notebooks/build_notebooks.py`**

```python
"""Generate Colab notebooks from cell specs. Run: python -m notebooks.build_notebooks"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

REPO_URL = "https://github.com/REPLACE_ME/trainDronisight.git"  # set before publishing

_SETUP = [
    "# @title Setup: GPU check + repo + deps",
    "!nvidia-smi",
    f"from pathlib import Path\n"
    f"REPO='/content/repo'\n"
    f"!git clone {REPO_URL} $REPO 2>/dev/null || (cd $REPO && git pull)\n"
    "%cd $REPO\n"
    "!pip -q install uv && uv pip install --system -e .",
    "import torch; print('CUDA:', torch.cuda.is_available())",
    "from notebooks.colab_utils import mount_drive, drive_db_zip, ensure_dataset\nmount_drive()",
]

NOTEBOOKS = {
    "00_data_prep": _SETUP + [
        "# Unzip both DBs from Drive to fast local storage\n"
        "ensure_dataset(drive_db_zip('yolo_train_db'), '/content/data', 'yolo_train_db')\n"
        "ensure_dataset(drive_db_zip('RF_DETR_Faster_RCNN_train_db'), '/content/data', 'RF_DETR_Faster_RCNN_train_db')",
        "# (Optional) Re-run data-prep from raw mem* if they're on Drive instead of prebuilt DBs\n"
        "# !python -m data_prep.build_dataset --subset all",
        "!python -m data_prep.verify_dataset --subset pole\n"
        "!python -m data_prep.verify_dataset --subset components",
    ],
    "01_train_yolo": _SETUP + [
        "ensure_dataset(drive_db_zip('yolo_train_db'), '/content/data', 'yolo_train_db')",
        "!python -m train_yolo.train_pole --version clahe --epochs 100 --imgsz 1280 --batch 16",
        "!python -m train_yolo.train_components --version clahe --epochs 150 --imgsz 1280 --batch 16",
    ],
    "02_train_faster_rcnn": _SETUP + [
        "ensure_dataset(drive_db_zip('RF_DETR_Faster_RCNN_train_db'), '/content/data', 'RF_DETR_Faster_RCNN_train_db')",
        "!python -m train_faster_rcnn.train --subset pole --version clahe --epochs 30 --batch 4",
        "!python -m train_faster_rcnn.train --subset components --version clahe --epochs 30 --batch 4",
    ],
    "03_train_rf_detr": _SETUP + [
        "ensure_dataset(drive_db_zip('RF_DETR_Faster_RCNN_train_db'), '/content/data', 'RF_DETR_Faster_RCNN_train_db')",
        "!python -m train_rf_detr.train --subset pole --version clahe --epochs 50 --batch 4",
        "!python -m train_rf_detr.train --subset components --version clahe --epochs 50 --batch 4",
    ],
    "04_inference_pipeline": _SETUP + [
        "# point these at trained weights (on Drive or in runs/)\n"
        "import glob\n"
        "POLE=sorted(glob.glob('runs/pole/yolo*/weights/best.pt'))[-1]\n"
        "COMP=sorted(glob.glob('runs/components/yolo*/weights/best.pt'))[-1]\nprint(POLE, COMP)",
        "import glob\nIMG=sorted(glob.glob('/content/data/yolo_train_db/components/images/test/orig/*.jpg'))[0]\nprint(IMG)",
        "!python -m inference.pipeline --image \"$IMG\" --pole-weights $POLE --comp-weights $COMP --out /content/result.json",
        "import json; print(json.dumps(json.load(open('/content/result.json')), indent=2))",
    ],
}


def _to_notebook(title, cells):
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(f"# {title}\n\nGenerated by `notebooks/build_notebooks.py`. "
                                      "Runtime → Change runtime type → GPU, then Run all."))
    for c in cells:
        nb.cells.append(new_code_cell(c))
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": []}
    return nb


def build_all(out_dir=None):
    out_dir = Path(out_dir or Path(__file__).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, cells in NOTEBOOKS.items():
        nb = _to_notebook(name, cells)
        path = out_dir / f"{name}.ipynb"
        nbformat.write(nb, str(path))
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    for p in build_all():
        print("wrote", p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_notebooks.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Generate the notebooks and commit**

Run: `python -m notebooks.build_notebooks`
Expected: writes the 5 `.ipynb` files into `notebooks/`.

```bash
git add notebooks/build_notebooks.py tests/test_build_notebooks.py notebooks/*.ipynb
git commit -m "feat: generate Colab notebooks for data-prep, training, inference"
```

---

### Task 3: Set the repo URL + README usage

**Files:**
- Modify: `notebooks/build_notebooks.py` (set `REPO_URL`)
- Create: `notebooks/README.md`

- [ ] **Step 1: Set `REPO_URL`**

Replace `https://github.com/REPLACE_ME/trainDronisight.git` with the actual repo URL (or a placeholder the user fills before publishing). If the user has no remote yet, leave a clear `REPLACE_ME` and document it in the README.

- [ ] **Step 2: Write `notebooks/README.md`**

```markdown
# Colab Notebooks

Generated by `python -m notebooks.build_notebooks` (edit cell specs there, never the .ipynb by hand).

## One-time setup
1. Zip each DB and upload to Google Drive at `MyDrive/dronisight/`:
   - `yolo_train_db.zip`
   - `RF_DETR_Faster_RCNN_train_db.zip`
2. Set `REPO_URL` in `build_notebooks.py` to your Git remote and regenerate.

## Order
- `00_data_prep` — verify DBs (or rebuild from raw `mem*`).
- `01_train_yolo` — primary (YOLO26x, falls back to yolo11x if needed).
- `02_train_faster_rcnn`, `03_train_rf_detr` — comparison models (CUDA).
- `04_inference_pipeline` — run the two-stage chain, print structured JSON.

Each notebook: **Runtime → GPU → Run all.** The `CUDA→MPS→CPU` selector means the
same code runs locally on the M4 (MPS) and here (CUDA).
```

- [ ] **Step 3: Regenerate notebooks with the real URL and commit**

Run: `python -m notebooks.build_notebooks`

```bash
git add notebooks/build_notebooks.py notebooks/README.md notebooks/*.ipynb
git commit -m "docs: set repo URL and notebook usage README"
```

---

### Task 4: Colab validation (gate — run in a browser)

**Files:** none (manual validation on Colab)

- [ ] **Step 1: Full local suite passes**

Run: `pytest -q`
Expected: all PASS.

- [ ] **Step 2: Push the repo so Colab can clone it**

Run: `git push` (after setting a remote, if not already).
Expected: remote has the latest code + generated notebooks.

- [ ] **Step 3: Open `01_train_yolo.ipynb` in Colab**

- Upload/open via GitHub in Colab; set Runtime → GPU.
- Run all cells.
- Expected: `nvidia-smi` shows a GPU; `CUDA: True`; YOLO trains; weights written. (Use small `--epochs 1` first to validate plumbing fast.)

- [ ] **Step 4: Open `04_inference_pipeline.ipynb` and Run all**

Expected: structured JSON printed with `poles[]` → `components[]`.

- [ ] **Step 5: Note results**

If any cell fails (path, dep, Drive layout), fix the cell spec in `build_notebooks.py`, regenerate, recommit. Do **not** hand-edit `.ipynb`.

```bash
git add -A && git commit -m "test: Colab notebooks validated end-to-end"
```

---

## Self-Review Notes (completed)
- **Spec coverage:** §5 notebooks (data-prep, 3 trainings, inference) → Tasks 2–3 (the 5 notebooks); CUDA path (§1/§6.6) → GPU runtime + `select_device`; no duplicated logic — notebooks only call Plan 1–2 entrypoints.
- **Placeholders:** the only intentional placeholder is `REPO_URL` (`REPLACE_ME`), explicitly handled in Task 3 and documented — it's a user-supplied value, not an incomplete step.
- **Type consistency:** notebooks invoke the exact CLI module paths defined in Plans 1–2 (`data_prep.build_dataset`, `train_yolo.train_pole`, `inference.pipeline`, …); `colab_utils` functions match their tests.
