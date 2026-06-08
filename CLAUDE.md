# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Three-detector object-detection pipeline for **electric-pole inspection** from DJI drone imagery.
**Model 1 (`pole`)** detects the pole on the full frame → crop to the pole box (+pad) → then **two
component detectors run on the crop**:
- **`component_above_1000`** — the high-frequency classes `wire, h_insulator, v_insulator, crossarm_stright`.
- **`component_below_1000`** — the rare classes `vegetation, top_crossarm, om_crossarm, rust`.

Both component detectors' boxes are remapped to the full frame and each component is re-cropped →
structured JSON (`poles[]` each with `components_above[]` and `components_below[]`). The three models
are **separate trainings** producing separate weights, chained only at inference (`inference/pipeline.py`).

Detection-only (v1). Condition classifier, scoring, and the OpenStreetMap report UI are future work — not in this repo.

## Two-machine workflow (important)

Code + the one-time data build happen on the dev laptop. **All training/inference runs on a Mac M4 Pro (Apple Silicon / MPS).** No CUDA on Apple Silicon — YOLO and Faster R-CNN train on MPS; RF-DETR-L wants CUDA so it runs on Colab. Device is auto-selected everywhere via `shared/device.py` (CUDA → MPS → CPU); the same commands work on any machine.

## Setup & commands

Always use `uv`, never bare `pip` (see global instruction). Work inside the activated venv.

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
pytest -q                          # full suite (~99 test functions)
pytest tests/test_pipeline.py -q   # one file
pytest tests/test_pipeline.py::test_name -q   # one test
```

Data root is set by the `DRONISIGHT_DATA` env var (defaults to `/Volumes/dronisight`). It drives **every** path in `shared/config.py` — the two output DBs and the raw source folders. `config._SOURCE_FOLDER_NAMES` (currently 11 folders incl. `mem2 5th june`, `mem 7.1 5th june`, `mem10`, `4thJuneMem4/8`) is the single place to add/remove captures; `config.SUBSET_CLASSES` is the single place that maps each subset → its class list. Verify with:
```bash
python -c "from shared import config; print(config.YOLO_DB)"
```

Training / inference (run on the M4) — train in this order:
```bash
python -m train_yolo.train_pole                                   --version clahe --epochs 100 --imgsz 640  --batch 4
python -m train_yolo.train_components --subset component_above_1000 --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26m.pt
python -m train_yolo.train_components --subset component_below_1000 --version clahe --epochs 200 --imgsz 1280 --batch 4 --model yolo26m.pt
python -m inference.pipeline --image x.jpg \
  --pole-weights runs/pole/yolo/weights/best.pt \
  --comp-above-weights runs/component_above_1000/yolo/weights/best.pt \
  --comp-below-weights runs/component_below_1000/yolo/weights/best.pt
```
Rebuild data (only if class policy / CLAHE / split / dedup / oversampling logic changes — DBs are otherwise prebuilt):
```bash
python -m data_prep.build_dataset --subset all                       # add --no-balance to keep all images
python -m data_prep.verify_dataset --subset component_above_1000     # leakage + label-validity gate (run per subset)
```

`INSTRUCTION.md` is the deep run guide (M4 tuning, Colab, troubleshooting); `docs/superpowers/` holds the design spec and the 3 build plans.

## Architecture & non-obvious invariants

**Two parallel DBs, one source of truth.** `data_prep/build_dataset.py` writes both formats from the same parse: `yolo_train_db/` (YOLO `.txt`, primary) and `RF_DETR_Faster_RCNN_train_db/` (COCO JSON, for Faster R-CNN + RF-DETR). Each has three subsets: `pole/`, `component_above_1000/`, `component_below_1000/`. **VOC XML is the label source of truth** — any index-based YOLO `.txt` in the raw data is ignored; everything is re-derived from XML names via `shared/labels.py`.

**Annotation dedup.** Some flights were annotated twice in different folders (e.g. `mem7` and `mem 7.1 5th june` share 161 stems, 157 byte-identical). `config.DEDUP_PAIRS` lists `(primary, secondary)` folder pairs; `data_prep/dedup.py` drops the secondary copy when its `annotation_hash` (from `shared/labels.py`) equals the primary's, and keeps both when they differ. Dedup is **scoped to the configured pair** and matched by filename stem — never global, because DJI counters reset per card so the same stem in unrelated folders is a different image.

**Cross-folder image merge (per-annotator data).** The `6th june ` condition captures (subset `component_classification`) were labeled by **8-9 members who each annotated different classes over the same ~9k-photo pool** — so one physical photo lives in several `6thMem*AllTeam1` folders, each copy carrying only that member's classes. Treating each folder copy as its own image (the source-namespaced key) double-counts it, **leaks** the same photo across train/val/test, and trains the unlabeled-in-this-copy objects as background (**partial-label poisoning**). `data_prep/merge_annotations.py` (`merge_by_image_identity`, run in `build_dataset` before grouping) collapses every **byte-identical** copy into one entry holding the de-duplicated **union** of all members' boxes. It is keyed on image **content hash** (not stem — DJI counters reset per card), so it's enabled for **all** subsets (`config.MERGE_CROSS_FOLDER`) and is a pure no-op on disjoint captures; it also subsumes the byte-identical `mem7`/`mem7.1` overlap. `verify_dataset.assert_no_image_content_leakage` re-hashes the written DB and **fails if any photo appears in >1 split**. Real impact on the 6th-june build: 3641 copies → 2301 unique images, 1340 duplicate copies collapsed, +2469 boxes recovered.

**`orig` vs `clahe` variants.** Every image is stored twice: untouched and with adaptive CLAHE (LAB L-channel only, params chosen per-image from an exposure profile in `data_prep/preprocess.py` + `profile_images.py`). Train both variants and keep whichever wins on **val mAP** — CLAHE is a hypothesis, not a guarantee.

**YOLO label-layout invariant (easy to break).** Ultralytics resolves a label by swapping the last `/images/` → `/labels/` in the image path. So labels must mirror the image variant nesting: `labels/<split>/<orig|clahe>/`. Boxes are identical across variants, so the same `.txt` is written into both. After fixing any label/path issue, **delete `*.cache`** under the DB — a poisoned cache keeps reporting `0 images, N backgrounds` even after the fix.

**`data.yaml` is regenerated at train time.** The yaml written during the build hard-codes the build machine's absolute `path:`, which breaks after copying the DB. `train_yolo/train_pole.py` and `train_components.py` rewrite it to the current DB location on every run (`write_data_yaml`). `train_components.py` is parameterized by `--subset {component_above_1000,component_below_1000}` and reads its class list from `config.SUBSET_CLASSES`.

**Leakage-safe splits.** `data_prep/grouping.py` groups consecutive DJI frames into capture sequences (>60s gap = new group); `split.py` splits by group, never splitting one, stratified per source folder so every location appears in train. `verify_dataset.py` asserts no group spans multiple splits.

**Class policy — three subsets.** All 9 classes are now trained, split by frequency in `config.SUBSET_CLASSES`: `pole`; `component_above_1000` (the 4 high-frequency classes); `component_below_1000` (the 4 rare classes `vegetation, top_crossarm, om_crossarm, rust`, formerly ignored). `shared/labels.py` now maps all 9 to canonical names; only genuinely unknown names → `None`.

**Balancing differs per subset.** `component_above_1000` uses the down-cap (`balance.select_balanced` caps each class toward the rarest, `v_insulator`). `component_below_1000` instead uses **offline oversampling**: `data_prep/oversample.py` (`plan_oversample` + `augment_image`, via albumentations) duplicates rare-class **train** images with bbox-aware, orientation-aware augmentation until each class approaches the max count; val/test stay raw. `build_dataset` forces balance OFF for below_1000 and runs the oversampler only on the train split, writing `<key>_augN.jpg`. `sample_weights.csv` is still emitted.

**Inference must match training preprocessing.** `inference/pipeline.py` applies EXIF-orient + CLAHE **once** on the full frame; every pole crop inherits it, so both component models see their trained distribution. **Both** `component_above_1000` and `component_below_1000` detectors run on the **pole crop** (per design), each remapped to the full frame. Use `--no-clahe` only for `orig`-trained weights. Default imgsz mirrors training (pole 640, components 1280). The single-model CLIs (`infer_pole`, `infer_components`) do the same.

**YOLO weights fallback.** `train_yolo/weights.py` tries the preferred `yolo26x.pt` and falls back to `yolo11x.pt` with a printed warning if YOLO26 weights aren't fetchable on the installed Ultralytics version. The fallback is expected, not an error.

**Augmentation policy** (`shared/train_args.py`): poles/insulators have a strong vertical orientation prior → **no vertical flip**, only mild rotation. Components get wider scale-jitter + mixup + copy_paste (trained on full frames but run on zoomed crops, and the 4-class task is harder). Small data + large model → explicit weight_decay + dropout + early stopping; **prefer a smaller `--model` (yolo26m/l) over yolo26x** as the main anti-overfit lever, and watch the train/val gap in `results.png`.

**Detector abstraction.** `inference/backends.py` defines a `Detector` Protocol returning `Detection(class_name, confidence, box)`; `YoloDetector`, `TorchvisionDetector`, `RFDetrDetector` all conform, so the pipeline is backend-agnostic. Faster R-CNN uses 1-based labels (0 = background); the COCO DB stores 0-based `category_id` and the dataset/`backends` code converts.

## Notebooks

The 5 Colab notebooks in `notebooks/` are **generated** by `notebooks/build_notebooks.py` — edit the spec there and regenerate (`python -m notebooks.build_notebooks`), never hand-edit the `.ipynb`. `REPO_URL` is already set. They are Google-Drive-backed: data unzips from `MyDrive/dronisight/*.zip` to `/content/data` (with `DRONISIGHT_DATA` set there), and each trainer saves `runs/` back to Drive via `colab_utils.save_runs_to_drive()` since Colab runtimes are ephemeral. The full Colab/Drive walkthrough is `colab_instruction.md`.

## macOS / exFAT gotcha

The data SSD is exFAT, so macOS scatters AppleDouble `._*` sidecar files. Collection and verification filter them, and the build self-cleans them, but prefer `rsync --exclude '._*' --exclude '.DS_Store'` when copying DBs.
