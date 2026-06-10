# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **12-model object-detection pipeline for electric-pole inspection** from DJI drone imagery, plus a
web app to run it. The chain (all YOLO26 by default; Faster R-CNN / RF-DETR backends also supported):

1. **`pole`** — detect the pole on the full frame.
2. **5 component specialists** run on the padded pole crop, segregated by visual scale so they
   discriminate well: **`comp_wire`**, **`comp_insulator`** (h/v), **`comp_crossarm`** (straight/top/om —
   one softmax so the om↔straight mis-typing stops), **`comp_vegetation`**, **`comp_rust`**. Their boxes
   are remapped to the full frame and de-duplicated with class-aware NMS.
3. **6 condition specialists** run on a crop of each detected component, routed by family
   (`config.COMPONENT_TO_CONDITION_MODEL`): **`cond_v_insulator`**, **`cond_h_insulator`**,
   **`cond_straight_crossarm`**, **`cond_top_crossarm`**, **`cond_om_crossarm`**, **`cond_wire`**.

All 12 are **separate trainings producing separate weights**, chained only at inference
(`inference/pipeline.py`). The single source of truth for the subset→class-list map is
`config.SUBSET_CLASSES`; `config.SUBSETS` is the canonical 12-entry list.

## Two-machine workflow (important)

Code + the one-time data build happen on a dev laptop (no SSD-resident data needed for code work).
**Training/inference runs on Apple-Silicon (MPS)** or a CUDA box; RF-DETR-L wants CUDA → Colab. Device
is auto-selected everywhere via `shared/device.py` (CUDA → MPS → CPU) and threaded into every detector,
so the same commands work on any machine. On Apple Silicon prefix runs with
`PYTORCH_ENABLE_MPS_FALLBACK=1` so ops MPS lacks fall back to CPU instead of crashing.

## Setup & commands

Always use `uv`, never bare `pip` (global instruction). Work inside the activated venv.

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[app,dev]"   # everything: train+infer+web app+tests
pytest -q                                   # full suite (~164 tests)
pytest tests/test_pipeline.py::test_name -q # one test
```

Data root is set by the `DRONISIGHT_DATA` env var (defaults to `/Volumes/dronisight`); it drives every
path in `shared/config.py` (the two output DBs + raw source folders). Verify with
`python -c "from shared import config; print(config.YOLO_DB)"`.

**Train** (run on the M-series / Colab) — `train_components` is parameterized by `--subset` over the 11
non-pole subsets and reads its class list from `config.SUBSET_CLASSES`:
```bash
python -m train_yolo.train_pole       --version clahe --epochs 100 --imgsz 640  --batch 4
python -m train_yolo.train_components --subset comp_insulator --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26m.pt
python -m train_yolo.eval_yolo --subset comp_insulator --weights runs/comp_insulator/yolo/weights/best.pt --split val
```

**Inference** — weights are **auto-discovered by subset name** under `--weights-dir` (pole + 5 comp + 6
cond + the unified classifier for the condition ensemble), so no per-model flags:
```bash
python -m inference.pipeline --image path_or_dir --weights-dir models
```

**Web app** (`app/`, FastAPI + zero-build SPA): upload → animated "analysing" → structured report
(annotated frame w/ layer toggle, padded crops, condition badges, CSV/JSON). Auto-finds weights in
`models/` then `runs/` then `~/Downloads/runs`:
```bash
python -m app.server          # http://127.0.0.1:8000 ; override with DRONISIGHT_WEIGHTS
```

**Rebuild data** (only when class policy / CLAHE / split / dedup / balance / crop logic changes — DBs
are otherwise prebuilt):
```bash
python -m data_prep.build_dataset --subset all          # all 12 (or all_cond = the 6 condition subsets, or one subset)
python -m data_prep.verify_dataset --subset comp_insulator   # leakage + label-validity gate, per subset
```

`INSTRUCTION.md` / `TRAINING_GUIDE.md` / `windows_instruction.md` / `colab_instruction.md` are the deep
run guides; `docs/superpowers/` holds design specs.

## Architecture & non-obvious invariants

**Two parallel DBs, one parse.** `data_prep/build_dataset.py` writes both formats from the same parse:
`yolo_train_db/` (YOLO `.txt`, primary) and `RF_DETR_Faster_RCNN_train_db/` (COCO JSON, for Faster
R-CNN + RF-DETR), each with a folder per subset. **VOC XML is the label source of truth** — any
index-based YOLO `.txt` in the raw data is ignored; everything is re-derived from XML names via
`shared/labels.py` (maps all known names → canonical, genuinely-unknown → `None`).

**Content-hash cross-folder merge (per-annotator data).** The `6th june` condition captures were
labeled by 8-9 members who each annotated different classes over the same photo pool, so one physical
photo lives in several `6thMem*` folders with only partial labels. `data_prep/merge_annotations.py`
(`merge_by_image_identity`, keyed on image **content hash**, not stem — DJI counters reset per card)
collapses every byte-identical copy into one entry holding the **union** of all members' boxes, BEFORE
splitting. Enabled for **every** subset (`config.MERGE_CROSS_FOLDER`); a no-op on disjoint captures. It
subsumes the older stem-based `config.DEDUP_PAIRS` (`data_prep/dedup.py`). `verify_dataset` re-hashes
the written DB and **fails if any photo appears in >1 split**.

**Condition-conflict resolution (build time).** After merging, members may give the same object
different condition labels. `resolve_cross_class_conflicts` (`config.RESOLVE_CONDITION_CONFLICTS`, on
for every `cond_*`): defect beats normal, defect-vs-defect dropped as ambiguous.

**`orig` vs `clahe` variants.** Every image is stored twice: untouched and with adaptive CLAHE (LAB
L-channel; per-image params from `data_prep/preprocess.py` + `profile_images.py`). Train both, keep the
**val-mAP** winner. Models here are trained on `clahe`, so inference applies CLAHE by default.

**YOLO label-layout invariant (easy to break).** Ultralytics resolves a label by swapping the last
`/images/` → `/labels/`. Labels must mirror the image variant nesting: `labels/<split>/<orig|clahe>/`;
the same `.txt` is written into both variants (boxes are identical). After fixing any label/path issue,
**delete `*.cache`** under the DB or a poisoned cache keeps reporting `0 images, N backgrounds`.

**`data.yaml` is regenerated at train time.** The build-time yaml hard-codes the build machine's
absolute `path:`. `train_pole.py` / `train_components.py` rewrite it to the current DB location every
run (`emit_yolo.write_data_yaml`).

**Leakage-safe splits.** `data_prep/grouping.py` groups consecutive DJI frames into capture sequences
(>60s gap = new group); `split.py` splits by group (never splitting one), stratified per source folder.
Crop items inherit their source image's group, so crops of one photo never split across train/val/test.

**Balance = keep ALL data + oversample rare.** No class is ever capped (`config.BALANCE_TARGET == {}`).
Multi-class subsets keep 100% of real data and **oversample the rarer classes up toward the max class**
(`data_prep/oversample.py`, bbox/orientation-aware albumentations, TRAIN split only) bounded by
`MAX_OVERSAMPLE_FACTOR`. Single-class subsets (wire/vegetation/rust/pole) get nothing. Earlier fixed
caps (400/1000) discarded most data and are gone.

**Crop-aligned datasets (train scale == inference scale).** `data_prep/crop_align.py` + `config.CROP_ALIGN`:
`comp_*` use **"anchor"** mode (crop each frame to its **pole** box + `POLE_CROP_PAD`, keep visible
component boxes); `cond_*` use **"self"** mode (crop to each **component** box + `CONDITION_CROP_PAD`).
CLAHE is applied to the full frame then sliced (identical to inference).

**Padding is split build-vs-inference (`shared/config.py`).** `POLE_CROP_PAD=0.05`,
`CONDITION_CROP_PAD=0.25` (BUILD pad — annotators boxed to the hinge; changing it requires rebuilding
`cond_*`). `CONDITION_INFER_PAD=0.40` is the LARGER **inference** pad: the component detector boxes the
head (tighter than the GT box) so 0.25 would clip the band — no rebuild needed. `COMPONENT_CROP_PAD=0.15`
pads the **saved/display** crop only (the condition model is fed the `CONDITION_INFER_PAD` crop).

**Inference pipeline specifics (`inference/pipeline.py`).** Single-image flow: EXIF-orient + CLAHE once
on the full frame → pole detect (`nms_detections` removes duplicate poles) → 5 component detectors on
the pole crop → remap → **class-aware** `nms_components` (same class deduped at `COMPONENT_SAME_CLASS_IOU`,
different classes kept unless `COMPONENT_NMS_IOU`) → route each surviving component to its condition
detector. `discover_weights(weights_dir, subsets)` globs `**/<subset>/**/weights/best.pt` (falls back to
`last.pt`). `build_component_detector` / `build_condition_detector` apply the two model-substitution
policies below. Result schema: `poles[].components[]`, each with `condition` (best) + `conditions[]`
(multi-label) + `crop_path`; `result_to_rows`/`write_csv` flatten to `result.csv`.

**Vegetation override (`config.COMPONENT_WEIGHTS_OVERRIDE`).** The solo `comp_vegetation` model
over-fires (single-class → nothing to lose to). Vegetation is instead served by the older
`component_below_1000_crop` detector wrapped in a `FilteredDetector` that **keeps only `vegetation`**
(its other 3 classes belong to comp_crossarm/comp_rust). Falls back to the solo weights if absent.

**Condition ensemble + defect priority (`config.CONDITION_ENSEMBLE`).** Family specialists have low
DEFECT recall (real broken/chipped get shown `normal`). Each family runs the specialist and/or the OLD
unified 14-class classifier (`component_classification_crop`), UNIONed via `EnsembleDetector`; a
per-family `defect_conf` floor drops weak false defects. `resolve_condition_overlaps` then applies
**defect priority** (intersection-over-smaller): a defect is never removed by a higher-confidence
`normal`; `normal` survives only if no defect overlaps it; multiple defects coexist. These thresholds
were calibrated on the val studies in `scripts/` (`condition_conflict_study.py`,
`condition_ensemble_study.py`) — rerun them after any retrain.

**Detector abstraction (`inference/backends.py`).** A `Detector` Protocol returning
`Detection(class_name, confidence, box)`; `YoloDetector`, `TorchvisionDetector`, `RFDetrDetector`,
`FilteredDetector`, `EnsembleDetector` all conform, so the pipeline is backend-agnostic. Faster R-CNN
uses 1-based labels (0 = background); the COCO DB stores 0-based `category_id` and the code converts.
`TorchvisionDetector` defaults `min_size=2000` to match training (serving at torchvision's 800 collapses
small objects). `train_yolo/weights.py` tries `yolo26x.pt` and falls back to `yolo11x.pt` with a printed
warning — expected, not an error.

**Web app (`app/`).** `app/server.py` (FastAPI): `POST /api/analyze` → job; poll `/api/jobs/{id}`; GET
result; `/api/health`; per-job artifacts served read-only under `/files`. A single-worker
`ThreadPoolExecutor` serializes GPU/MPS work; uploads are streamed with a size cap; finished jobs + run
dirs are evicted beyond `DRONISIGHT_MAX_JOBS`. `app/inference_service.py` loads the detectors once
(cached) and builds the URL-bearing report (summary + per-pole crops + attention flags). The SPA in
`app/static/` is plain HTML/CSS/JS (no build step).

## Notebooks

`colab_train_yolo.ipynb` trains the 11 subsets on Colab; `colab_infer.ipynb` runs the full pipeline from
Drive weights. **Colab dep gotchas baked into both:** install the repo `--no-deps` (the full deps pull
`rfdetr`, whose torch/torchvision pins corrupt Colab's CUDA stack → `torchvision::nms` error), force
`ultralytics>=8.4.60` (8.3.x silently degrades yolo26 → nano), and pin `pillow==11.2.1` (11.3+ `ImageText`
`_Ink` import is broken on Colab). The 5 notebooks in `notebooks/` are **generated** by
`notebooks/build_notebooks.py` — edit the spec there and regenerate, never hand-edit the `.ipynb`.

## macOS / exFAT gotcha

The data SSD is exFAT, so macOS scatters AppleDouble `._*` + `.DS_Store` sidecars (and zips add
`__MACOSX/`). YOLO crashes trying to read `._IMG.jpg` as an image. Collection/verification/build self-clean
these, the Colab loader strips them, and you should prefer `rsync --exclude '._*' --exclude '.DS_Store'`
when copying DBs/weights.
