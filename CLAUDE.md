# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two-stage object-detection pipeline for **electric-pole inspection** from DJI drone imagery:
**Model 1** detects the `pole` on the full frame → crop to the pole box (+pad) → **Model 2** detects
components (`wire`, `h_insulator`, `v_insulator`, `crossarm_stright`) on the *crop* → boxes remapped to
the full frame, each component re-cropped → structured JSON. The two models are **separate trainings**
producing separate weights, chained only at inference (`inference/pipeline.py`).

Detection-only (v1). Condition classifier, scoring, and the OpenStreetMap report UI are future work — not in this repo.

## Two-machine workflow (important)

Code + the one-time data build happen on the dev laptop. **All training/inference runs on a Mac M4 Pro (Apple Silicon / MPS).** No CUDA on Apple Silicon — YOLO and Faster R-CNN train on MPS; RF-DETR-L wants CUDA so it runs on Colab. Device is auto-selected everywhere via `shared/device.py` (CUDA → MPS → CPU); the same commands work on any machine.

## Setup & commands

Always use `uv`, never bare `pip` (see global instruction). Work inside the activated venv.

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
pytest -q                          # full suite (~87 test functions)
pytest tests/test_pipeline.py -q   # one file
pytest tests/test_pipeline.py::test_name -q   # one test
```

Data root is set by the `DRONISIGHT_DATA` env var (defaults to `/Volumes/dronisight`). It drives **every** path in `shared/config.py` — the two output DBs and the raw source folders (`mem2..mem8` + `4thJuneMem4` + `4thJuneMem8`; the `SOURCE_DIRS` list in `config.py` is the single place to add more). Verify with:
```bash
python -c "from shared import config; print(config.YOLO_DB)"
```

Training / inference (run on the M4):
```bash
python -m train_yolo.train_pole       --version clahe --epochs 100 --imgsz 640  --batch 4
python -m train_yolo.train_components  --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26m.pt
python -m inference.pipeline --image x.jpg --pole-weights runs/pole/yolo/weights/best.pt --comp-weights runs/components/yolo/weights/best.pt
```
Rebuild data (only if class policy / CLAHE / split logic changes — DBs are otherwise prebuilt):
```bash
python -m data_prep.build_dataset --subset all          # add --no-balance to keep all images
python -m data_prep.verify_dataset --subset pole        # leakage + label-validity gate
```

`INSTRUCTION.md` is the deep run guide (M4 tuning, Colab, troubleshooting); `docs/superpowers/` holds the design spec and the 3 build plans.

## Architecture & non-obvious invariants

**Two parallel DBs, one source of truth.** `data_prep/build_dataset.py` writes both formats from the same parse: `yolo_train_db/` (YOLO `.txt`, primary) and `RF_DETR_Faster_RCNN_train_db/` (COCO JSON, for Faster R-CNN + RF-DETR). Each has `pole/` and `components/` subsets. **VOC XML is the label source of truth** — any index-based YOLO `.txt` in the raw data is ignored; everything is re-derived from XML names via `shared/labels.py`.

**`orig` vs `clahe` variants.** Every image is stored twice: untouched and with adaptive CLAHE (LAB L-channel only, params chosen per-image from an exposure profile in `data_prep/preprocess.py` + `profile_images.py`). Train both variants and keep whichever wins on **val mAP** — CLAHE is a hypothesis, not a guarantee.

**YOLO label-layout invariant (easy to break).** Ultralytics resolves a label by swapping the last `/images/` → `/labels/` in the image path. So labels must mirror the image variant nesting: `labels/<split>/<orig|clahe>/`. Boxes are identical across variants, so the same `.txt` is written into both. After fixing any label/path issue, **delete `*.cache`** under the DB — a poisoned cache keeps reporting `0 images, N backgrounds` even after the fix.

**`data.yaml` is regenerated at train time.** The yaml written during the build hard-codes the build machine's absolute `path:`, which breaks after copying the DB. `train_yolo/train_pole.py` and `train_components.py` rewrite it to the current DB location on every run (`write_data_yaml`).

**Leakage-safe splits.** `data_prep/grouping.py` groups consecutive DJI frames into capture sequences (>60s gap = new group); `split.py` splits by group, never splitting one, stratified per source folder so every location appears in train. `verify_dataset.py` asserts no group spans multiple splits.

**Class policy.** Only classes with >1000 instances are kept (the 1 pole + 4 component classes in `config.py`). Rare classes (`rust`, `om_crossarm`, `top_crossarm`, `vegetation`) normalize to `None` in `labels.py` — **ignored, never deleted** from source, so they can be reintroduced once more annotations exist.

**Balancing.** `balance.py` caps each subset toward the rarest kept class, but multi-label co-occurrence leaves residual imbalance (`wire` ≫ `crossarm_stright`). `sample_weights.csv` is emitted for inverse-frequency weighted sampling as an alternative — compare on val.

**Inference must match training preprocessing.** `inference/pipeline.py` applies EXIF-orient + CLAHE **once** on the full frame; every pole crop inherits it, so Model 2 sees its trained distribution. Use `--no-clahe` only for `orig`-trained weights. Default imgsz mirrors training (pole 640, components 1280). The single-model CLIs (`infer_pole`, `infer_components`) do the same.

**YOLO weights fallback.** `train_yolo/weights.py` tries the preferred `yolo26x.pt` and falls back to `yolo11x.pt` with a printed warning if YOLO26 weights aren't fetchable on the installed Ultralytics version. The fallback is expected, not an error.

**Augmentation policy** (`shared/train_args.py`): poles/insulators have a strong vertical orientation prior → **no vertical flip**, only mild rotation. Components get wider scale-jitter + mixup + copy_paste (trained on full frames but run on zoomed crops, and the 4-class task is harder). Small data + large model → explicit weight_decay + dropout + early stopping; **prefer a smaller `--model` (yolo26m/l) over yolo26x** as the main anti-overfit lever, and watch the train/val gap in `results.png`.

**Detector abstraction.** `inference/backends.py` defines a `Detector` Protocol returning `Detection(class_name, confidence, box)`; `YoloDetector`, `TorchvisionDetector`, `RFDetrDetector` all conform, so the pipeline is backend-agnostic. Faster R-CNN uses 1-based labels (0 = background); the COCO DB stores 0-based `category_id` and the dataset/`backends` code converts.

## Notebooks

The 5 Colab notebooks in `notebooks/` are **generated** by `notebooks/build_notebooks.py` — edit the spec there and regenerate (`python -m notebooks.build_notebooks`), never hand-edit the `.ipynb`. Set `REPO_URL` before regenerating.

## macOS / exFAT gotcha

The data SSD is exFAT, so macOS scatters AppleDouble `._*` sidecar files. Collection and verification filter them, and the build self-cleans them, but prefer `rsync --exclude '._*' --exclude '.DS_Store'` when copying DBs.
