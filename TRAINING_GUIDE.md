# Training & Inference Guide — all 4 stages × 3 model families

Pipeline: **pole** → crop → **component_above_1000** + **component_below_1000** (both on the pole crop)
→ crop each component → **component_classification** (condition, 14 classes, runs on the component crop).

Three independent model families per stage: **YOLO26x**, **Faster R-CNN**, **RF-DETR-L**. They are
separate trainings producing separate weights, chained only at inference. Train all, then keep the
family that wins on **val mAP@.5** for each stage.

## 0. Where each model runs

| Family | Device | Where | Why |
|---|---|---|---|
| YOLO26x | MPS (Apple Silicon) | **M4 Pro** | trains fine on MPS |
| Faster R-CNN | MPS | **M4 Pro** | trains fine on MPS |
| RF-DETR-L | CUDA only | **Colab** (notebook `03_train_rf_detr`) | DINOv2 backbone needs CUDA |

Device is auto-selected (`shared/device.py`: CUDA→MPS→CPU) — the same command works everywhere.
Always train `--version clahe` and `--version orig`, keep whichever wins on val mAP (CLAHE is a hypothesis).

Weights land at:
- YOLO  → `runs/<subset>/yolo/weights/best.pt`
- FRCNN → `runs/<subset>/faster_rcnn/best.pt`
- RFDETR→ `runs/<subset>/rfdetr/checkpoint_best_ema.pth`

The 4 subsets, in train order: `pole`, `component_above_1000`, `component_below_1000`, `component_classification`.

---

## 1. TRAINING

### 1A. YOLO26x  (run on the M4)

```bash
source .venv/bin/activate
# yolo26x @ 1280 is heavy on MPS — if you hit OOM, drop --batch (4→2) or use --model yolo26l.pt.
# If yolo26x.pt can't be fetched on the installed Ultralytics, it auto-falls-back to yolo11x.pt (a warning, not an error).

# 1) POLE — fills the frame -> imgsz 640
python -m train_yolo.train_pole       --version clahe --epochs 100 --imgsz 640  --batch 4 --model yolo26x.pt

# 2) COMPONENT_ABOVE_1000 (wire, h_insulator, v_insulator, crossarm_stright) -> 1280 for thin wires
python -m train_yolo.train_components  --subset component_above_1000 --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26x.pt

# 3) COMPONENT_BELOW_1000 (vegetation, top_crossarm, om_crossarm, rust) — oversampled train, more epochs
python -m train_yolo.train_components  --subset component_below_1000 --version clahe --epochs 200 --imgsz 1280 --batch 4 --model yolo26x.pt

# 4) COMPONENT_CLASSIFICATION (14 condition classes; train balanced ~400/class)
python -m train_yolo.train_components  --subset component_classification --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26x.pt
```
Watch `runs/<subset>/yolo/results.png` for the train/val gap. yolo26x is the biggest anti-overfit risk on
this small data — if val mAP plateaus while train keeps climbing, step down to `yolo26l.pt`/`yolo26m.pt`.

### 1B. Faster R-CNN  (run on the M4, or Colab `02_train_faster_rcnn`)

```bash
# min_size 2000 keeps thin wires/insulators from being shrunk away (torchvision default 800 loses them).
# batch 2 is the safe default at that resolution; raise --workers to cpu_count for fast data loading.
python -m train_faster_rcnn.train --subset pole                   --version clahe --epochs 30 --batch 2
python -m train_faster_rcnn.train --subset component_above_1000   --version clahe --epochs 30 --batch 2
python -m train_faster_rcnn.train --subset component_below_1000   --version clahe --epochs 30 --batch 2
python -m train_faster_rcnn.train --subset component_classification --version clahe --epochs 60 --batch 2
```
Has val-loss early stopping (`--patience 7`), cosine LR, augmentation. `best.pt` is the lowest-val-loss
checkpoint. FRCNN "loss" is a sum of 4 losses — magnitude is meaningless; only the train-vs-val *trend* matters.

Per-class AP (after training):
```bash
python -m train_faster_rcnn.eval --subset component_above_1000 --version clahe --split test
# defaults: --weights runs/<subset>/faster_rcnn/best.pt  --conf 0.05  --min-size 2000
```

### 1C. RF-DETR-L  (run on Colab — notebook `03_train_rf_detr`; CUDA only)

```bash
# resolution must be divisible by 56 (training). 1120 is divisible by BOTH 56 and 32 (predict),
# so there's no inference-shape rounding for the high-res classification stage.
python -m train_rf_detr.train --subset pole                   --version clahe --epochs 50 --batch 4 --resolution 728
python -m train_rf_detr.train --subset component_above_1000   --version clahe --epochs 50 --batch 4 --resolution 728
python -m train_rf_detr.train --subset component_below_1000   --version clahe --epochs 50 --batch 4 --resolution 728
python -m train_rf_detr.train --subset component_classification --version clahe --epochs 50 --batch 8 --resolution 1120
```
On Colab the notebook unzips the COCO DB, runs this, and `save_runs_to_drive()` copies `runs/` back
(runtimes are ephemeral). Use a bigger `--resolution` (1008/1120) if the GPU has memory — helps thin wires.

---

## 2. INFERENCE — single model, single stage

The single-stage CLIs (`infer_pole`, `infer_components`) are **YOLO** (class names are embedded in the `.pt`).
CLAHE is applied by default; add `--no-clahe` only for `orig`-trained weights.

```bash
# pole only
python -m inference.infer_pole       --image x.jpg --weights runs/pole/yolo/weights/best.pt --imgsz 640

# any component/condition detector on a (already-cropped) image
python -m inference.infer_components  --image pole_crop.jpg --weights runs/component_above_1000/yolo/weights/best.pt --imgsz 1280
python -m inference.infer_components  --image pole_crop.jpg --weights runs/component_below_1000/yolo/weights/best.pt --imgsz 1280
python -m inference.infer_components  --image comp_crop.jpg --weights runs/component_classification/yolo/weights/best.pt --imgsz 1280
```

Per-family single-stage detection is also reachable through the backend classes in
`inference/backends.py` (`YoloDetector`, `TorchvisionDetector`, `RFDetrDetector`) — all return the same
`Detection(class_name, confidence, box)`, so they're drop-in interchangeable.

---

## 3. COMBINATIONS — the chained pipeline

`inference/pipeline.py` runs **pole → crop → BOTH component detectors on the pole crop → remap → JSON**.
EXIF-orient + CLAHE are applied **once** on the full frame so every crop inherits the trained distribution.

### 3A. All-YOLO (default, fastest)
```bash
python -m inference.pipeline --image x.jpg \
  --pole-weights        runs/pole/yolo/weights/best.pt \
  --comp-above-weights  runs/component_above_1000/yolo/weights/best.pt \
  --comp-below-weights  runs/component_below_1000/yolo/weights/best.pt \
  --out runs/inference/result.json --crop-dir runs/inference/crops
# defaults: --pole-conf 0.12 (recall-leaning) --comp-conf 0.25 --pole-imgsz 640 --comp-imgsz 1280 --pole-pad 0.05
```

### 3B. Mixed backends (e.g. RF-DETR for the hard component stages, YOLO for pole)
`--pole-backend / --comp-above-backend / --comp-below-backend` each accept `yolo` or `rfdetr`.
Point the matching `*-weights` at that family's checkpoint.
```bash
python -m inference.pipeline --image x.jpg \
  --pole-backend yolo   --pole-weights        runs/pole/yolo/weights/best.pt \
  --comp-above-backend rfdetr --comp-above-weights runs/component_above_1000/rfdetr/checkpoint_best_ema.pth \
  --comp-below-backend rfdetr --comp-below-weights runs/component_below_1000/rfdetr/checkpoint_best_ema.pth \
  --rfdetr-resolution 728 \
  --out runs/inference/result.json
```
Faster R-CNN is also a valid backend via `TorchvisionDetector`; the pipeline CLI currently exposes
`{yolo, rfdetr}` flags — to drive FRCNN in the chain, use it as the single-stage detector or extend the
backend flag (small change — ask and I'll wire `frcnn` into the CLI choices).

### 3C. The 4th stage — condition classification (component_classification)
**Status:** `pipeline.py` chains 3 stages (pole + above + below). The condition classifier is trained and
ready, but is **not yet wired as the 4th stage**. Two ways to use it today:

1. **Standalone** on a component crop the pipeline already saved to `--crop-dir`:
   ```bash
   python -m inference.infer_components --image runs/inference/crops/<...>_above0.jpg \
     --weights runs/component_classification/yolo/weights/best.pt --imgsz 1280
   ```
2. **Wire it into pipeline.py** as the automatic 4th step (each detected above/below component crop →
   condition detector → condition field on every component in the JSON). This is a clean, ~30-line
   addition — say the word and I'll add `--cond-weights/--cond-backend` and the per-component call.

---

## 4. Model-selection checklist (after all trainings)

1. For each stage, compare **val mAP@.5** across YOLO / FRCNN / RF-DETR and across `clahe` vs `orig`.
2. Keep the single winning checkpoint per stage; that's what the pipeline points at.
3. Sanity-test the chosen combo on a frame containing many classes (not a sparse crop) before trusting it.
4. Universal ceiling: thin wires + rare conditions are data-limited, not model-limited — judge them on recall.
