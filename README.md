# trainDronisight

Two-stage object-detection pipeline for **electric-pole inspection** from DJI drone imagery.
Detects poles, crops them, then detects pole components (insulators, wires, crossarms) on the crop —
the foundation for downstream condition assessment.

> **Status:** Detection v1 — data-prep + 3 trainers + inference pipeline. **82 tests passing.**
> Condition classifier, point/scoring system, and the OpenStreetMap report UI are **future work** (see [Roadmap](#roadmap)).

---

## What it does

```
image ─▶ Model 1 (pole detector, full frame)
            └─ crop to pole box (+5% margin)
                  └─▶ Model 2 (component detector, on crop)
                        └─ remap boxes to full frame + crop each component
                              └─▶ structured JSON  { poles[] → components[] }
```

- **Model 1 — pole detector:** one class, `pole`, trained on full 12 MP frames.
- **Model 2 — component detector:** `wire`, `h_insulator`, `v_insulator`, `crossarm_stright`, trained on full frames.
- **Three model families**, so you can compare: **YOLO26x** (primary, Apple-Silicon/MPS), **Faster R-CNN** (torchvision), **RF-DETR-L** (Roboflow, CUDA-preferred).
- Device selection is automatic everywhere: **CUDA → MPS → CPU**.

## Where to start

➡️ **Read [`INSTRUCTION.md`](INSTRUCTION.md)** — the deep, step-by-step guide for running everything on the **Mac M4 Pro** (data, setup, training, inference, Colab, troubleshooting, M4 tuning).

Quick version:
```bash
git clone https://github.com/arupa444/trainDronisight.git && cd trainDronisight
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
export DRONISIGHT_DATA=/path/to/your/data        # folder holding the two DBs
python -m train_yolo.train_pole                  # Model 1 (MPS)
python -m train_yolo.train_components            # Model 2 (MPS)
python -m inference.pipeline --image <img.jpg> \
  --pole-weights runs/pole/yolo/weights/best.pt \
  --comp-weights runs/components/yolo/weights/best.pt
```

## Repository layout

| Path | Purpose |
|------|---------|
| `shared/` | `config` (paths/classes), `device` (CUDA→MPS→CPU), `labels` (VOC parse/normalize), `train_args` (YOLO aug policy) |
| `data_prep/` | XML→clean labels, merge, grouped leakage-safe split, class balancing, CLAHE, emit YOLO + COCO DBs |
| `train_yolo/` | YOLO26x trainers (pole + components), with `yolo26x → yolo11x` fallback |
| `train_faster_rcnn/` | torchvision Faster R-CNN (COCO) |
| `train_rf_detr/` | RF-DETR-L (Roboflow, COCO view) |
| `inference/` | `Detection`/`Detector` backends, crop geometry, single-model CLIs, two-stage `pipeline` |
| `notebooks/` | 5 generated Colab notebooks (set `REPO_URL` first) |
| `tests/` | 82 unit tests |
| `docs/superpowers/` | design **spec** + the 3 implementation **plans** |

## The data DBs

Built once by `data_prep` onto the data root (default `/Volumes/dronisight`, override with `DRONISIGHT_DATA`):

```
<DRONISIGHT_DATA>/
├── yolo_train_db/                 # YOLO format (primary)
│   ├── pole/        images/{train,val,test}/{orig,clahe}/  labels/  data_{orig,clahe}.yaml
│   └── components/  …same…        + manifest.csv, sample_weights.csv, dataset_meta.json
└── RF_DETR_Faster_RCNN_train_db/  # COCO format
    ├── pole/        images/…       annotations/instances_{split}_{orig,clahe}.json
    └── components/  …same…
```

- **`orig` vs `clahe`:** every image is stored both untouched and with adaptive CLAHE (exposure fix for backlit/blown-sky frames). Train both, keep whichever wins on val mAP.
- **Splits** are grouped by capture sequence (no near-duplicate leakage) and stratified across all 7 capture locations.
- Built from ~1,041 annotated frames: **pole** 995 imgs (743/204/48), **components** 1,023 imgs (810/186/27).

## Models & current dataset notes

- **Primary:** YOLO26x via MPS on the M4 Pro. Falls back to `yolo11x` (with a printed warning) if YOLO26 weights aren't reachable on your Ultralytics version.
- **Class balance:** components are capped toward the rarest kept class, but multi-label co-occurrence leaves residual imbalance (`wire` ≫ `crossarm_stright`). `sample_weights.csv` is emitted so you can try inverse-frequency weighted sampling instead — compare on val mAP.
- Rare classes (`rust`, `om_crossarm`, `top_crossarm`, `vegetation`, all <1000 instances) are **ignored** in v1, not deleted — re-introduce once more annotations exist.

## Roadmap (out of scope for this repo today)

1. **Condition classifier** — classify each component crop as normal/defective (broken insulator, bent crossarm…). *Blocked on a separate labeling effort: current annotations mark object type only, not condition.*
2. **Point/scoring system** — per-pole health score from component conditions.
3. **Report + OpenStreetMap UI** — UUID (lat/long hash) → map pin → per-pole detail page.

## Development

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
pytest -q          # 82 tests
```

Design rationale lives in [`docs/superpowers/specs/`](docs/superpowers/specs/); the task-by-task build plans in [`docs/superpowers/plans/`](docs/superpowers/plans/).
