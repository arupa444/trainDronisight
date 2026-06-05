# trainDronisight

Two-stage object-detection pipeline for **electric-pole inspection** from DJI drone imagery.
Detects poles, crops them, then detects pole components (insulators, wires, crossarms) on the crop —
the foundation for downstream condition assessment.

> **Status:** Detection v1 — data-prep + 3 trainers + 3-detector inference pipeline. **99 tests passing.**
> Condition classifier, point/scoring system, and the OpenStreetMap report UI are **future work** (see [Roadmap](#roadmap)).

---

## What it does

```
image ─▶ Model 1 (pole detector, full frame)
            └─ crop to pole box (+5% margin)
                  ├─▶ Model 2a (component_above_1000 detector, on crop)
                  └─▶ Model 2b (component_below_1000 detector, on crop)
                        └─ remap boxes to full frame + crop each component
                              └─▶ structured JSON { poles[] → components_above[], components_below[] }
```

- **Model 1 — pole detector:** one class, `pole`, trained on full 12 MP frames.
- **Model 2a — component_above_1000:** the high-frequency classes `wire`, `h_insulator`, `v_insulator`, `crossarm_stright` (balance-capped toward the rarest, `v_insulator`).
- **Model 2b — component_below_1000:** the rare classes `vegetation`, `top_crossarm`, `om_crossarm`, `rust` (train split offline-oversampled with bbox-aware augmentation to equalize class sizes).
- **Three model families**, so you can compare: **YOLO26x** (primary, Apple-Silicon/MPS), **Faster R-CNN** (torchvision), **RF-DETR-L** (Roboflow, CUDA-preferred).
- Device selection is automatic everywhere: **CUDA → MPS → CPU**.

## Where to start

➡️ **Read [`INSTRUCTION.md`](INSTRUCTION.md)** — the deep, step-by-step guide for running everything on the **Mac M4 Pro** (data, setup, training, inference, Colab, troubleshooting, M4 tuning).

Quick version:
```bash
git clone https://github.com/arupa444/trainDronisight.git && cd trainDronisight
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
export DRONISIGHT_DATA=/path/to/your/data        # folder holding the two DBs
python -m train_yolo.train_pole                                      # Model 1 (MPS)
python -m train_yolo.train_components --subset component_above_1000  # Model 2a
python -m train_yolo.train_components --subset component_below_1000  # Model 2b
python -m inference.pipeline --image <img.jpg> \
  --pole-weights runs/pole/yolo/weights/best.pt \
  --comp-above-weights runs/component_above_1000/yolo/weights/best.pt \
  --comp-below-weights runs/component_below_1000/yolo/weights/best.pt
```

## Repository layout

| Path | Purpose |
|------|---------|
| `shared/` | `config` (paths, `SUBSET_CLASSES`, dedup pairs), `device` (CUDA→MPS→CPU), `labels` (VOC parse/normalize + annotation hash), `train_args` (YOLO aug policy) |
| `data_prep/` | XML→clean labels, annotation `dedup`, grouped leakage-safe split, class balancing, `oversample` (below_1000), CLAHE, emit YOLO + COCO DBs |
| `train_yolo/` | YOLO26x trainers: `train_pole` + `train_components --subset {above,below}_1000`, with `yolo26x → yolo11x` fallback |
| `train_faster_rcnn/` | torchvision Faster R-CNN (COCO) |
| `train_rf_detr/` | RF-DETR-L (Roboflow, COCO view) |
| `inference/` | `Detection`/`Detector` backends, crop geometry, single-model CLIs, three-detector `pipeline` |
| `notebooks/` | 5 generated Colab notebooks (Drive-backed) — see [`colab_instruction.md`](colab_instruction.md) |
| `tests/` | 99 unit tests |
| `docs/superpowers/` | design **spec** + the 3 implementation **plans** |

## The data DBs

Built once by `data_prep` onto the data root (default `/Volumes/dronisight`, override with `DRONISIGHT_DATA`):

```
<DRONISIGHT_DATA>/
├── yolo_train_db/                       # YOLO format (primary)
│   ├── pole/                  images/{train,val,test}/{orig,clahe}/  labels/  data_{orig,clahe}.yaml
│   ├── component_above_1000/  …same…    + manifest.csv, sample_weights.csv, dataset_meta.json
│   └── component_below_1000/  …same…    (train split has extra <key>_augN oversampled copies)
└── RF_DETR_Faster_RCNN_train_db/        # COCO format
    ├── pole/                  images/…  annotations/instances_{split}_{orig,clahe}.json
    ├── component_above_1000/  …same…
    └── component_below_1000/  …same…
```

- **`orig` vs `clahe`:** every image is stored both untouched and with adaptive CLAHE (exposure fix for backlit/blown-sky frames). Train both, keep whichever wins on val mAP.
- **Splits** are grouped by capture sequence (no near-duplicate leakage) and stratified across all 11 capture folders.
- **Dedup:** re-annotated overlaps (`mem7` ↔ `mem 7.1 5th june`) are de-duplicated by annotation hash — identical → one copy, different → both kept.
- Built from 11 source folders (post-dedup instance counts): pole 2424 · wire 3532 · h_insulator 3358 · crossarm_stright 2971 · v_insulator 2669 · top_crossarm 637 · vegetation 634 · om_crossarm 444 · rust 225.

## Models & current dataset notes

- **Primary:** YOLO26x via MPS on the M4 Pro. Falls back to `yolo11x` (with a printed warning) if YOLO26 weights aren't reachable on your Ultralytics version.
- **`component_above_1000` balance:** capped toward the rarest kept class, `v_insulator` (~2669). Multi-label co-occurrence leaves residual imbalance and the cap removes few images; `sample_weights.csv` is emitted for inverse-frequency weighted sampling as an alternative — compare on val mAP.
- **`component_below_1000` balance:** the 4 rare classes (rust 225 … top_crossarm 637) are no longer ignored — the **train** split is offline-oversampled with bbox-aware augmentation (`data_prep/oversample.py`) to equalize class sizes; val/test stay raw for honest evaluation. Expect modest AP here — oversampling raises sample count, not true diversity; report per-class AP.

## Roadmap (out of scope for this repo today)

1. **Condition classifier** — classify each component crop as normal/defective (broken insulator, bent crossarm…). *Blocked on a separate labeling effort: current annotations mark object type only, not condition.*
2. **Point/scoring system** — per-pole health score from component conditions.
3. **Report + OpenStreetMap UI** — UUID (lat/long hash) → map pin → per-pole detail page.

## Development

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
pytest -q          # 99 tests
```

Design rationale lives in [`docs/superpowers/specs/`](docs/superpowers/specs/); the task-by-task build plans in [`docs/superpowers/plans/`](docs/superpowers/plans/).
