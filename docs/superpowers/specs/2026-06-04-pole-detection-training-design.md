# Electric-Pole Inspection — Detection Training Pipeline (v1)

**Date:** 2026-06-04
**Status:** Approved design — ready for implementation plan
**Scope of this spec:** Data preparation + training + inference for the **object-detection** stages only.
**Explicitly OUT of scope (future specs):** condition/defect classifier, point/scoring system, report generation, UUID/geo lookup, OpenStreetMap UI.

---

## 1. Problem & context

Drone (DJI) imagery of 11kV electric poles is used to assess pole health. The end product is a two-stage
detection pipeline:

1. **Model 1 (pole detector):** detect `pole` on the full frame. At **inference** the predicted box is cropped.
2. **Model 2 (component detector):** detect pole components. **Trained on full frames**; at inference it is run
   on the cropped pole region from Model 1.

Three model families are built so they can be compared, with **YOLO26x as primary** (best Apple-Silicon support):
- **YOLO26x** (Ultralytics) — primary, trains on the M4 Pro via MPS.
- **Faster R-CNN** (torchvision) — comparison baseline.
- **RF-DETR-L** (Roboflow, Large) — comparison; CUDA strongly preferred (use Colab).

### Hardware
- **Training machine:** MacBook M4 Pro, **24 GB unified memory** (≈16–18 GB usable for training; Apple Silicon, no CUDA).
- Data currently lives on an external SSD mounted at `/Volumes/dronisight` (also accessed from a second M1/8GB laptop).
- **Cloud option:** Google Colab notebooks provided for CUDA training (esp. RF-DETR / Faster R-CNN).
- **Device priority everywhere:** `CUDA → MPS → CPU` (single shared selector).

---

## 2. Source data — findings (the inputs we must handle)

### 2.1 Inventory
- ~15,600 raw `.JPG` across `mem1`–`mem8`. **Only ~1,200 are annotated.** `mem1` has **no** labels (raw only) → unused in v1.
- Annotated folders: **`mem2`–`mem8`** (7 annotators / 7 locations).
- Each annotated image exported 3 ways: Pascal VOC `.xml`, YOLO `.txt`, LabelMe-style `.json`.
- **Mixed resolution:** `4032×3024` (~750 imgs) and `4096×3072` (~291 imgs). Both 12MP, 4:3.

### 2.2 Label-quality problems (MUST be fixed in data-prep)
- **Inconsistent class index order across annotators** → the index-based YOLO `.txt` files are unreliable
  (e.g. `mem6` reorders classes; `mem2` swaps `top_crossarm`/`vegetation` vs `mem3/4/7`; `mem5` has a stray 10th index).
  → **Source of truth = the name-based `.xml` files.** YOLO/COCO labels are regenerated from XML.
- **Crossarm name variants must be merged** to a single canonical class:
  - `crossarm_stright` (1529) + `crossarm_Stright` (107, in mem8) + `crossarmStright` (25, the mem5 stray 10th class) → **`crossarm_stright`** (1661).
- **Severe class imbalance** (instances): `wire` 3532, `h_insulator` 3358, `v_insulator` 2669, `crossarm_stright` 1661, `pole` 1072, `vegetation` 297, `top_crossarm` 252, `om_crossarm` 78, `rust` 46.

### 2.3 Class policy (per user decisions)
- **Keep only classes with >1000 instances** (post-merge). Others are **ignored** (filtered out of generated labels) —
  **never deleted** from source.
  - Excluded in v1: `vegetation`, `top_crossarm`, `om_crossarm`, `rust`.
- **Model 1 (pole):** `pole`
- **Model 2 (components):** `wire`, `h_insulator`, `v_insulator`, `crossarm_stright`
- **"Stable frequency" balancing (cap ON by default):** cap = lowest kept count in the set
  (components ≈ 1661). Over-represented classes are reduced toward the cap via **greedy image selection**
  (prefer images carrying under-represented kept classes; stop adding images that only feed already-capped classes).
  Configurable; disable with one flag. Caveat documented: capping discards real images.

---

## 3. Image preprocessing (CV analysis)

**Observed:** strong backlighting / blown-out sky with dark, silhouetted metal hardware (high dynamic range);
thin high-frequency structures (wires, insulator pins).

**Approach:**
- An automated **exposure-profiling pass** computes per-image brightness + highlight/shadow-clipping histograms
  over the dataset and reports stats (drives default params; surfaces outliers).
- **Default preprocessing: CLAHE on the luminance (L) channel** (LAB space) to recover detail in dark hardware,
  plus optional gamma from the profile. Resolution normalization handled at train time via letterbox.
- **Both image versions are stored** in each DB: `orig/` and `clahe/`, with matching configs, so CLAHE can be A/B tested.
  Bounding boxes are identical between versions (only pixels differ).
- Train large: `imgsz ≥ 1280` to preserve thin wires / small insulators.

---

## 4. Output datasets (two self-contained DBs on the SSD)

Each DB is **standalone** (full-res images included) so a single folder can be copied to the M4 for fast local I/O.

```
/Volumes/dronisight/
├── yolo_train_db/                         # YOLO format (primary)
│   ├── pole/
│   │   ├── images/{train,val,test}/{orig,clahe}/*.jpg
│   │   ├── labels/{train,val,test}/*.txt              # shared across orig/clahe
│   │   ├── data_orig.yaml
│   │   └── data_clahe.yaml
│   └── components/
│       ├── images/{train,val,test}/{orig,clahe}/*.jpg
│       ├── labels/{train,val,test}/*.txt
│       ├── data_orig.yaml
│       └── data_clahe.yaml
└── RF_DETR_Faster_RCNN_train_db/          # COCO format
    ├── pole/
    │   ├── images/{train,val,test}/{orig,clahe}/*.jpg
    │   └── annotations/instances_{train,val,test}_{orig,clahe}.json
    └── components/
        ├── images/{train,val,test}/{orig,clahe}/*.jpg
        └── annotations/instances_{train,val,test}_{orig,clahe}.json
```

- **Full 12MP** images preserved.
- **Split: grouped 80/15/5**, stratified across all 7 locations, **sequence-grouped** (consecutive drone frames kept
  together) to prevent near-duplicate leakage.
- Images with zero kept-class labels (for a given sub-dataset) are excluded from that sub-dataset.
- Source `mem*` folders are never modified.

---

## 5. Code architecture (`PycharmProjects/trainDronisight/`)

Modular; each unit has one purpose and a clear interface.

```
trainDronisight/
├── shared/
│   ├── device.py            # select_device(): CUDA → MPS → CPU (used by all training + inference)
│   ├── labels.py            # XML parsing, name normalization/merge, class policy, kept-class filter
│   └── config.py            # paths, class lists, cap target, split ratios, imgsz, etc.
├── data_prep/
│   ├── profile_images.py    # exposure/brightness/clipping histograms → report + preprocessing params
│   ├── preprocess.py        # CLAHE(L-channel)+gamma; emits orig + clahe versions
│   ├── build_dataset.py     # XML→clean labels→merge→filter→balance→grouped split→emit YOLO + COCO into both DBs
│   └── verify_dataset.py    # sanity checks: counts per class/split, leakage check, box validity, visual spot-render
├── train_yolo/
│   ├── train_pole.py        # YOLO26x (fallback flagged if v26 weights unavailable), pole
│   └── train_components.py  # YOLO26x, components
├── train_faster_rcnn/
│   └── train.py             # torchvision Faster R-CNN, COCO, --target {pole,components}
├── train_rf_detr/
│   └── train.py             # RF-DETR-L, COCO, --target {pole,components}  (CUDA-preferred warning)
├── inference/
│   ├── backends.py          # Detector abstraction: load YOLO / FRCNN / RF-DETR weights → unified predict()
│   ├── infer_pole.py        # Model 1 only, on a full image
│   ├── infer_components.py  # Model 2 only, on an image/crop
│   └── pipeline.py          # full chain: pole detect → crop pole box → component detect (on crop)
│                            #             → crop each component → structured output (boxes+classes+confidences, JSON)
├── notebooks/
│   ├── 00_data_prep.ipynb           # Colab: mount Drive, run data_prep
│   ├── 01_train_yolo.ipynb          # Colab CUDA
│   ├── 02_train_faster_rcnn.ipynb
│   ├── 03_train_rf_detr.ipynb
│   └── 04_inference_pipeline.ipynb
├── docs/superpowers/specs/...
├── pyproject.toml           # deps via uv
└── README.md
```

### Key interfaces
- `select_device() -> str` — returns `"cuda" | "mps" | "cpu"` by priority; respected by every train/infer entrypoint.
- `Detector.predict(image) -> list[Detection]` where `Detection = {box(xyxy), class_name, confidence}` —
  one abstraction over all three backends so `pipeline.py` is backend-agnostic (`--pole-backend`, `--comp-backend`).
- Structured pipeline output (JSON): per pole → pole box + score; per component → class, confidence, box (full-frame
  coords), and saved crop path. (Defect/condition fields intentionally absent — future classifier.)

---

## 6. M4 Pro utilization (sustainable, not "100% = throttling")
- `device="mps"`; dataset caching where it fits in 24 GB; tuned `batch`/`imgsz`; AMP; `workers` set for M4 cores.
- Keep plugged in, disable low-power mode. Goal = fastest *correct* training, not max GPU%.
- For RF-DETR (and heavy Faster R-CNN runs), prefer the Colab CUDA notebooks.

---

## 7. Dependencies (via `uv`, inside an activated `.venv`)
`ultralytics` (YOLO26), `torch`/`torchvision` (MPS + CUDA builds), `rfdetr`, `pycocotools`, `opencv-python`,
`numpy`, `pillow`, `lxml`, `pyyaml`, `tqdm`. (Exact pins resolved in the implementation plan.)

---

## 8. Success criteria (v1)
1. `data_prep` produces both DBs, fully self-contained, with **clean, merged, name-derived labels** (no index chaos,
   crossarm variants merged, excluded classes filtered), both `orig`+`clahe` versions, and a passing `verify_dataset` report.
2. Each model family trains to completion on its target(s) with the CUDA→MPS→CPU selector, YOLO26x running on M4 MPS.
3. `inference/pipeline.py` runs end-to-end on a sample image and emits structured JSON (pole → crop → components → crops).
4. Colab notebooks reproduce data-prep, all trainings, and the inference pipeline.

## 9. Open items deferred to the plan / later
- Exact dependency version pins and YOLO26 weight-name confirmation (with fallback handling).
- Whether to mine `mem1` for confirmed pole-free background negatives (precision boost) — future.
- Re-introducing rare classes (`rust`, etc.) once more annotations exist.
