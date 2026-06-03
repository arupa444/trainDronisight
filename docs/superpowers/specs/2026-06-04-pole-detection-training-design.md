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
  (components ≈ 1661). Over-represented classes are reduced toward the cap via **greedy image selection** that
  **maximizes retained instances of the scarcest classes first** (sort images by rarity-weighted contribution;
  admit until each class reaches the cap; never drop an image while it still feeds an under-cap class).
  Configurable; disable with one flag.
  - **Data-science caveat (documented in-code):** instance-capping by dropping images discards real, informative
    backgrounds and co-occurring objects, and cannot perfectly equalize because classes co-occur in the same image.
    The balanced set is therefore the **default training set**, but data-prep *also* emits the **full uncapped set**
    and a **per-image inverse-frequency sampling weight** manifest, so the alternative (train on all data + weighted
    sampling / loss) can be run without re-prepping. Both are evaluated; the better val mAP wins.

---

## 3. Image preprocessing (CV-scientist methodology)

### 3.1 Diagnosis (what the imagery actually is)
- **High-dynamic-range backlighting:** bright/over-exposed sky against dark, silhouetted metal hardware. The
  detector's signal (insulator sheds, crossarm edges, wire crimps) sits in the **crushed shadow** region.
- **Thin, high-frequency targets:** wires and insulator pins are 1–3 px wide at native scale and are the first
  thing destroyed by downscaling, blur, or aggressive denoise.
- **JPEG-compressed 12 MP:** 8×8 DCT blocking + chroma subsampling artifacts, worst in flat sky regions —
  any contrast stretch will **amplify these artifacts** if applied blindly.
- **Mixed sensors** (`4032×3024`, `4096×3072`) and **DJI EXIF orientation** tags that must be honored before
  reading boxes (otherwise labels misalign).

### 3.2 Principle: enhance for recoverability, not aesthetics
The pretrained backbones (COCO/ImageNet stats) expect roughly natural image statistics. Heavy global enhancement
shifts the input distribution away from what the backbone learned and **can lower mAP even when images "look"
better to a human.** Therefore preprocessing is **mild, local, and empirically validated** — and we always keep
the untouched `orig/` version as the control. The enhancement track is treated as a **hypothesis to be tested**
(per-class AP, `clahe` vs `orig`), not a foregone conclusion.

### 3.3 Profiling pass (drives everything, per-image)
`profile_images.py` computes and persists, per image:
- Luminance (LAB-L) **histogram**, mean/median, std (global contrast).
- **Highlight-clip fraction** (pixels ≥ 250) and **shadow-clip fraction** (pixels ≤ 5).
- **Backlit score** = highlight-clip% combined with low shadow-region contrast (identifies the silhouette images).
- Estimated **haze/veiling-light** (dark-channel prior statistic).
- Sharpness proxy (variance of Laplacian) to flag motion-blurred frames.
Output: a dataset-level report + per-image params, plus a flag list (severely backlit / hazy / blurry outliers).

### 3.4 The `clahe` enhancement track (adaptive, not blanket)
Applied in **LAB**, on the **L channel only** (chroma untouched → no color casts), then back to RGB:
1. **Adaptive CLAHE** — `clipLimit` and `tileGridSize` chosen *per image from the profile* (stronger only on
   genuinely backlit frames; near-identity on already well-exposed frames like the building shot). Defaults to
   `clipLimit≈2.0`, `tileGridSize=8×8`; capped to avoid sky-noise blow-up.
2. **Mild gamma** (<1 to lift shadows) only when shadow-clip% is high.
3. **Optional dark-channel dehaze** only for frames the profile flags as hazy.
4. **No global histogram equalization, no denoise, no strong sharpening** near native scale — these erase wires.
   A *light* unsharp mask is allowed only if the profile shows the frame is soft.
All steps are **parameterized and logged per image** so the dataset is reproducible.

### 3.5 What is NOT baked in (left to the model's own pipeline)
- **Resize/letterbox** (aspect-preserving, pad value 114) and **per-channel normalization** are done by the
  training framework — never pre-baked, to avoid double-normalization and to keep `imgsz` a free hyperparameter.
- Train large: **`imgsz ≥ 1280`** (ablate 1280 vs 1536) to keep thin wires alive.

### 3.6 Storage
Both versions stored per DB (`orig/`, `clahe/`) with identical boxes; the `clahe` params manifest is saved so any
image can be regenerated. **Inference must apply the exact same profiling+CLAHE transform** when the `clahe` model
is used — the transform is shared code (`data_prep/preprocess.py`), not duplicated.

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

## 6. Training methodology (ML-scientist)

### 6.1 The two-stage train/inference domain gap (critical)
Model 2 is **trained on full frames** but **runs on cropped pole regions** at inference → a scale/context shift
that silently tanks recall if ignored. Mitigation baked into Model-2 training:
- **Scale-jitter / random-resized-crop augmentation** that simulates the zoomed-in cropped distribution
  (large `scale` range; crops centered on annotated objects), so the model sees both full-frame and crop-like scales.
- **Multi-scale training** (vary `imgsz`) for the same reason.
- **Pipeline-level eval (§6.5)** measures the real end-to-end number, not just the per-model full-frame mAP.

### 6.2 Transfer learning & schedule
- Start from **COCO-pretrained** weights (all three families). Optional brief **backbone freeze + warmup**, then
  full fine-tune. **Cosine LR** with linear warmup; **early stopping** on val mAP (`patience`).
- **AMP** on; **fixed seeds** + logged configs for reproducibility; deterministic ops where the backend allows.
- Batch size set **manually per backend** (MPS auto-batch is unreliable); start conservative for 24 GB and scale up.

### 6.3 Domain-appropriate augmentation (and what to avoid)
- **Use:** HSV jitter (outdoor lighting variance), horizontal flip, scale/translate, mild rotation (±10°),
  mosaic for small-object density, **copy-paste** to help the scarcer kept classes; `close_mosaic` for the final
  epochs so the model finishes on realistic full images.
- **Avoid / limit:** vertical flip and large rotations (poles/insulators have a strong up-down orientation prior),
  heavy blur/noise (kills thin wires), extreme color distortion (rust/metal cues are color-dependent).

### 6.4 Imbalance handling at train time (in addition to §2.3 capping)
- Even with capping, residual skew exists → enable **inverse-frequency / focal-style loss weighting** where the
  framework supports it, and report **per-class AP** every eval (never hide behind a single mAP).

### 6.5 Evaluation protocol (data integrity first)
- **Grouped split (§4)** prevents near-duplicate leakage; `verify_dataset` asserts **no capture-group spans two splits**.
- Primary metrics: **mAP@0.5** and **mAP@0.5:0.95**, plus **per-class AP**, PR curves, confusion matrix.
- **Test set touched once**, at the end. Confidence threshold for the pipeline is **tuned on val**, then frozen.
- **End-to-end pipeline metric:** evaluate `pole→crop→components` jointly (does cropping help/hurt component recall
  vs running Model 2 on the full frame?) — this decides whether the two-stage crop is actually worth it.
- Track runs (Ultralytics `runs/` + a CSV manifest; W&B optional). Each run records the **dataset version hash**
  (so results are tied to an exact, regenerable dataset state).

### 6.6 M4 Pro utilization (sustainable, not "100% = throttling")
- `device="mps"`; cache dataset to RAM where it fits in 24 GB (else `cache="disk"`); tuned `batch`/`imgsz`; AMP;
  `workers` matched to M4 cores. Keep plugged in, disable low-power mode. Goal = fastest *correct* training, not max GPU%.
- **RF-DETR-L and heavy Faster R-CNN runs → use the Colab CUDA notebooks** (DETR training is impractical on MPS).
  YOLO26x is the one that genuinely belongs on the M4.

---

## 7. Dependencies (via `uv`, inside an activated `.venv`)
`ultralytics` (YOLO26), `torch`/`torchvision` (MPS + CUDA builds), `rfdetr`, `pycocotools`, `opencv-python`,
`albumentations` (scale-jitter/copy-paste-style aug for the FRCNN/RF-DETR pipelines), `scikit-learn` (grouped/
stratified split), `pandas` (image-profile + dataset-version manifests), `numpy`, `pillow`, `lxml`, `pyyaml`,
`tqdm`. (Exact pins resolved in the implementation plan.) Installed via `uv pip install` inside an activated `.venv`.

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
