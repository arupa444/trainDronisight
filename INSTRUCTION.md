# INSTRUCTION.md — trainDronisight, end to end (SSD in → trained pipeline out)

The complete runbook for **electric-pole inspection from DJI drone imagery**: plug in the SSD,
set up the repo, (optionally) rebuild the datasets, train the detectors, and run the four-stage
inference pipeline. Detection + condition v1.

> **Your input is the SSD.** Plug the `dronisight` SSD into the M4 (mounts at
> `/Volumes/dronisight`). It holds **both** the raw source folders **and** the prebuilt
> training DBs, so out of the box you can skip straight to training — no data build needed.
>
> **On Windows + NVIDIA CUDA?** Use [`windows_instruction.md`](windows_instruction.md) instead —
> same project, CUDA-specific setup, and all three families (incl. RF-DETR-L) train locally.

---

## 0. Mental model (read once)

- **Specialist pipeline — 12 models in 3 stages** (each component/condition split into focused
  specialists for accuracy + precise output):
  1. **`pole`** — full frame.
  2. **5 component detectors**, each on the pole crop (segregated by visual scale; the 3 crossarm
     types share one detector so they stop being mis-typed):
     `comp_wire` · `comp_insulator` (h_insulator, v_insulator) · `comp_crossarm` (crossarm_stright,
     top_crossarm, om_crossarm) · `comp_vegetation` · `comp_rust`.
  3. **6 condition specialists**, each on the matching component's crop:
     `cond_v_insulator` (normal/band/broken/chip_off) · `cond_h_insulator` (normal/broken/chip_off) ·
     `cond_straight_crossarm` (normal/band) · `cond_top_crossarm` (normal/band) ·
     `cond_om_crossarm` (normal/band) · `cond_wire` (wire_normal/cross_wire).
  Each detected component routes to its condition specialist via `config.COMPONENT_TO_CONDITION_MODEL`
  (vegetation/rust have no condition). All trained separately → chained at inference.
- **Three model families, pick the winner.** YOLO26 and Faster R-CNN train on the M4 (MPS);
  RF-DETR-L wants CUDA → Colab. The `Detection`/`Detector` backends are interchangeable, so any
  stage can run any family. Keep whichever wins on **val mAP@.5**.
- **Crop-aligned by construction.** Component detectors train on the **pole crop** and condition
  specialists on the **component crop** (`config.CROP_ALIGN`, pads `POLE_CROP_PAD` / `CONDITION_CROP_PAD`)
  so train scale == inference scale. `pole` trains on the full frame.
- **Two machines.** Code + the one-time data build happen on the dev laptop; **all training /
  inference runs on the M4 Pro (24 GB, Apple Silicon / MPS)**. Device auto-selects CUDA→MPS→CPU.
- **The data is already built.** You only rebuild if you change taxonomy / CLAHE / split / balance.

Conventions: commands assume you've `cd`-ed into the repo and **activated the venv**
(`source .venv/bin/activate`).

---

## 1. Quick start — copy-paste (clone → train → infer)

```bash
# 1. Tools (one-time)
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh           # restart shell after

# 2. Code
git clone https://github.com/arupa444/trainDronisight.git
cd trainDronisight

# 3. Python env + deps (torch-MPS, ultralytics, rfdetr, torchvision, opencv, …)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 4. Plug the SSD in -> default paths just work. Sanity:
python -c "from shared import config; print(config.YOLO_DB)"   # -> /Volumes/dronisight/yolo_train_db
pytest -q                                                       # full suite passes

# 5. Train YOLO — pole + 5 component + 6 condition specialists (12 models). On Colab use
#    colab_train_yolo.ipynb (A100). Locally (MPS), per §9; e.g. one component specialist:
python -m train_yolo.train_pole       --version clahe --epochs 100 --imgsz 640  --batch 4 --model yolo26x.pt
python -m train_yolo.train_components --subset comp_crossarm --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26x.pt
#    ... (comp_wire/insulator/vegetation/rust + the 6 cond_*; see §9)

# 6. Inference (the pipeline currently chains pole -> component -> condition; multi-specialist
#    routing across the 5 component + 6 condition models is being wired — see §13).
python -m inference.pipeline --image some.jpg \
  --pole-weights      runs/pole/yolo/weights/best.pt \
  --comp-above-weights runs/comp_crossarm/yolo/weights/best.pt \
  --comp-below-weights runs/comp_wire/yolo/weights/best.pt \
  --condition-weights runs/cond_v_insulator/yolo/weights/best.pt \
  --out-dir runs/inference
```

Everything below explains each step and the comparison models / crop ablation.

---

## 2. Prerequisites

1. **macOS** on the M4 Pro, plugged into power (disable Low Power Mode — the GPU throttles on battery).
2. **Xcode CLT:** `xcode-select --install`.
3. **uv** (this project uses `uv`, never bare `pip`): `curl -LsSf https://astral.sh/uv/install.sh | sh`, then restart the shell and confirm `uv --version`.
4. **git** (ships with the CLT).

## 3. Get the code

```bash
git clone https://github.com/arupa444/trainDronisight.git
cd trainDronisight
```

## 4. Python environment

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"   # expect True
pytest -q                                                                              # full suite passes
```
If `MPS available: False`, you're on an Intel Mac or an old torch — reinstall torch ≥ 2.2.

## 5. Point the code at the SSD

`shared/config.py` reads everything off the root in `DRONISIGHT_DATA` (default `/Volumes/dronisight`).

### Option A — run straight off the SSD (simplest)
Plug it in. The defaults resolve to `/Volumes/dronisight/{yolo_train_db, RF_DETR_Faster_RCNN_train_db}`
and the raw source folders. **Nothing to set.** (Training I/O is slower over USB, but it works.)

### Option B — copy the DBs to local NVMe (faster training, avoids exFAT hiccups — recommended)
```bash
mkdir -p ~/dronisight_data
rsync -a --exclude '._*' --exclude '.DS_Store' \
  /Volumes/dronisight/yolo_train_db \
  /Volumes/dronisight/RF_DETR_Faster_RCNN_train_db \
  ~/dronisight_data/
export DRONISIGHT_DATA=~/dronisight_data          # put in ~/.zshrc; verify with the print below
python -c "from shared import config; print(config.YOLO_DB)"
```
> ⚠️ The SSD is **exFAT** → macOS scatters `._*` AppleDouble sidecars. The code filters them and
> the build self-cleans them, but `--exclude '._*'` on copy keeps counts honest. **Don't unplug or
> let the SSD sleep during a build** — a mid-write disconnect corrupts the run.

## 6. (Optional) Rebuild the datasets from raw annotations

Only if you change class policy / CLAHE / split / balance / crop logic. The DBs are otherwise ready.

```bash
python -m data_prep.build_dataset --subset all          # all 12 subsets into both DBs (pole + 5 comp + 6 cond)
# or just the condition specialists:  --subset all_cond ;  or a single subset name

for s in pole comp_wire comp_insulator comp_crossarm comp_vegetation comp_rust \
         cond_v_insulator cond_h_insulator cond_straight_crossarm cond_top_crossarm cond_om_crossarm cond_wire; do
  python -m data_prep.verify_dataset --subset "$s"      # group + image-content leakage + label validity
done
```

What the build does (the non-obvious parts):
- **VOC XML is the label source of truth** (`shared/labels.py`); index-based YOLO `.txt` in the raw
  data is ignored. All names are normalized to canonical classes; unknown → dropped.
- **Per-annotator merge.** The 6th-june condition data was labeled by 8–9 members who each annotated
  *different* classes over the *same* photo pool. `build_dataset` content-hash-**merges** every
  byte-identical copy into one entry holding the **union** of all members' boxes — preventing
  cross-split leakage and partial-label poisoning. No-op on the disjoint mem captures.
- **Condition-conflict resolution** (condition subset): when two members gave one object different
  conditions, **defect beats normal**; a defect-vs-defect disagreement is **dropped** as ambiguous.
- **Balance per subset** (TRAIN only; val/test stay raw): `above` caps toward the rarest class
  (`v_insulator`); `below` oversamples rare classes up with bbox-aware augmentation; `classification`
  targets **400/class** (cap big classes down, augment small ones up).
- **orig + clahe variants.** Every image is stored twice — untouched and with adaptive CLAHE
  (LAB L-channel, per-image clip from an exposure profile). Train both, keep the val-mAP winner.
- **Leakage-safe splits.** Frames are grouped into capture sequences (>60 s gap = new group) and
  split by group, stratified per source. `verify_dataset` asserts no group spans splits **and**
  re-hashes the written DB to assert no physical photo appears in >1 split.
- **Crop-aligned variants** (`*_crop`): see §8.

## 7. The datasets (what you train on)

Two parallel DBs, same splits: `yolo_train_db/` (YOLO `.txt`, primary) and
`RF_DETR_Faster_RCNN_train_db/` (COCO JSON, for Faster R-CNN + RF-DETR). 12 subsets:

| Subset | classes | scale | balance |
|---|---|---|---|
| `pole` | pole | full frame | n/a |
| `comp_wire` | wire | pole crop | n/a (1 class) |
| `comp_insulator` | h_insulator, v_insulator | pole crop | cap→rarest |
| `comp_crossarm` | crossarm_stright, top_crossarm, om_crossarm | pole crop | target-1000 |
| `comp_vegetation` | vegetation | pole crop | n/a (1 class) |
| `comp_rust` | rust | pole crop | n/a (1 class, data-limited) |
| `cond_v_insulator` | normal, band, broken, chip_off | component crop | target-400 |
| `cond_h_insulator` | normal, broken, chip_off | component crop | target-400 |
| `cond_straight_crossarm` | normal, band | component crop | target-400 |
| `cond_top_crossarm` | normal, band | component crop | target-400 |
| `cond_om_crossarm` | normal, band | component crop | target-400 |
| `cond_wire` | wire_normal, cross_wire | component crop | target-400 |

YOLO weights land at `runs/<subset>/yolo/weights/best.pt`. Build: `--subset all` (all 12),
`all_cond` (the 6 condition), or a single name.

## 8. Preprocessing (already correct — what it does)

- **CLAHE** on the **LAB L-channel only** (chroma untouched → no color shift), adaptive clip per
  image from the exposure profile. Verified **byte-exact** between the stored `clahe` training
  variant and what inference produces from the raw frame.
- **EXIF orientation** applied at build and inference (`load_oriented_bgr`); stored dims match the XML.
- **Crop-aligned (`*_crop`) subsets** close the train/serve **scale gap**: the component/condition
  detectors run on crops at inference but were trained on full ~4000×3000 frames. `*_crop` builds
  crop each frame to the **pole** box (above/below) or the **component** box (condition) and remap
  the boxes, so train scale == inference scale. CLAHE is applied to the **full frame then sliced**
  (identical to inference). Crops of one photo all share its capture group → never split across
  train/val/test. **Train both full-frame and crop, keep the val-mAP winner** (helps thin wires /
  small insulators most).

## 9. Train — YOLO (M4 / MPS, or Colab via `colab_train_yolo.ipynb`)

The authoritative training path is **`colab_train_yolo.ipynb`** (A100, auto-batch, saves to Drive
per model). Equivalent commands:
```bash
# pole (full frame, 640)
python -m train_yolo.train_pole --version clahe --epochs 100 --imgsz 640 --batch 4 --model yolo26x.pt

# 5 component specialists (pole crop, 1280). Data-rich -> yolo26x; small -> yolo26m.
python -m train_yolo.train_components --subset comp_wire        --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26x.pt
python -m train_yolo.train_components --subset comp_insulator   --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26x.pt
python -m train_yolo.train_components --subset comp_crossarm    --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26x.pt
python -m train_yolo.train_components --subset comp_vegetation  --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26m.pt
python -m train_yolo.train_components --subset comp_rust        --version clahe --epochs 200 --imgsz 1280 --batch 4 --model yolo26m.pt

# 6 condition specialists (component crop, 1280) -> yolo26m (small focused families)
for s in cond_v_insulator cond_h_insulator cond_straight_crossarm cond_top_crossarm cond_om_crossarm cond_wire; do
  python -m train_yolo.train_components --subset $s --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26m.pt
done
```
- **`imgsz 640` for pole** (fills the frame); **`imgsz 1280`** for all crops (thin wires / subtle defects).
- **Model choice:** data-rich detectors (`comp_wire/insulator/crossarm`) → **yolo26x**; small/data-limited
  (`comp_vegetation`, `comp_rust`, every `cond_*`) → **yolo26m** (resists overfit on ~200–1.5k instances).
- **OOM on 24 GB?** `yolo26x@1280` → drop `--batch` to 2 or use `yolo26l/m`.
- **`yolo26x → yolo11x` fallback** is a printed warning (YOLO26 weights not fetchable on your
  Ultralytics), **not an error**.
- Train **both `--version clahe` and `--version orig`**; compare on val mAP. Watch the train/val gap
  in `runs/<subset>/yolo/results.png`; report **per-class AP**, not just mAP.

## 10. Train — Faster R-CNN (M4 / MPS, comparison)

Same `--subset` names as YOLO (any of the 12). Examples:
```bash
python -m train_faster_rcnn.train --subset pole          --version clahe --epochs 30 --batch 2
python -m train_faster_rcnn.train --subset comp_crossarm --version clahe --epochs 30 --batch 2
# ... repeat for each comp_* / cond_* you want to compare ...
# per-class AP (use the SAME --min-size you trained with):
python -m train_faster_rcnn.eval  --subset comp_crossarm --version clahe --split test
```
- **`--min-size` defaults to 2000** (torchvision's default 800 shrinks thin wires away). `best.pt`
  (lowest val loss, `--patience 7` early stop) is the checkpoint to use. Output:
  `runs/<subset>/faster_rcnn/{best,last}.pt`. First run downloads COCO-pretrained weights.

## 11. Train — RF-DETR-L (Colab / CUDA, comparison)

Impractical on MPS — use Colab/CUDA. Same `--subset` names (any of the 12):
```bash
python -m train_rf_detr.train --subset pole          --version clahe --epochs 50 --batch 4 --resolution 672
python -m train_rf_detr.train --subset comp_crossarm --version clahe --epochs 50 --batch 4 --resolution 672
# ... repeat per subset; raise --resolution (e.g. 1120) for the small-object specialists ...
```
- `--resolution` must be a multiple of the model **block_size** (`patch_size*num_windows` = **32** on
  the current RF-DETR build). **672 / 896 / 1120** are multiples of *both* 32 and 56, so they're safe
  across versions and need no inference-shape rounding. Output:
  `runs/<subset>/rfdetr/checkpoint_best_ema.pth`.

## 12. Choosing the final per-stage model

For **each stage**, compare across the axes and keep one checkpoint:
1. **Family:** YOLO vs Faster R-CNN vs RF-DETR (val mAP@.5).
2. **Variant:** `clahe` vs `orig`.
3. **Scale:** full-frame vs `_crop`.
Report **per-class AP** (a good mAP hides thin wires / rare conditions). Tune the confidence
threshold on **val**, freeze it, touch **test** once. Rare classes like `rust` have tiny test sets
(n≈3) — judge them on val and on recall.

## 13. Inference

> **Routing update in progress.** With the new architecture there are **5 component detectors + 6
> condition specialists**. `inference/pipeline.py` currently chains 4 detectors (pole + 2 component
> slots + 1 condition) — the multi-specialist routing (run all 5 component detectors → NMS → route
> each detection to its condition model via `config.COMPONENT_TO_CONDITION_MODEL`) is being wired and
> will land once the new models finish training. The single-model CLIs below work today for any subset.

### Single stage (debugging; YOLO CLIs)
`--image` takes a file **or a directory**. `--out-csv` writes a flat per-detection CSV (`image,class,confidence,x1,y1,x2,y2`); `--out` writes JSON.
```bash
python -m inference.infer_pole       --image some.jpg --weights runs/pole/yolo/weights/best.pt --out-csv pole.csv
python -m inference.infer_components  --image crop.jpg --weights runs/component_above_1000/yolo/weights/best.pt --out-csv comp.csv
```

### Full four-stage pipeline (with CSV)
`--image` is a **file or a directory** (batches the whole folder into one CSV). Each run is saved to its **own auto-named folder** under `--out-dir` so nothing is overwritten:
`<out-dir>/<input>_inference[N]/` — `<input>` is the file stem or directory name, with `2/3/…` appended if it already exists. Inside: `result.json`, `result.csv`, `crops/`, and (unless `--no-viz`) `viz/{pole,components,conditions,all}/`. Every saved image is named `<img-stem>_inference.jpg`.
```bash
python -m inference.pipeline \
  --image "some.jpg_or_folder/" \
  --pole-weights        runs/pole/yolo/weights/best.pt \
  --comp-above-weights  runs/component_above_1000/yolo/weights/best.pt \
  --comp-below-weights  runs/component_below_1000/yolo/weights/best.pt \
  --condition-weights   runs/component_classification/yolo/weights/best.pt \
  --out-dir runs/inference          # -> runs/inference/some_inference/{result.csv,result.json,crops/,viz/}
```
(Override individual paths with `--out/--out-csv/--crop-dir/--viz-dir` if you need a specific location; `--no-viz` skips the annotated images.)
- **One box per object (cross-detector NMS):** the above & below detectors both run on the pole crop, so the same physical object can be detected twice (often as different classes). Class-agnostic NMS (`--nms-iou`, default 0.55; `--no-nms` to disable) keeps the single highest-confidence box per object. A genuinely co-located distinct object (e.g. a wire crossing a crossarm) stays because its IoU is lower.
- **Condition is merged into the component:** there is ONE box per component, labeled with its class **and** condition inline (e.g. `v_insulator | v_insulator_band 0.42`). The `viz/conditions/` layer relabels that same box with just the condition; there's no separate near-duplicate condition box.
- **Condition crop padding (train==serve):** the condition model is fed each component crop padded by `config.CONDITION_CROP_PAD` (default 0.25) — the SAME pad used to build `component_classification_crop` (its `self`-mode pad). Keep them equal: if you rebuild the condition dataset with a different pad, pass the matching `--condition-pad` at inference. Padding gives subtle insulator defects their surrounding context and tolerates loose detector boxes.
- **Condition mapping:** the condition model runs on each component crop, but its output is **filtered to that component's family** (`config.COMPONENT_TO_CONDITIONS`) — a `v_insulator` crop can only get a `v_insulator_*` condition, never a crossarm/wire one; `vegetation`/`rust` have no condition family (left blank). The top in-family condition is `condition`; all in-family ones are `conditions`.
- **Too many weak boxes?** raise `--comp-conf` (e.g. 0.35–0.4) to drop low-confidence detections, and/or lower `--nms-iou` for more aggressive de-dup.
- **`result.csv` columns:** `image, pole_index, pole_confidence, pole_x1..y2, group(above/below), component_class, component_confidence, comp_x1..y2, condition_class, condition_confidence, crop_path`.
- **Annotated views:** add `--viz-dir runs/inference/viz` to also write 4 boxed images per frame:
  `viz/pole/` (pole only), `viz/components/` (above+below boxes), `viz/conditions/` (per-component condition boxes), `viz/all/` (everything overlaid). Drawn on the EXIF-oriented frame; condition boxes are remapped from the component crop to the full frame.
- **Preprocessing matches training:** EXIF-orient + CLAHE applied **once** on the full frame; every
  crop inherits it. Pass `--no-clahe` only for `orig`-trained weights.
- **Stage 4 is optional:** omit `--condition-weights` to stop at component detection.
- **Confidence defaults:** `--pole-conf 0.12` (recall-leaning), `--comp-conf 0.25`,
  `--condition-conf 0.25`. **imgsz:** `--pole-imgsz 640`, `--comp-imgsz 1280`, `--condition-imgsz 1280`.
  `--pole-pad 0.05` keeps edge components from being clipped.
- **Mixed backends:** every stage takes `--pole-backend / --comp-above-backend /
  --comp-below-backend / --condition-backend {yolo,rfdetr,frcnn}`; point the matching `*-weights` at
  that family's checkpoint. RF-DETR stages share `--rfdetr-resolution` (672/1120); FRCNN stages share
  `--frcnn-min-size` (default 2000 — **must match training** or small objects collapse). Example:
  ```bash
  python -m inference.pipeline --image some.jpg \
    --pole-backend yolo  --pole-weights runs/pole/yolo/weights/best.pt \
    --comp-above-backend rfdetr --comp-above-weights runs/component_above_1000/rfdetr/checkpoint_best_ema.pth \
    --comp-below-backend rfdetr --comp-below-weights runs/component_below_1000/rfdetr/checkpoint_best_ema.pth \
    --rfdetr-resolution 672 --out runs/inference/result.json
  ```
- **Output JSON** (`conditions` present only with `--condition-weights`):
  ```json
  { "image": "some.jpg", "poles": [ {
      "box": [x1,y1,x2,y2], "confidence": 0.97, "crop_path": "…/some_pole0.jpg",
      "components_above": [ { "class": "v_insulator", "confidence": 0.88,
          "box_full": [..], "box_crop": [..], "crop_path": "…_above_comp0.jpg",
          "condition": { "class": "v_insulator_broken", "confidence": 0.71 },
          "conditions": [ { "class": "v_insulator_broken", "confidence": 0.71, "box_comp": [..] } ] } ],
      "components_below": [ { "class": "vegetation", "confidence": 0.41, "box_full": [..], "box_crop": [..], "crop_path": "…_below_comp0.jpg" } ]
  } ] }
  ```
  `box_full` = original-frame coords; `box_crop` = within the pole crop; `box_comp` = within the component crop.

## 14. Get the most out of the M4 (sustainably)

- Plugged in, Low Power Mode **off**. Watch GPU: Activity Monitor → Window → GPU History, or
  `sudo powermetrics --samplers gpu_power -i1000`.
- **Batch is the main lever** — push `--batch` until ~80–85 % memory, then back off one step. Bigger
  batch = better MPS utilization. Lower `--imgsz` or use a smaller `--model` if you OOM.
- Don't chase a literal 100 % GPU — a pegged laptop GPU just thermal-throttles. Goal = fastest
  *converging* run, kept cool and plugged in. Close other heavy apps (MPS shares the 24 GB).

## 15. Colab path (CUDA — required for RF-DETR, faster for all)

Notebooks are **generated** from `notebooks/build_notebooks.py` (`REPO_URL` already set) — edit the
spec, never the `.ipynb`, then `python -m notebooks.build_notebooks`. Full walkthrough:
[`colab_instruction.md`](colab_instruction.md).
1. Zip the two DBs to `MyDrive/dronisight/` (exclude `._*`).
2. Open a notebook → **Runtime → Change runtime type → GPU** → **Run all**:
   `00_data_prep` (verify), `01_train_yolo` (incl. the optional `_crop` ablation cells),
   `02_train_faster_rcnn`, `03_train_rf_detr` (`--resolution` 672/1120), `04_inference_pipeline`.
   Each trainer copies `runs/` back to Drive (Colab runtimes are ephemeral); `04` restores it.

## 16. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Scan says **`0 images, N backgrounds`** | YOLO can't find labels — they must mirror the variant dir (`labels/<split>/<orig\|clahe>/`). Fixed in current code; after any fix `find <DB> -name '*.cache' -delete` (a poisoned cache reports 0 even after fixing). |
| Training reads a **different path** than your data | `data.yaml` had a build-machine absolute `path:`; the trainer regenerates it each run. `git pull`, delete `*.cache`, re-run. |
| `MPS backend out of memory` | Lower `--batch` (4→2), lower `--imgsz`, or `--model yolo26m.pt`. Close apps; stay plugged in. |
| Build produced **2× / duplicated images**, or counts look wrong | The SSD disconnected mid-build (check `mount \| grep dronisight` device id changes). Re-mount, **don't unplug**, clear the subset dir and rebuild; `verify_dataset` will confirm no leakage. |
| Errors reading `._*.jpg` / weird counts | exFAT AppleDouble sidecars — `rsync --exclude '._*'` or `find <DB> -name '._*' -delete`. The build self-cleans them. |
| First Faster R-CNN run stalls | Downloading COCO-pretrained weights (~160 MB). Cached after. |
| RF-DETR `resolution ... not divisible` | Use a multiple of 32 (672/896/1120). |
| `FileNotFoundError` on a DB path | `DRONISIGHT_DATA` unset/wrong, or SSD not mounted. `echo $DRONISIGHT_DATA`, re-check §5. |

## 17. What's NOT here yet

The condition classifier **exists** (`component_classification`, the optional 4th pipeline stage).
Still future work:
- **Per-pole scoring** — aggregate the per-component conditions into a pole health score.
- **Report + OpenStreetMap UI** — UUID (from the drone EXIF GPS) → map pin → per-pole detail page.
