# INSTRUCTION.md — Running trainDronisight on the Mac M4 Pro

A deep, do-this-then-that guide for taking this repo + the prepared data and **training, evaluating, and
running inference on the M4 Pro (24 GB, Apple Silicon / MPS)**.

Conventions: shell commands assume you've `cd`-ed into the repo and **activated the venv**
(`source .venv/bin/activate`) unless stated otherwise.

---

## 0. Mental model (read once)

- **Two machines.** Code + the one-time data build happen on the dev laptop. **All training/inference runs here on the M4 Pro.**
- **Four subsets, four-stage pipeline.** Model 1 finds the pole; then **two** component detectors run on the cropped pole region — `component_above_1000` (the 4 frequent classes) and `component_below_1000` (the 4 rare classes); finally each detected component crop is fed to `component_classification` (the **14 condition classes**, e.g. `v_insulator_broken`). All four are **separate trainings** producing separate weights, chained only at inference (the condition stage is the optional 4th step, see Section 11).
- **Per-annotator 6th-june data, content-hash merged.** The condition captures were labeled by 8–9 members who each annotated *different* classes over the *same* photo pool, so one physical photo lives in several member folders with only partial labels. `build_dataset` content-hash-MERGES every byte-identical copy into one entry holding the **union** of all members' boxes (and, for the condition subset, resolves objects two members labeled with conflicting conditions: defect beats normal, defect-vs-defect dropped) — preventing cross-split leakage and partial-label poisoning.
- **No CUDA on Apple Silicon.** YOLO and Faster R-CNN train fine on **MPS**. RF-DETR-L really wants CUDA → run that one on **Colab** (Section 10). The code auto-selects `CUDA → MPS → CPU`, so the same commands work everywhere.
- **The data is already built.** You do **not** need to re-run data-prep unless you want to change the taxonomy/preprocessing. You just point the code at the DB folders.

---

## TL;DR — full run, copy-paste (clone → train → infer)

The complete happy path on the M4 with the SSD plugged in at `/Volumes/dronisight`. Each step is explained in the numbered sections below; this is the at-a-glance sequence.

```bash
# 1. Tools (one-time): Xcode CLT + uv
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh        # restart shell after

# 2. Code
git clone https://github.com/arupa444/trainDronisight.git
cd trainDronisight

# 3. Python env + deps (torch-MPS, ultralytics, rfdetr, torchvision, opencv, …)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 4. Point at the data. Plug the SSD in -> default path just works. (Or copy DBs local: see Section 3.)
python -c "from shared import config; print(config.YOLO_DB)"   # sanity: prints the DB path
pytest -q                                                       # sanity: full suite passes

# 5. (Optional) rebuild the DBs from raw annotations — ONLY if you changed taxonomy/CLAHE/split.
#    Otherwise skip: the 4 subsets are already built. (Do NOT unplug the SSD mid-build.)
python -m data_prep.build_dataset --subset all
for s in pole component_above_1000 component_below_1000 component_classification; do
  python -m data_prep.verify_dataset --subset "$s"; done

# 6. Train all 4 YOLO detectors (the primary models; run on MPS)
python -m train_yolo.train_pole       --version clahe --epochs 100 --imgsz 640  --batch 4 --model yolo26x.pt
python -m train_yolo.train_components --subset component_above_1000    --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26x.pt
python -m train_yolo.train_components --subset component_below_1000    --version clahe --epochs 200 --imgsz 1280 --batch 4 --model yolo26x.pt
python -m train_yolo.train_components --subset component_classification --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26x.pt
#    (yolo26x at 1280 may OOM on 24 GB -> drop --batch to 2 or use --model yolo26l.pt / yolo26m.pt)
#    Comparison families: Faster R-CNN (Section 9, MPS) and RF-DETR-L (Section 10, Colab).

# 7. Full 4-stage inference (pole -> above+below on the crop -> condition on each component crop)
python -m inference.pipeline --image some.jpg \
  --pole-weights        runs/pole/yolo/weights/best.pt \
  --comp-above-weights  runs/component_above_1000/yolo/weights/best.pt \
  --comp-below-weights  runs/component_below_1000/yolo/weights/best.pt \
  --condition-weights   runs/component_classification/yolo/weights/best.pt \
  --out runs/inference/result.json
```

---

## 1. Prerequisites

1. **macOS** on the M4 Pro, plugged into power.
2. **Xcode command-line tools:** `xcode-select --install` (for git, compilers).
3. **uv** (Python manager — this project uses `uv`, never bare `pip`):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Restart the shell, confirm: `uv --version`.
4. **git** (comes with the CLT above).

---

## 2. Get the code

```bash
git clone https://github.com/arupa444/trainDronisight.git
cd trainDronisight
```
(Private repo — authenticate with `gh auth login` or a GitHub token if prompted.)

---

## 3. Get the data onto the M4

The two DB folders (`yolo_train_db`, `RF_DETR_Faster_RCNN_train_db`) were built on the SSD. Pick **one**:

### Option A — plug the SSD into the M4 (simplest)
If the external `dronisight` SSD is mounted at `/Volumes/dronisight`, the default paths just work. **Skip the `export` below.**

### Option B — copy the DBs to fast local storage (recommended for training speed)
Local SSD I/O beats the external drive. Copy **only the two DBs** and **exclude macOS AppleDouble junk**:
```bash
mkdir -p ~/dronisight_data
rsync -a --exclude '._*' --exclude '.DS_Store' \
  /Volumes/dronisight/yolo_train_db \
  /Volumes/dronisight/RF_DETR_Faster_RCNN_train_db \
  ~/dronisight_data/
```
Then tell the code where the data lives (this overrides the default `/Volumes/dronisight`):
```bash
export DRONISIGHT_DATA=~/dronisight_data
```
> Put that `export` in your `~/.zshrc` so every shell sees it. Verify:
> ```bash
> python -c "from shared import config; print(config.YOLO_DB)"
> # -> /Users/you/dronisight_data/yolo_train_db
> ```

⚠️ **The `--exclude '._*'` matters.** The SSD is exFAT, so macOS scatters `._*` sidecar files. The code filters them, but excluding them on copy keeps the dataset clean and your image counts honest.

---

## 4. Python environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```
This installs torch (MPS build), torchvision, ultralytics, rfdetr, opencv, etc.

**Sanity checks:**
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"   # expect True
pytest -q                                                                              # expect the full suite to pass
```

If `MPS available: False`, you're likely on an Intel Mac or an old torch — reinstall torch ≥ 2.2.

---

## 5. (Optional) Rebuild the data from raw annotations

**Only if** you changed the class policy, CLAHE, or split logic. Requires the raw source folders present under `DRONISIGHT_DATA` — the **11 mem-captures** (`mem2 5th june, mem3…mem8, mem 7.1 5th june, mem10, 4thJuneMem4, 4thJuneMem8`) feed pole/components, and the **8 `6thMem*AllTeam1` folders** under `6th june ` feed `component_classification` (the exact lists are `config._SOURCE_FOLDER_NAMES` + `config._CONDITION_FOLDER_NAMES`). Otherwise skip — the DBs are ready.

```bash
python -m data_prep.build_dataset --subset all          # builds all 4 subsets into both DBs
python -m data_prep.verify_dataset --subset pole        # group + image-content leakage + label-validity gate
python -m data_prep.verify_dataset --subset component_above_1000
python -m data_prep.verify_dataset --subset component_below_1000
python -m data_prep.verify_dataset --subset component_classification
```
Useful flags: `--no-balance` (keep all images, skip the class cap — pair with `sample_weights.csv` at train time).
Each run self-cleans AppleDouble sidecars and prints split sizes + any skipped (EXIF-mismatch / unparseable) files.

**Builds 4 subsets** (`pole`, `component_above_1000`, `component_below_1000`, `component_classification`). Key data-prep behavior:
- **Content-hash cross-folder merge** (`data_prep/merge_annotations.py`, `config.MERGE_CROSS_FOLDER`): every byte-identical photo copy collapses into one entry with the **union** of all members' boxes, *before* grouping/splitting. This fixes the per-annotator 6th-june data (same photo, partial labels, in many folders) and subsumes the old `mem7`/`mem7.1` dedup; it's a no-op on disjoint captures. Measured on the condition build: 3641 copies → 2301 unique images, +2469 boxes recovered.
- **Condition-conflict resolution** (`config.RESOLVE_CONDITION_CONFLICTS`, `component_classification` only): when members gave one physical object different condition labels, **defect beats normal** and a **defect-vs-defect** disagreement is **dropped** as ambiguous.
- **Balancing differs per subset:** `component_above_1000` is balance-capped toward the rarest class (`v_insulator`); `component_below_1000` keeps all images and **oversamples** its train split with augmentation; `component_classification` uses a **fixed per-class target** (`config.BALANCE_TARGET = 400`): on **train** each class is capped DOWN to 400 and under-target classes are augmented UP toward 400, while **val/test stay raw**.
- **`verify_dataset` asserts BOTH** no capture-group leakage **and** no image-**content** leakage (it re-hashes the written DB and fails if any photo appears in >1 split), then prints the unique-image count and validates every YOLO label.

---

## 6. Train Model 1 — pole detector (YOLO26x, MPS)

```bash
python -m train_yolo.train_pole --version clahe --epochs 100 --imgsz 640 --batch 4
```
- `--version {orig,clahe}` — which image variant to train on (train both, compare).
- `--model` — preferred weights (default `yolo26x.pt`). Use `--model yolo26m.pt` (or `yolo26l.pt`) if `yolo26x` runs out of MPS memory; for a 1-class, big-object task on ~1000 images the medium model is faster and generalizes just as well.
- **`imgsz 640` is deliberate:** poles fill most of the frame, so 640 detects them fine and uses ~¼ the memory of 1280. (Save the high res for the *component* model.)
- It prints whether it's using **`yolo26x`** or fell back to **`yolo11x`** (fallback is normal if YOLO26 weights aren't fetchable on your Ultralytics version — not an error).
- **Output:** `runs/pole/yolo/weights/best.pt` (and `last.pt`). Re-runs go to `runs/pole/yolo2/…`.
- **Sanity check the scan line:** it must say `<N> images, 0 backgrounds` (e.g. `1387 images` for the pole train split). If it says `0 images, 1387 backgrounds`, labels aren't being found — see Troubleshooting.

**MPS OOM?** lower `--batch` (4→2), lower `--imgsz`, or switch `--model yolo26m.pt`.

## 7. Train Model 2 — the two component detectors

```bash
# 2a) the 4 frequent classes (wire, h_insulator, v_insulator, crossarm_stright)
python -m train_yolo.train_components --subset component_above_1000 --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26m.pt
# 2b) the 4 rare classes (vegetation, top_crossarm, om_crossarm, rust); train split is oversampled
python -m train_yolo.train_components --subset component_below_1000 --version clahe --epochs 200 --imgsz 1280 --batch 4 --model yolo26m.pt
# 2c) the 14 CONDITION classes (runs on the component crop at inference); train balanced to ~400/class
python -m train_yolo.train_components --subset component_classification --version clahe --epochs 150 --imgsz 1280 --batch 4 --model yolo26m.pt
```
- **Keep `imgsz 1280`** here — components include thin wires that vanish at low res.
- **`component_classification`** is the 4th-stage condition model: it detects condition classes (normal/band/broken/chip_off/…) on the **component crop**, not the pole. Output: `runs/component_classification/yolo/weights/best.pt`.
- **`yolo26x` at 1280 will likely OOM on 24 GB** → use `--model yolo26m.pt` (recommended) or `yolo26l.pt`, and/or `--batch 2`.
- **Output:** `runs/component_above_1000/yolo/weights/best.pt` and `runs/component_below_1000/yolo/weights/best.pt`.
- `component_below_1000` is the harder one (genuinely scarce classes); give it more epochs. Its train split is already offline-oversampled to equalize classes, and `train_args.py` adds stronger online aug on top.
- Always check **per-class AP** in the run's results (don't trust the single mAP) — Ultralytics writes a `results.csv` and PR curves under the run dir. Evaluate on the (un-augmented) val/test.

---

## 8. Get 100% out of the M4 (sustainably)

- **Always plugged in**, and disable Low Power Mode (System Settings → Battery). On battery the GPU throttles hard.
- **Watch utilization:** Activity Monitor → Window → **GPU History**, and `sudo powermetrics --samplers gpu_power -i1000` for GPU residency/Watts. Aim for high, steady GPU use — not thermal-throttled spikes.
- **Batch is the main lever.** Push `--batch` up until memory (`mactop`/Activity Monitor "Memory") is ~80–85 % used or you OOM, then back off one step. Bigger batch = better MPS utilization.
- **Dataset caching:** Ultralytics caches images to RAM by default when it fits 24 GB; if you see RAM pressure, the dataset is large — let it use disk cache instead (it falls back automatically).
- **Don't chase a literal 100 % GPU number** — a pegged laptop GPU just thermal-throttles. The real goal is fastest *converging* training: tune `batch`/`imgsz`, keep it cool, keep it plugged in.
- Close other heavy apps; MPS shares the 24 GB unified memory with the whole system.

---

## 9. Comparison models (optional)

### Faster R-CNN (torchvision, runs on MPS — slower than YOLO)
```bash
python -m train_faster_rcnn.train --subset pole                   --version clahe --epochs 30 --batch 2
python -m train_faster_rcnn.train --subset component_above_1000   --version clahe --epochs 30 --batch 2
python -m train_faster_rcnn.train --subset component_below_1000   --version clahe --epochs 30 --batch 2
python -m train_faster_rcnn.train --subset component_classification --version clahe --epochs 60 --batch 2
# downloads COCO-pretrained ResNet50-FPN weights on first run (needs network)
# output: runs/{subset}/faster_rcnn/{last.pt,best.pt}  -- best.pt = lowest val loss (--patience 7 early stop)
```
- **`--min-size` defaults to 2000** (torchvision's default 800 shrinks thin wires/insulators away); keep it high. Lower it only if you OOM.
- **Per-class AP:** `python -m train_faster_rcnn.eval --subset <s> --version clahe --split test` (uses `best.pt`; pass the **same `--min-size`** you trained with).

### RF-DETR-L (use Colab / CUDA — impractical on MPS)
Locally it will warn and try MPS; for a real run use the Colab notebook (Section 10):
```bash
python -m train_rf_detr.train --subset pole                   --version clahe --epochs 50 --batch 4 --resolution 672
python -m train_rf_detr.train --subset component_above_1000   --version clahe --epochs 50 --batch 4 --resolution 672
python -m train_rf_detr.train --subset component_below_1000   --version clahe --epochs 50 --batch 4 --resolution 672
python -m train_rf_detr.train --subset component_classification --version clahe --epochs 50 --batch 8 --resolution 1120
# output: runs/{subset}/rfdetr/checkpoint_best_ema.pth
```
- **`--resolution` must be a multiple of the model's block_size** (`patch_size*num_windows`, read from the installed lib: **32** on the current RF-DETR build, 56 on older ones). **672 / 896 / 1120** are multiples of *both* 32 and 56, so they're safe across versions and need no inference-shape rounding. Raise it (1120) for more small-object detail if the GPU has memory.

---

## 10. Colab path (CUDA — best for RF-DETR, faster for all)

➡️ **Full Colab + Google-Drive walkthrough: [`colab_instruction.md`](colab_instruction.md).** Quick version:

Notebooks are **generated** from `notebooks/build_notebooks.py` — edit the spec there, never the `.ipynb` by hand. `REPO_URL` is already set to `https://github.com/arupa444/trainDronisight.git` (only re-edit + regenerate if you fork/move the repo).

1. **Upload the DBs to Google Drive** as zips at `MyDrive/dronisight/`:
   ```bash
   (cd ~/dronisight_data && zip -rqX yolo_train_db.zip yolo_train_db -x '*/._*' \
      && zip -rqX RF_DETR_Faster_RCNN_train_db.zip RF_DETR_Faster_RCNN_train_db -x '*/._*')
   # then upload both .zip to Google Drive: MyDrive/dronisight/
   ```
2. In Colab open the notebook, **Runtime → Change runtime type → GPU**, then **Run all**:
   - `00_data_prep` (verify), `01_train_yolo`, `02_train_faster_rcnn`, `03_train_rf_detr`, `04_inference_pipeline`.
   - Each trainer copies `runs/` (weights + plots) to `MyDrive/dronisight/runs/` at the end; `04` restores it. See `colab_instruction.md` for the Drive layout and save/restore details.

---

## 11. Inference

### Single model (debugging)
```bash
python -m inference.infer_pole       --image some.jpg --weights runs/pole/yolo/weights/best.pt --conf 0.25
python -m inference.infer_components  --image crop.jpg --weights runs/component_above_1000/yolo/weights/best.pt --conf 0.25
```
(`infer_components` is generic YOLO — point `--weights` at any component or condition model.)

### Full pipeline (the real thing)
```bash
python -m inference.pipeline \
  --image some.jpg \
  --pole-weights runs/pole/yolo/weights/best.pt \
  --comp-above-weights runs/component_above_1000/yolo/weights/best.pt \
  --comp-below-weights runs/component_below_1000/yolo/weights/best.pt \
  --condition-weights runs/component_classification/yolo/weights/best.pt \
  --pole-pad 0.05 \
  --crop-dir runs/inference/crops \
  --out runs/inference/result.json
```
- Both component detectors run on the **pole crop**; their boxes are remapped to the full frame. **Preprocessing matches training:** EXIF-orient + CLAHE are applied **once** on the full frame and inherited by every crop. Pass `--no-clahe` only for `orig`-trained weights. Default imgsz mirrors training (pole 640, components 1280).
- **Stage 4 (optional):** add `--condition-weights …` and each detected component also carries a `conditions` list (the 14-class condition model run on its crop). Omit it to stop at detection.
- **Confidence defaults:** `--pole-conf 0.12` (recall-leaning — don't miss poles), `--comp-conf 0.25`, `--condition-conf 0.25`.
- **Mixed backends:** each stage takes `--pole-backend / --comp-above-backend / --comp-below-backend / --condition-backend {yolo,rfdetr,frcnn}`; point the matching `*-weights` at that family's checkpoint. RF-DETR stages share `--rfdetr-resolution` (a block_size multiple, e.g. 672/1120). Example — YOLO pole, RF-DETR components:
  ```bash
  python -m inference.pipeline --image some.jpg \
    --pole-backend yolo  --pole-weights runs/pole/yolo/weights/best.pt \
    --comp-above-backend rfdetr --comp-above-weights runs/component_above_1000/rfdetr/checkpoint_best_ema.pth \
    --comp-below-backend rfdetr --comp-below-weights runs/component_below_1000/rfdetr/checkpoint_best_ema.pth \
    --rfdetr-resolution 672 --out runs/inference/result.json
  ```
- `--pole-pad 0.05` adds a 5 % margin around the pole crop so components on the pole edge aren't clipped. Set `0.0` for a tight crop.
- **Output JSON shape:**
  ```json
  {
    "image": "some.jpg",
    "poles": [
      {
        "box": [x1,y1,x2,y2], "confidence": 0.97, "crop_path": "runs/inference/crops/some_pole0.jpg",
        "components_above": [
          { "class": "v_insulator", "confidence": 0.88,
            "box_full": [x1,y1,x2,y2], "box_crop": [x1,y1,x2,y2],
            "crop_path": "runs/inference/crops/some_pole0_above_comp0.jpg",
            "conditions": [ { "class": "v_insulator_broken", "confidence": 0.71, "box_comp": [x1,y1,x2,y2] } ] }
        ],
        "components_below": [ { "class": "vegetation", "confidence": 0.41, "box_full": [x1,y1,x2,y2], "box_crop": [x1,y1,x2,y2], "crop_path": "…_below_comp0.jpg" } ]
      }
    ]
  }
  ```
  `box_full` = full-frame coords (for drawing on the original); `box_crop` = coords within the pole crop; `conditions` is present only when `--condition-weights` is given (`box_comp` = coords within the component crop).

---

## 12. Evaluation discipline (per the design spec)

- Tune the **confidence threshold on val**, then freeze it; touch the **test split only once** at the end.
- Report **per-class AP**, not just mAP — `crossarm_stright` and small parts hide behind a good average.
- Run the **end-to-end** metric (`pole→crop→components`) to confirm the two-stage crop actually helps vs running Model 2 on full frames. The `--pole-pad` flag is your knob for that ablation.
- Compare **`orig` vs `clahe`** trainings — CLAHE is a hypothesis, not a guarantee; keep whichever wins on val.

---

## 13. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| Scan says **`0 images, N backgrounds`** / "Labels are missing or empty" | YOLO can't find the labels: they must mirror the image variant dir (`labels/<split>/<orig\|clahe>/`). **Fixed in current code** — `git pull` and rebuild, **or** patch existing data in place: see snippet below this table. Always `find <DB> -name '*.cache' -delete` afterward (a poisoned cache reports 0 labels even after fixing). |
| Training **scans a different path** than your data (e.g. reads `/Volumes/dronisight/…` even with `DRONISIGHT_DATA` set) | The `data.yaml` has an absolute `path:` baked in at build time. **Fixed in current code** — the trainer regenerates the yaml to the current DB location on each run. `git pull`, then `find "$DRONISIGHT_DATA/yolo_train_db" -name '*.cache' -delete` and re-run. |
| `MPS backend out of memory` | Lower `--batch` (4→2), lower `--imgsz`, or use a lighter `--model yolo26m.pt`. Close other apps. Keep plugged in. |
| Training is on CPU, not GPU | `torch.backends.mps.is_available()` is False → reinstall torch ≥ 2.2 in the venv (`uv pip install -e ".[dev]"`). |
| "using yolo11x fallback weights" warning | Expected if YOLO26 weights aren't fetchable on your Ultralytics version. Harmless; update `ultralytics` if you specifically want YOLO26. |
| Errors reading `._*.jpg` / weird image counts | exFAT AppleDouble sidecars. Re-copy with `rsync --exclude '._*'`, or `find <DB> -name '._*' -delete`. The pipeline filters them, but clean data is best. |
| `verify_dataset` UnicodeDecode / leakage error | Re-run `data_prep.build_dataset` (it self-cleans + re-splits deterministically), then re-verify. |
| First Faster R-CNN run stalls | It's downloading COCO-pretrained weights (~160 MB). Needs network; cached after. |
| `FileNotFoundError` on a DB path | `DRONISIGHT_DATA` isn't set/exported, or points at the wrong folder. `echo $DRONISIGHT_DATA` and re-check Section 3. |
| RF-DETR painfully slow / unstable on MPS | Expected — use the Colab GPU notebook (Section 10). |

**Patch already-built data in place** (if you copied DBs built before the label-layout fix — no rebuild needed):
```bash
DB=/Volumes/dronisight/yolo_train_db          # or $DRONISIGHT_DATA/yolo_train_db
for sub in pole component_above_1000 component_below_1000 component_classification; do
  for split in train val test; do
    s="$DB/$sub/labels/$split"
    for v in orig clahe; do mkdir -p "$s/$v"; cp "$s"/*.txt "$s/$v"/ 2>/dev/null; done
  done
done
find "$DB" -name "*.cache" -delete
```

---

## 14. What's NOT here yet (so you don't go looking)

The **condition classifier now exists** — `component_classification` (14 condition classes) is trained like the other detectors and runs as the optional 4th pipeline stage (`--condition-weights`, Section 11). Still deliberately future work:
- **Point/scoring system** per pole (aggregate the per-component conditions into a pole health score).
- **Report + OpenStreetMap** UI (UUID from lat/long → map pin → detail page).

The drone EXIF already carries GPS, so lat/long is available to wire up when those stages begin.
