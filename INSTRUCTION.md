# INSTRUCTION.md — Running trainDronisight on the Mac M4 Pro

A deep, do-this-then-that guide for taking this repo + the prepared data and **training, evaluating, and
running inference on the M4 Pro (24 GB, Apple Silicon / MPS)**.

Conventions: shell commands assume you've `cd`-ed into the repo and **activated the venv**
(`source .venv/bin/activate`) unless stated otherwise.

---

## 0. Mental model (read once)

- **Two machines.** Code + the one-time data build happen on the M1 laptop. **All training/inference runs here on the M4 Pro.**
- **Two-stage detection.** Model 1 finds the pole; Model 2 finds components on the cropped pole region. They are **separate trainings** producing separate weights, chained only at inference.
- **No CUDA on Apple Silicon.** YOLO and Faster R-CNN train fine on **MPS**. RF-DETR-L really wants CUDA → run that one on **Colab** (Section 9). The code auto-selects `CUDA → MPS → CPU`, so the same commands work everywhere.
- **The data is already built.** You do **not** need to re-run data-prep unless you want to change the taxonomy/preprocessing. You just point the code at the DB folders.

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
pytest -q                                                                              # expect 82 passed
```

If `MPS available: False`, you're likely on an Intel Mac or an old torch — reinstall torch ≥ 2.2.

---

## 5. (Optional) Rebuild the data from raw annotations

**Only if** you changed the class policy, CLAHE, or split logic. Requires the raw `mem2…mem8` folders present under `DRONISIGHT_DATA`. Otherwise skip — the DBs are ready.

```bash
python -m data_prep.build_dataset --subset all          # builds pole + components into both DBs
python -m data_prep.verify_dataset --subset pole        # leakage + label-validity gate
python -m data_prep.verify_dataset --subset components
```
Useful flags: `--no-balance` (keep all images, skip the class cap — pair with `sample_weights.csv` at train time).
Each run self-cleans AppleDouble sidecars and prints split sizes + any skipped (EXIF-mismatch / unparseable) files.

---

## 6. Train Model 1 — pole detector (YOLO26x, MPS)

```bash
python -m train_yolo.train_pole --version clahe --epochs 100 --imgsz 1280 --batch 4
```
- `--version {orig,clahe}` — which image variant to train on (train both, compare).
- It prints whether it's using **`yolo26x`** or fell back to **`yolo11x`** (fallback is normal if YOLO26 weights aren't fetchable on your Ultralytics version — not an error).
- **Output:** `runs/pole/yolo/weights/best.pt` (and `last.pt`). Re-runs go to `runs/pole/yolo2/…`.

**Batch sizing for 24 GB (important):** the `x` models are large. Start at `--batch 4 --imgsz 1280`. If you hit an MPS out-of-memory error, drop to `--batch 2`, or lower `--imgsz` to 1024/960. If memory is comfortable and the GPU isn't saturated, try `--batch 8`. (The CLI default is 8 — lower it if you OOM.)

## 7. Train Model 2 — component detector

```bash
python -m train_yolo.train_components --version clahe --epochs 150 --imgsz 1280 --batch 4
```
- **Output:** `runs/components/yolo/weights/best.pt`.
- More epochs than pole (4 classes, harder).
- **Class imbalance:** the default DB is instance-capped but co-occurrence leaves `wire` ≫ `crossarm_stright`. Two levers if `crossarm_stright`/recall lags:
  1. Rebuild with `--no-balance` and use `<DB>/components/sample_weights.csv` for inverse-frequency weighted sampling.
  2. Check **per-class AP** in the run's results (don't trust the single mAP) — Ultralytics writes a `results.csv` and PR curves under the run dir.

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
python -m train_faster_rcnn.train --subset pole       --version clahe --epochs 30 --batch 2
python -m train_faster_rcnn.train --subset components --version clahe --epochs 30 --batch 2
# downloads COCO-pretrained ResNet50-FPN weights on first run (needs network)
# output: runs/{subset}/faster_rcnn/last.pt
```

### RF-DETR-L (use Colab / CUDA — impractical on MPS)
Locally it will warn and try MPS; for a real run use the Colab notebook (Section 10):
```bash
python -m train_rf_detr.train --subset components --version clahe --epochs 50 --batch 4
```

---

## 10. Colab path (CUDA — best for RF-DETR, faster for all)

Notebooks are **generated** from `notebooks/build_notebooks.py` — edit the spec there, never the `.ipynb` by hand.

1. **Set your repo URL:** edit `REPO_URL` in `notebooks/build_notebooks.py` to
   `https://github.com/arupa444/trainDronisight.git`, then regenerate:
   ```bash
   python -m notebooks.build_notebooks
   git add notebooks && git commit -m "chore: set Colab REPO_URL" && git push
   ```
2. **Upload the DBs to Google Drive** as zips at `MyDrive/dronisight/`:
   ```bash
   (cd ~/dronisight_data && zip -rqX yolo_train_db.zip yolo_train_db -x '*/._*' \
      && zip -rqX RF_DETR_Faster_RCNN_train_db.zip RF_DETR_Faster_RCNN_train_db -x '*/._*')
   # then upload both .zip to Google Drive: MyDrive/dronisight/
   ```
3. In Colab open the notebook, **Runtime → Change runtime type → GPU**, then **Run all**:
   - `00_data_prep` (verify), `01_train_yolo`, `02_train_faster_rcnn`, `03_train_rf_detr`, `04_inference_pipeline`.

---

## 11. Inference

### Single model (debugging)
```bash
python -m inference.infer_pole       --image some.jpg --weights runs/pole/yolo/weights/best.pt --conf 0.25
python -m inference.infer_components  --image crop.jpg --weights runs/components/yolo/weights/best.pt --conf 0.25
```

### Full two-stage pipeline (the real thing)
```bash
python -m inference.pipeline \
  --image some.jpg \
  --pole-weights runs/pole/yolo/weights/best.pt \
  --comp-weights runs/components/yolo/weights/best.pt \
  --pole-pad 0.05 \
  --crop-dir runs/inference/crops \
  --out runs/inference/result.json
```
- `--pole-pad 0.05` adds a 5 % margin around the pole crop so components on the pole edge (wire ends, crossarm tips) aren't clipped. Set `0.0` for a tight crop; raise it to test the trade-off.
- **Output JSON shape:**
  ```json
  {
    "image": "some.jpg",
    "poles": [
      {
        "box": [x1,y1,x2,y2], "confidence": 0.97, "crop_path": "runs/inference/crops/some_pole0.jpg",
        "components": [
          { "class": "wire", "confidence": 0.88,
            "box_full": [x1,y1,x2,y2], "box_crop": [x1,y1,x2,y2],
            "crop_path": "runs/inference/crops/some_pole0_comp0.jpg" }
        ]
      }
    ]
  }
  ```
  `box_full` = full-frame coords (for drawing on the original); `box_crop` = coords within the pole crop; each component is also saved as its own crop (ready for the future condition classifier).

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
| `MPS backend out of memory` | Lower `--batch` (8→4→2) and/or `--imgsz` (1280→1024→960). Close other apps. |
| Training is on CPU, not GPU | `torch.backends.mps.is_available()` is False → reinstall torch ≥ 2.2 in the venv (`uv pip install -e ".[dev]"`). |
| "using yolo11x fallback weights" warning | Expected if YOLO26 weights aren't fetchable on your Ultralytics version. Harmless; update `ultralytics` if you specifically want YOLO26. |
| Errors reading `._*.jpg` / weird image counts | exFAT AppleDouble sidecars. Re-copy with `rsync --exclude '._*'`, or `find <DB> -name '._*' -delete`. The pipeline filters them, but clean data is best. |
| `verify_dataset` UnicodeDecode / leakage error | Re-run `data_prep.build_dataset` (it self-cleans + re-splits deterministically), then re-verify. |
| First Faster R-CNN run stalls | It's downloading COCO-pretrained weights (~160 MB). Needs network; cached after. |
| `FileNotFoundError` on a DB path | `DRONISIGHT_DATA` isn't set/exported, or points at the wrong folder. `echo $DRONISIGHT_DATA` and re-check Section 3. |
| RF-DETR painfully slow / unstable on MPS | Expected — use the Colab GPU notebook (Section 10). |

---

## 14. What's NOT here yet (so you don't go looking)

This repo is **detection only**. These are deliberately future work:
- **Condition classifier** (broken/defective per component) — needs new condition labels; current annotations are object-type only.
- **Point/scoring system** per pole.
- **Report + OpenStreetMap** UI (UUID from lat/long → map pin → detail page).

The drone EXIF already carries GPS, so lat/long is available to wire up when those stages begin.
