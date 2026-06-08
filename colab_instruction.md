# colab_instruction.md — Training on Google Colab (data on Google Drive)

How to train all three model families — **YOLO26**, **Faster R-CNN**, **RF-DETR-L** — on a Colab
GPU, with the datasets stored on **Google Drive**. RF-DETR-L in particular wants CUDA, so Colab is
the recommended home for it; the same notebooks also train YOLO and Faster R-CNN faster than the M4.

The five notebooks in `notebooks/` are **generated** by `notebooks/build_notebooks.py` — edit the
spec there and regenerate (`python -m notebooks.build_notebooks`), **never hand-edit the `.ipynb`**.

---

## 0. Mental model

- **Data lives on Drive as two zips**; each notebook mounts Drive, unzips the DB it needs to fast
  local `/content/data`, and points `DRONISIGHT_DATA` there so `shared/config.py` resolves.
- **Colab runtimes are ephemeral.** Anything under `/content` (including `runs/` with your trained
  weights) is **wiped** when the runtime recycles. So every training notebook ends by copying
  `runs/` back to Drive (`save_runs_to_drive()`), and the inference notebook restores it
  (`restore_runs_from_drive()`). **If you skip the save cell, your weights are gone.**
- **Device auto-selects** `CUDA → MPS → CPU`, so the identical commands run on the M4 (MPS) and
  Colab (CUDA) unchanged.

---

## 1. One-time setup on Google Drive

1. **Zip both DBs** (on the machine that built them) and upload to `MyDrive/dronisight/`:
   ```bash
   (cd ~/dronisight_data && \
     zip -rqX yolo_train_db.zip yolo_train_db -x '*/._*' && \
     zip -rqX RF_DETR_Faster_RCNN_train_db.zip RF_DETR_Faster_RCNN_train_db -x '*/._*')
   # upload both .zip to Google Drive at: MyDrive/dronisight/
   ```
   The `-x '*/._*'` drops the exFAT AppleDouble sidecars.

2. **Expected Drive layout** (the notebooks assume exactly this — root `MyDrive/dronisight`):
   ```
   MyDrive/dronisight/
   ├── yolo_train_db.zip                  # YOLO DB (pole + above/below components + component_classification)
   ├── RF_DETR_Faster_RCNN_train_db.zip   # COCO DB (Faster R-CNN + RF-DETR)
   └── runs/                              # created automatically — your saved weights/plots land here
   ```
   To use a different Drive folder, change `DRIVE_ROOT` in `notebooks/colab_utils.py` and regenerate.

3. `REPO_URL` is already set to `https://github.com/arupa444/trainDronisight.git` in
   `build_notebooks.py`. Only revisit if you fork/move the repo.

---

## 2. Run order

Open each notebook in Colab → **Runtime → Change runtime type → GPU** → **Run all**. The first cells
(`!nvidia-smi`, clone, `uv pip install -e .`, CUDA check, `mount_drive()`, set `DRONISIGHT_DATA`) are
shared across all five.

| Notebook | Does | Outputs (also copied to Drive `runs/`) |
|---|---|---|
| `00_data_prep` | Mount Drive, unzip both DBs to `/content/data`, `verify_dataset` for all 4 subsets | — |
| `01_train_yolo` | Train **pole** (640) + **component_above_1000** + **component_below_1000** + **component_classification** (all 1280) | `runs/{subset}/yolo/weights/best.pt` |
| `02_train_faster_rcnn` | Train Faster R-CNN for the 4 subsets | `runs/{subset}/faster_rcnn/best.pt` |
| `03_train_rf_detr` | Train RF-DETR-L for the 4 subsets (the CUDA reason for Colab); `--resolution` a block_size multiple (672 / 1120) | `runs/{subset}/rfdetr/checkpoint_best_ema.pth` |
| `04_inference_pipeline` | `restore_runs_from_drive()`, run the YOLO pipeline (optionally with the 4th condition stage), save `result.json` | `runs/inference/result.json` |

You only need `00` once per session (the unzip is cached by `ensure_dataset`). Then run whichever
trainer(s) you want; they're independent.

---

## 3. The three trainings

### YOLO26 (`01_train_yolo`) — primary
```bash
python -m train_yolo.train_pole                                       --version clahe --epochs 100 --imgsz 640  --batch 16
python -m train_yolo.train_components --subset component_above_1000   --version clahe --epochs 150 --imgsz 1280 --batch 16 --model yolo26m.pt
python -m train_yolo.train_components --subset component_below_1000   --version clahe --epochs 200 --imgsz 1280 --batch 16 --model yolo26m.pt
python -m train_yolo.train_components --subset component_classification --version clahe --epochs 150 --imgsz 1280 --batch 16 --model yolo26m.pt
```
- **Pole at 640** is deliberate — poles fill the frame; 640 detects them fine at ¼ the memory of 1280.
- **Components at 1280** — thin wires vanish at low res. `yolo26m` is the recommended size (anti-OOM
  and the main anti-overfit lever); bump to `yolo26l`/`yolo26x` only if the GPU has room and the val
  curve isn't overfitting.
- **`component_below_1000`** holds the rare classes; its train split is already offline-oversampled to
  equalize classes, so give it more epochs and check per-class AP (eval on the un-augmented val/test).
- Prints whether it's using `yolo26x` or fell back to `yolo11x` (fallback is expected if YOLO26
  weights aren't fetchable on the installed Ultralytics — not an error).

### Faster R-CNN (`02_train_faster_rcnn`) — torchvision baseline
```bash
python -m train_faster_rcnn.train --subset pole                   --version clahe --epochs 30 --batch 4
python -m train_faster_rcnn.train --subset component_above_1000   --version clahe --epochs 30 --batch 4
python -m train_faster_rcnn.train --subset component_below_1000   --version clahe --epochs 30 --batch 4
python -m train_faster_rcnn.train --subset component_classification --version clahe --epochs 60 --batch 2
```
Downloads COCO-pretrained ResNet50-FPN on first run (needs network; cached after). Reads the COCO DB.
`best.pt` (lowest val loss) is the checkpoint to use; per-class AP via `train_faster_rcnn.eval`.

### RF-DETR-L (`03_train_rf_detr`) — CUDA-preferred
```bash
python -m train_rf_detr.train --subset pole                   --version clahe --epochs 50 --batch 4 --resolution 672
python -m train_rf_detr.train --subset component_above_1000   --version clahe --epochs 50 --batch 4 --resolution 672
python -m train_rf_detr.train --subset component_below_1000   --version clahe --epochs 50 --batch 4 --resolution 672
python -m train_rf_detr.train --subset component_classification --version clahe --epochs 50 --batch 8 --resolution 1120
```
`--resolution` must be a multiple of the model block_size (32 on the current build); 672/896/1120 also satisfy 56.
Builds a `valid`-named COCO view via symlinks (no image copy) and trains RF-DETR-L. This is the model
that's impractical on MPS — Colab CUDA is its intended home.

---

## 4. GPU memory & batch tuning

`--batch 16` is set for a roomy GPU (A100/L4). On a **T4 (16 GB)** the 1280-px component job is the
most likely to OOM — drop to `--batch 8` (or `4`), and/or keep `yolo26m`. Check your card with the
`!nvidia-smi` cell. For Faster R-CNN / RF-DETR, lower `--batch` similarly if you hit CUDA OOM.

To change a batch size or epochs, edit the command string in `notebooks/build_notebooks.py`, run
`python -m notebooks.build_notebooks`, commit, and re-open the notebook in Colab (it `git pull`s on
the next run).

---

## 5. Saving & reusing weights (don't lose your training)

- Each training notebook's **last cell** calls `save_runs_to_drive()` → copies the whole local
  `runs/` tree (weights, `results.csv`, PR curves) to `MyDrive/dronisight/runs/`.
- `04_inference_pipeline` calls `restore_runs_from_drive()` first, so you can run inference in a
  brand-new runtime without retraining — it pulls the weights back from Drive into `runs/`.
- To grab weights locally afterward, just download them from `MyDrive/dronisight/runs/` in Drive.

---

## 6. Evaluation discipline (same as the M4 guide)

- Tune the confidence threshold on **val**, freeze it, touch **test** only once.
- Report **per-class AP**, not just mAP — `crossarm_stright` and small parts hide behind a good average.
- Compare **`orig` vs `clahe`** trainings (swap `--version`); keep whichever wins on val mAP.
- Inference applies EXIF-orient + CLAHE once on the full frame, so a `clahe`-trained model sees its
  trained distribution. Use `--no-clahe` only for `orig`-trained weights.

---

## 7. Troubleshooting (Colab-specific)

| Symptom | Fix |
|---|---|
| `FileNotFoundError` on `/content/data/...` or a `/Volumes/dronisight` path | The DB zip isn't on Drive at `MyDrive/dronisight/`, or `00`'s unzip cell didn't run. `DRONISIGHT_DATA` is set to `/content/data` in setup — re-run the setup + `ensure_dataset` cells. |
| `git clone` fails in setup | Repo is private — authenticate, or confirm `REPO_URL` is correct. |
| `CUDA: False` printed | Runtime isn't on GPU: **Runtime → Change runtime type → GPU**, then **Run all** again. |
| `CUDA out of memory` | Lower `--batch` (16→8→4); keep `yolo26m`; lower `--imgsz` only for components as a last resort. |
| Weights gone after reconnect | You skipped the `save_runs_to_drive()` cell (or the runtime recycled mid-run). Re-train and let the save cell finish; verify files appear under `MyDrive/dronisight/runs/`. |
| `verify_dataset` leakage/label error | Re-build the DB on the source machine and re-upload the zip; don't hand-edit the DB. |
| RF-DETR import error locally | `rfdetr` installs on Colab; the trainer mocks it in tests. Use the Colab GPU notebook for real runs. |

---

## 8. What's NOT here

Detection only. Condition classifier, per-pole scoring, and the OpenStreetMap report UI are future
work (see the repo `README.md` roadmap).
