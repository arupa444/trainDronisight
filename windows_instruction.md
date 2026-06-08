# windows_instruction.md — trainDronisight on Windows + NVIDIA CUDA

The Windows/CUDA runbook. Same project as [`INSTRUCTION.md`](INSTRUCTION.md) (the macOS/M4 guide)
— this covers the Windows-specific setup. **Your input is the SSD** (plug it in; it mounts as a
drive letter such as `E:\` and holds the raw source folders **and** the prebuilt DBs).

> **Why Windows+CUDA is the best box for this repo:** `shared/device.py` auto-selects
> `CUDA → MPS → CPU`, so every command is identical to the Mac guide — but on CUDA **all three
> model families train locally, including RF-DETR-L** (which has to run on Colab from a Mac).
> No code changes; just point `DRONISIGHT_DATA` at your drive.

Use **PowerShell** for everything below (commands shown PowerShell-style). `cmd` works too — only
the env-var, activation, and line-continuation syntax differ (noted where it matters).

---

## 0. What you need

- Windows 10/11 x64 with an **NVIDIA GPU**.
- A recent **NVIDIA driver** (GeForce/Studio). You do **not** need the standalone CUDA Toolkit —
  the PyTorch CUDA wheels bundle their own CUDA runtime. Verify the driver sees the GPU:
  ```powershell
  nvidia-smi          # shows the GPU, driver version, and the max CUDA version it supports
  ```
- **Git for Windows**, and **uv** (the Python manager — this project uses `uv`, never bare `pip`):
  ```powershell
  winget install --id Git.Git -e
  winget install --id astral-sh.uv -e        # or: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  Restart the shell, confirm `uv --version` and `git --version`.

## 1. Get the code

```powershell
git clone https://github.com/arupa444/trainDronisight.git
cd trainDronisight
```

## 2. Python env + the CUDA build of PyTorch  ⚠️ most important step

```powershell
uv venv
.\.venv\Scripts\Activate.ps1          # cmd: .venv\Scripts\activate.bat
                                      # if PowerShell blocks it: Set-ExecutionPolicy -Scope Process RemoteSigned
uv pip install -e ".[dev]"
```

Now **verify CUDA is actually wired to torch** — the PyPI wheel can resolve to a CPU-only build:
```powershell
python -c "import torch; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda, '| available:', torch.cuda.is_available())"
```
- If `available: True` → you're done.
- If `available: False` (or `CUDA build: None`) → **force the CUDA wheel** matching your driver
  (`cu124` for recent drivers; `cu121` for older). Pick the one `nvidia-smi` supports:
  ```powershell
  uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
  python -c "import torch; print(torch.cuda.is_available())"     # must now print True
  ```

Sanity:
```powershell
pytest -q          # full suite passes
```

## 3. Point the code at the SSD (set DRONISIGHT_DATA)

`shared/config.py` defaults `DRONISIGHT_DATA` to the Mac path `/Volumes/dronisight`, so on Windows
you **must** set it to your drive. Say the SSD is `E:` and the DBs are at `E:\dronisight`:

```powershell
# this session only:
$env:DRONISIGHT_DATA = "E:\dronisight"
# persist for all NEW shells (then open a fresh terminal):
setx DRONISIGHT_DATA "E:\dronisight"

python -c "from shared import config; print(config.YOLO_DB)"     # -> E:\dronisight\yolo_train_db
```
`cmd` equivalent: `set DRONISIGHT_DATA=E:\dronisight`. Backslash paths are fine (`pathlib` handles them).

**Faster, safer (recommended): copy the DBs to a local NVMe drive.** USB/exFAT is slow and a
mid-build disconnect corrupts a rebuild. `robocopy` mirrors and skips the macOS `._*` sidecars:
```powershell
robocopy E:\dronisight\yolo_train_db                C:\dronisight\yolo_train_db                /E /XF ._*
robocopy E:\dronisight\RF_DETR_Faster_RCNN_train_db C:\dronisight\RF_DETR_Faster_RCNN_train_db /E /XF ._*
setx DRONISIGHT_DATA "C:\dronisight"     # open a new shell after setx
```
> The SSD is **exFAT** (built on a Mac), which Windows reads natively. The `._*` files are macOS
> AppleDouble sidecars; the code filters them, so they're harmless if present.

## 4. (Optional) Rebuild the datasets from raw annotations

Only if you change taxonomy / CLAHE / split / balance / crop logic — otherwise the 7 DBs are ready.
Identical to the Mac flow (no CUDA needed for the build itself):
```powershell
python -m data_prep.build_dataset --subset all          # 4 full-frame base subsets
python -m data_prep.build_dataset --subset all_crop      # 3 crop-aligned variants (or all_both for all 7)
foreach ($s in "pole","component_above_1000","component_below_1000","component_classification",
               "component_above_1000_crop","component_below_1000_crop","component_classification_crop") {
  python -m data_prep.verify_dataset --subset $s
}
```
What the build does (merge / condition-conflict resolution / per-subset balance / leakage-safe
splits / crop alignment) is explained in [`INSTRUCTION.md`](INSTRUCTION.md) §6–§8. **Keep the SSD
plugged in for the whole build.**

## 5. The datasets, preprocessing, model selection

These are identical across platforms — see [`INSTRUCTION.md`](INSTRUCTION.md):
- **§7 The datasets** — the 7 subsets (4 full-frame + 3 crop-aligned) and their sizes/balance.
- **§8 Preprocessing** — CLAHE (LAB-L, byte-exact train↔inference), EXIF, and the crop-aligned scale fix.
- **§12 Choosing the final per-stage model** — family × `orig`/`clahe` × full-frame/`_crop`, per-class AP.

## 6. Train (all three families run LOCALLY on CUDA)

CUDA fits much larger batches than the M4 — **tune `--batch` to your VRAM** (rough YOLO@1280 starting
points: 24 GB cards ≈ 16, 12 GB ≈ 6–8, 8 GB ≈ 4; halve for RF-DETR; pole@640 can go higher). Train
both `--version clahe` and `--version orig`, and both full-frame and the `_crop` variant, then keep
the val-mAP winner per stage.

> Multi-arg commands below are wrapped with PowerShell's backtick (`` ` ``) line-continuation. To
> paste as one line, drop the backticks. In `cmd`, use `^` instead of `` ` ``.

### 6a. YOLO26x (primary)
```powershell
python -m train_yolo.train_pole       --version clahe --epochs 100 --imgsz 640  --batch 16 --model yolo26x.pt
python -m train_yolo.train_components --subset component_above_1000    --version clahe --epochs 150 --imgsz 1280 --batch 8 --model yolo26x.pt
python -m train_yolo.train_components --subset component_below_1000    --version clahe --epochs 200 --imgsz 1280 --batch 8 --model yolo26x.pt
python -m train_yolo.train_components --subset component_classification --version clahe --epochs 150 --imgsz 1280 --batch 8 --model yolo26x.pt
# crop-aligned ablation (same flags, _crop subsets) — needs `build_dataset --subset all_crop`:
python -m train_yolo.train_components --subset component_above_1000_crop    --version clahe --epochs 150 --imgsz 1280 --batch 8 --model yolo26x.pt
python -m train_yolo.train_components --subset component_below_1000_crop    --version clahe --epochs 200 --imgsz 1280 --batch 8 --model yolo26x.pt
python -m train_yolo.train_components --subset component_classification_crop --version clahe --epochs 150 --imgsz 1280 --batch 8 --model yolo26x.pt
```
Weights: `runs\<subset>\yolo\weights\best.pt`. CUDA easily runs `yolo26x` at 1280 on a 24 GB card; on
smaller VRAM use `--model yolo26l.pt`/`yolo26m.pt` and/or lower `--batch`.

### 6b. Faster R-CNN
```powershell
python -m train_faster_rcnn.train --subset pole                    --version clahe --epochs 30 --batch 4
python -m train_faster_rcnn.train --subset component_above_1000    --version clahe --epochs 30 --batch 4
python -m train_faster_rcnn.train --subset component_below_1000    --version clahe --epochs 30 --batch 4
python -m train_faster_rcnn.train --subset component_classification --version clahe --epochs 60 --batch 2
python -m train_faster_rcnn.eval  --subset component_above_1000    --version clahe --split test     # per-class AP
```
`--min-size` defaults to **2000** (preserves thin wires); `best.pt` (lowest val loss) is the one to
use. `pin_memory` auto-enables on CUDA. If DataLoader workers stall on Windows, lower `--workers`
(default = CPU count) — Windows uses process-spawn, so very high worker counts add overhead.

### 6c. RF-DETR-L  (runs locally here — no Colab)
```powershell
python -m train_rf_detr.train --subset pole                    --version clahe --epochs 50 --batch 4 --resolution 672
python -m train_rf_detr.train --subset component_above_1000    --version clahe --epochs 50 --batch 4 --resolution 672
python -m train_rf_detr.train --subset component_below_1000    --version clahe --epochs 50 --batch 4 --resolution 672
python -m train_rf_detr.train --subset component_classification --version clahe --epochs 50 --batch 8 --resolution 1120
```
`--resolution` must be a multiple of the model **block_size** (`patch_size*num_windows` = 32 on the
current build); **672 / 896 / 1120** satisfy both 32 and 56. Output: `runs\<subset>\rfdetr\checkpoint_best_ema.pth`.
Raise `--resolution` (896/1120) for more small-object detail if VRAM allows.

## 7. Inference (four-stage pipeline)

Identical CLI to the Mac guide. CUDA is auto-used.
```powershell
python -m inference.pipeline --image some.jpg `
  --pole-weights        runs\pole\yolo\weights\best.pt `
  --comp-above-weights  runs\component_above_1000\yolo\weights\best.pt `
  --comp-below-weights  runs\component_below_1000\yolo\weights\best.pt `
  --condition-weights   runs\component_classification\yolo\weights\best.pt `
  --out runs\inference\result.json
```
- Omit `--condition-weights` to stop at component detection. Add `--no-clahe` only for `orig`-trained
  weights. Defaults: `--pole-conf 0.12`, `--comp-conf 0.25`, `--condition-conf 0.25`,
  `--pole-imgsz 640`, `--comp-imgsz 1280`, `--condition-imgsz 1280`, `--pole-pad 0.05`.
- **Mixed backends** per stage: `--pole-backend / --comp-above-backend / --comp-below-backend /
  --condition-backend {yolo,rfdetr,frcnn}` with the matching `*-weights`. RF-DETR stages share
  `--rfdetr-resolution` (672/1120); FRCNN stages share `--frcnn-min-size` (default 2000 — must match
  training). Single-stage debug CLIs: `inference.infer_pole`, `inference.infer_components`.
- Full output JSON shape is in [`INSTRUCTION.md`](INSTRUCTION.md) §13.

## 8. Long / unattended runs

No `caffeinate` needed. Just keep the terminal open, or detach:
```powershell
Start-Process -NoNewWindow -FilePath python -ArgumentList "-m","train_yolo.train_pole","--version","clahe","--epochs","100","--imgsz","640","--batch","16" -RedirectStandardOutput runs\logs\pole.out -RedirectStandardError runs\logs\pole.err
```
Disable sleep/hibernate during long trainings (Settings → Power, or `powercfg /change standby-timeout-ac 0`).

## 9. Troubleshooting (Windows-specific)

| Symptom | Cause / fix |
|---|---|
| `torch.cuda.is_available()` is **False** | CPU-only torch wheel. Reinstall from the CUDA index (§2): `uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124`. Match `cu124`/`cu121` to what `nvidia-smi` supports. |
| `FileNotFoundError` / wrong DB path | `DRONISIGHT_DATA` not set (defaults to the Mac path). `echo $env:DRONISIGHT_DATA`; set it to your drive (§3) and open a **new** shell after `setx`. |
| `CUDA out of memory` | Lower `--batch`, lower `--imgsz`, or use a smaller `--model` (yolo26l/m). Close other GPU apps. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation. |
| `Activate.ps1 cannot be loaded` (execution policy) | `Set-ExecutionPolicy -Scope Process RemoteSigned`, then re-run activate. Or use `cmd`: `.venv\Scripts\activate.bat`. |
| DataLoader hangs / slow at epoch start | Windows spawns worker processes; lower `--workers` (e.g. 4). The trainers already guard `__main__`, so this is just overhead, not a crash. |
| `nvidia-smi` not found | Install/repair the NVIDIA driver; reboot. |
| Long path / `MAX_PATH` errors on a deep `runs\...` tree | Enable Win32 long paths (`git config --system core.longpaths true`; or the LongPathsEnabled registry key). |
| exFAT `._*` files everywhere | macOS AppleDouble sidecars — harmless (code filters them). Exclude on copy with `robocopy ... /XF ._*`. |
| Mid-build duplicated images / odd counts | The SSD disconnected mid-build. Don't unplug; rebuild the affected subset; `verify_dataset` confirms no leakage. |

## 10. What's NOT here yet

Same roadmap as the Mac guide ([`INSTRUCTION.md`](INSTRUCTION.md) §17): the condition classifier
exists (the optional 4th stage); per-pole scoring and the OpenStreetMap report UI are future work.
