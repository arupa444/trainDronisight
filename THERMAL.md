# Thermal transformer-defect cascade

A self-contained capability (the `thermal/` package) for detecting overheating
conductors/connections on transformers from **colorized thermal images**. It runs
alongside the pole-inspection pipeline but is independent of it.

> Relative severity only (Normal/Watch/Investigate/Critical) — the input palette is
> auto-gained per frame, so there is no absolute °C. See "Limits" below.

## Architecture (one model + CV)

```
image → CLAHE → [YOLO26x transformer detector] → for each transformer:
                    pad 0.30 + scan crop for hot regions (relative heat, raw palette)
                                       ↓
                    connected-component hotspots → severity by heat-above-body
```

A single YOLO26x detector localizes the **transformer**; hot conductors/connections
are then found by **CV** (a region hotter than the transformer body) — there is no
learned wire model (thin clustered conductors weren't learnable; CV targets the
defect signal directly).

## Files
- `thermal/detector.py` — `YoloDetector` (loads `models/transformer.pt`; conf floor 0.12).
- `thermal/preprocess.py` — adaptive CLAHE (train/serve parity), fed to the detector as BGR.
- `thermal/colormap.py` — iron palette → 0–1 heat map (on the raw RGB image).
- `thermal/defects.py` — `find_hotspots` (Otsu body reference + severity floors).
- `thermal/pipeline.py` — `analyze_image(img_rgb, detector, c2h)` (the cascade).
- `thermal/report.py` — annotated image + JSON.
- `thermal/api.py` — FastAPI `/analyze`.
- `thermal/data_prep/` — VOC parse + canonical names, content-hash dedup/merge,
  leakage-safe grouped split, adaptive CLAHE, dataset builder.
- `models/transformer.pt` — the trained YOLO26x detector (gitignored, like other weights).

## Run
```bash
uv pip install -e .            # installs scikit-image etc.; exposes the `thermal` package
pytest tests/thermal -q        # 33 tests, no model needed

# API:
THERMAL_TRANSFORMER_WEIGHTS=models/transformer.pt uvicorn thermal.api:app --reload
curl -s -F "file=@frame.jpg" http://127.0.0.1:8000/analyze | jq
```
Returns `calibration_ok`, `defects[]` (component=`hotspot`, bbox, severity,
relative_delta), and `annotated_image_png_b64`.

## Tuning (one constant each)
- Detector recall vs precision: `thermal/detector.py` → `_DEFAULT_CONF` (0.12).
- Hotspot floor / severity: `thermal/defects.py` → `_HOTSPOT_MARGIN`,
  `_WATCH`/`_INVESTIGATE`/`_CRITICAL` (0.38/0.48/0.58, calibrated on 755 crops).
- Crop pad: `thermal/pipeline.py` → `HOTSPOT_PAD` (0.30).

## Limits (be honest)
The palette is auto-gained per frame → no absolute temperature. A white-hot
connection on an *already-warm* body has a compressed relative margin (~+0.40), so it
grades as Watch, not Critical, and is hard to separate from a normal warm connection.
**Radiometric (raw °C) data is the real fix** for accurate grading; a learned
defect-vs-normal classifier is the alternative if only screenshots are available.
