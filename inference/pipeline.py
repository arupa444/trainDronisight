"""Four-stage inference: pole detect -> crop -> run BOTH the component_above_1000 and
component_below_1000 detectors on the pole crop -> remap boxes to the full frame -> crop
each component -> (optional) run the component_classification condition detector on each
component crop -> structured JSON. Usage:
    python -m inference.pipeline --image path.jpg \
        --pole-weights pole.pt --comp-above-weights above.pt --comp-below-weights below.pt \
        [--condition-weights condition.pt]
Each stage's backend is independently selectable: yolo | rfdetr | frcnn.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2

from shared import config
from inference.backends import YoloDetector, RFDetrDetector, TorchvisionDetector
from inference.geometry import crop_with_pad, shift_detection
from data_prep.preprocess import load_oriented_bgr, clahe_image

BACKENDS = ["yolo", "rfdetr", "frcnn"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def build_detector(backend, weights, conf, imgsz, class_names, resolution=672, frcnn_min_size=2000):
    """Pick a detector backend for a pipeline stage. YOLO reads class names from the model;
    RF-DETR and Faster R-CNN need the explicit class_names (they return numeric class ids).
    frcnn_min_size MUST match the Faster R-CNN training --min-size (default 2000) or small
    objects are served at the wrong scale."""
    if backend == "rfdetr":
        return RFDetrDetector(weights, class_names, conf=conf, resolution=resolution)
    if backend == "frcnn":
        return TorchvisionDetector(weights, class_names, conf=conf, min_size=frcnn_min_size)
    return YoloDetector(weights, conf=conf, imgsz=imgsz)


def _save_crop(crop, crop_dir, name):
    if crop_dir is None or crop.size == 0:
        return None
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    path = crop_dir / name
    cv2.imwrite(str(path), crop)
    return str(path)


def _classify_condition(comp_crop, condition_detector, component_class):
    """Stage 4: run the condition detector on ONE component crop, then MAP — keep only the
    condition classes valid for this component's family (config.COMPONENT_TO_CONDITIONS). A
    v_insulator crop can't be a crossarm/wire condition. Returns (best, all_valid):
      * best = {"class","confidence"} highest-confidence in-family condition, or None
      * all_valid = every in-family condition detection (sorted), or None if no family / no detector.
    Components with no condition family (vegetation, rust) -> (None, None)."""
    allowed = config.COMPONENT_TO_CONDITIONS.get(component_class)
    if condition_detector is None or allowed is None or comp_crop.size == 0:
        return None, None
    allowed = set(allowed)
    dets = sorted((c for c in condition_detector.predict(comp_crop) if c.class_name in allowed),
                  key=lambda c: c.confidence, reverse=True)
    all_valid = [{"class": c.class_name, "confidence": c.confidence,
                  "box_comp": [int(v) for v in c.box]} for c in dets]
    best = {"class": dets[0].class_name, "confidence": dets[0].confidence} if dets else None
    return best, all_valid


def _iou_xyxy(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ba = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ba - inter)


def nms_components(items, iou_thresh):
    """Class-agnostic greedy NMS so each PHYSICAL object yields ONE box. items: list of
    (comp_det_in_crop_coords, full_det, group). Keeps the highest-confidence detection and drops
    any later box whose full-frame IoU with a kept box is >= iou_thresh — this removes the same
    insulator/crossarm detected by BOTH the above & below detectors (and intra-detector dupes).
    A high threshold keeps genuinely co-located distinct objects (e.g. a wire crossing a crossarm)."""
    items = sorted(items, key=lambda t: t[1].confidence, reverse=True)
    kept = []
    for it in items:
        if all(_iou_xyxy(it[1].box, k[1].box) < iou_thresh for k in kept):
            kept.append(it)
    return kept


def run_pipeline(image, pole_detector, above_detector, below_detector,
                 crop_dir, image_name, pole_pad=0.05, condition_detector=None, name_stem=None,
                 nms_iou=0.55):
    """image: BGR ndarray. Returns the structured result dict. Both component detectors run on
    the padded pole crop; their boxes are remapped to the full frame and de-duplicated with
    class-agnostic NMS (nms_iou; set >=1.0 to disable) so each object gets ONE box. Each surviving
    component carries its mapped `condition` (stage 4). `name_stem` overrides saved-crop filenames."""
    stem = name_stem or Path(image_name).stem
    result = {"image": image_name, "poles": []}
    for pi, pole in enumerate(pole_detector.predict(image)):
        pole_crop, (ox, oy) = crop_with_pad(image, pole.box, pad_frac=pole_pad)
        pole_crop_path = _save_crop(pole_crop, crop_dir, f"{stem}_pole{pi}.jpg")
        # both component detectors on the pole crop -> remap to full frame -> cross-detector NMS
        combined = ([(c, shift_detection(c, ox, oy), "above") for c in above_detector.predict(pole_crop)]
                    + [(c, shift_detection(c, ox, oy), "below") for c in below_detector.predict(pole_crop)])
        if nms_iou and nms_iou < 1.0:
            combined = nms_components(combined, nms_iou)
        above_out, below_out = [], []
        for ci, (comp, full, grp) in enumerate(combined):
            comp_crop, _ = crop_with_pad(image, full.box, pad_frac=0.0)
            entry = {
                "class": comp.class_name,
                "confidence": comp.confidence,
                "box_crop": [int(v) for v in comp.box],
                "box_full": [int(v) for v in full.box],
                "crop_path": _save_crop(comp_crop, crop_dir, f"{stem}_pole{pi}_{grp}_comp{ci}.jpg"),
            }
            best, all_valid = _classify_condition(comp_crop, condition_detector, comp.class_name)
            if all_valid is not None:               # condition stage ran for this component family
                entry["condition"] = best           # the mapped, in-family top condition (or None)
                entry["conditions"] = all_valid       # all in-family condition detections
            (above_out if grp == "above" else below_out).append(entry)
        result["poles"].append({
            "box": [int(v) for v in pole.box],
            "confidence": pole.confidence,
            "crop_path": pole_crop_path,
            "components_above": above_out,
            "components_below": below_out,
        })
    return result


CSV_COLUMNS = ["image", "pole_index", "pole_confidence", "pole_x1", "pole_y1", "pole_x2", "pole_y2",
               "group", "component_class", "component_confidence",
               "comp_x1", "comp_y1", "comp_x2", "comp_y2",
               "condition_class", "condition_confidence", "crop_path"]


def result_to_rows(result):
    """Flatten one pipeline result dict into CSV rows: ONE row per detected component (its mapped
    condition inline), plus a component-less row for any pole with no components."""
    rows = []
    img = result["image"]
    for pi, pole in enumerate(result["poles"]):
        px = pole["box"]
        base = {"image": img, "pole_index": pi, "pole_confidence": round(pole["confidence"], 4),
                "pole_x1": px[0], "pole_y1": px[1], "pole_x2": px[2], "pole_y2": px[3]}
        comps = ([("above", c) for c in pole.get("components_above", [])]
                 + [("below", c) for c in pole.get("components_below", [])])
        if not comps:
            rows.append({**base, "group": "", "component_class": "", "component_confidence": "",
                         "comp_x1": "", "comp_y1": "", "comp_x2": "", "comp_y2": "",
                         "condition_class": "", "condition_confidence": "", "crop_path": pole.get("crop_path") or ""})
            continue
        for grp, c in comps:
            b = c["box_full"]
            cond = c.get("condition")
            rows.append({**base, "group": grp,
                         "component_class": c["class"], "component_confidence": round(c["confidence"], 4),
                         "comp_x1": b[0], "comp_y1": b[1], "comp_x2": b[2], "comp_y2": b[3],
                         "condition_class": (cond["class"] if cond else ""),
                         "condition_confidence": (round(cond["confidence"], 4) if cond else ""),
                         "crop_path": c.get("crop_path") or ""})
    return rows


def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _image_paths(arg):
    """--image may be a single file or a directory of images."""
    p = Path(arg)
    if p.is_dir():
        return sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS and not q.name.startswith("._"))
    return [p]


def run_basename(image_arg):
    """Name a run after its input: '<dir>_inference' for a folder, '<stem>_inference' for a file."""
    p = Path(image_arg)
    stem = p.name if p.is_dir() else p.stem
    return f"{stem}_inference"


def unique_run_dir(out_dir, image_arg):
    """A fresh run folder under out_dir so a new inference never overwrites a previous one:
    <input>_inference, then _inference2, _inference3, … if it already exists."""
    base = Path(out_dir) / run_basename(image_arg)
    if not base.exists():
        return base
    i = 2
    while (base.parent / f"{base.name}{i}").exists():
        i += 1
    return base.parent / f"{base.name}{i}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="an image file OR a directory of images")
    ap.add_argument("--pole-weights", required=True)
    ap.add_argument("--comp-above-weights", required=True)
    ap.add_argument("--comp-below-weights", required=True)
    # Each run is saved to its OWN folder so nothing is overwritten: <out-dir>/<input>_inference[N]/
    # with result.json, result.csv, crops/, and viz/{pole,components,conditions,all}/.
    ap.add_argument("--out-dir", default="runs/inference",
                    help="base dir; a per-run subfolder '<input>_inference' is created inside it")
    ap.add_argument("--crop-dir", default=None, help="override crop dir (default <run>/crops)")
    ap.add_argument("--out", default=None, help="override JSON path (default <run>/result.json)")
    ap.add_argument("--out-csv", default=None, help="override CSV path (default <run>/result.csv)")
    ap.add_argument("--viz-dir", default=None, help="override viz dir (default <run>/viz)")
    ap.add_argument("--no-viz", action="store_true", help="skip the 4 annotated-view images")
    ap.add_argument("--pole-pad", type=float, default=0.05)
    ap.add_argument("--pole-conf", type=float, default=0.12,   # recall-leaning (Stage-1: don't miss poles)
                    help="pole-stage confidence; low favors recall")
    ap.add_argument("--comp-conf", type=float, default=0.25,
                    help="confidence for both component detectors")
    ap.add_argument("--nms-iou", type=float, default=0.55,
                    help="cross-detector NMS IoU: one box per physical object (lower = more aggressive de-dup)")
    ap.add_argument("--no-nms", action="store_true", help="disable cross-detector de-duplication")
    ap.add_argument("--pole-imgsz", type=int, default=640)     # match pole training
    ap.add_argument("--comp-imgsz", type=int, default=1280)    # match component training
    ap.add_argument("--no-clahe", action="store_true",
                    help="skip CLAHE (only for models trained on the 'orig' variant)")
    # per-stage backend: mix and match (e.g. YOLO pole + RF-DETR components + FRCNN condition)
    ap.add_argument("--pole-backend", choices=BACKENDS, default="yolo")
    ap.add_argument("--comp-above-backend", choices=BACKENDS, default="yolo")
    ap.add_argument("--comp-below-backend", choices=BACKENDS, default="yolo")
    ap.add_argument("--rfdetr-resolution", type=int, default=672,
                    help="RF-DETR inference resolution for any rfdetr stage; must match training "
                         "(multiple of the model block_size; 672/896/1120 are safe)")
    ap.add_argument("--frcnn-min-size", type=int, default=2000,
                    help="Faster R-CNN inference min_size for any frcnn stage; MUST match the "
                         "training --min-size (default 2000) or small objects collapse")
    # stage 4 (optional): component condition classification, run on each component crop
    ap.add_argument("--condition-weights", default=None,
                    help="component_classification weights; if set, each component gets a 'conditions' list")
    ap.add_argument("--condition-backend", choices=BACKENDS, default="yolo")
    ap.add_argument("--condition-conf", type=float, default=0.25)
    ap.add_argument("--condition-imgsz", type=int, default=1280)
    a = ap.parse_args()
    # build the detectors ONCE, then run over every image (file or directory).
    pole_det = build_detector(a.pole_backend, a.pole_weights, a.pole_conf, a.pole_imgsz,
                              config.POLE_CLASSES, a.rfdetr_resolution, a.frcnn_min_size)
    above_det = build_detector(a.comp_above_backend, a.comp_above_weights, a.comp_conf,
                               a.comp_imgsz, config.COMPONENT_ABOVE_CLASSES, a.rfdetr_resolution,
                               a.frcnn_min_size)
    below_det = build_detector(a.comp_below_backend, a.comp_below_weights, a.comp_conf,
                               a.comp_imgsz, config.COMPONENT_BELOW_CLASSES, a.rfdetr_resolution,
                               a.frcnn_min_size)
    condition_det = (build_detector(a.condition_backend, a.condition_weights, a.condition_conf,
                                    a.condition_imgsz, config.COMPONENT_CLASSIFICATION_CLASSES,
                                    a.rfdetr_resolution, a.frcnn_min_size)
                     if a.condition_weights else None)

    # Per-run output folder named after the input; never overwrites a previous run.
    run_dir = unique_run_dir(a.out_dir, a.image)
    run_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = a.crop_dir or str(run_dir / "crops")
    out_json = a.out or str(run_dir / "result.json")
    out_csv = a.out_csv or str(run_dir / "result.csv")
    viz_dir = a.viz_dir or str(run_dir / "viz")

    paths = _image_paths(a.image)
    results, rows = [], []
    for p in paths:
        # EXIF-orient ONCE; CLAHE is applied for inference (pole runs on it and every crop
        # inherits the CLAHE), but we draw the viz on the un-CLAHE'd `oriented` frame (geometry
        # is identical, colors look natural). Per-image outputs are named '<stem>_inference'.
        oriented = load_oriented_bgr(str(p))
        image = clahe_image(oriented) if not a.no_clahe else oriented
        out_stem = f"{p.stem}_inference"
        result = run_pipeline(image, pole_det, above_det, below_det,
                              crop_dir, p.name, pole_pad=a.pole_pad,
                              condition_detector=condition_det, name_stem=out_stem,
                              nms_iou=(1.0 if a.no_nms else a.nms_iou))
        results.append(result)
        rows.extend(result_to_rows(result))
        if not a.no_viz:
            from inference.visualize import save_layers
            save_layers(oriented, result, viz_dir, out_stem)
        print(f"[{p.name}] poles={len(result['poles'])} "
              f"components={sum(len(x['components_above'])+len(x['components_below']) for x in result['poles'])}")

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(results if len(results) != 1 else results[0], indent=2))
    write_csv(rows, out_csv)
    print(f"\n== Run saved to: {run_dir} ==")
    print(f"  {len(rows)} component rows across {len(paths)} image(s)")
    print("  contents: result.json, result.csv, crops/" +
          ("" if a.no_viz else ", viz/{pole,components,conditions,all}/"))


if __name__ == "__main__":
    main()
