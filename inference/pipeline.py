"""Multi-specialist inference for the 12-model architecture:
  Stage 1  pole          : 1 detector on the full frame -> pole boxes.
  Stage 2  components     : the 5 component specialists (comp_wire, comp_insulator, comp_crossarm,
                            comp_vegetation, comp_rust) ALL run on the padded pole crop; their boxes
                            are remapped to the full frame and de-duplicated with class-agnostic NMS
                            (one box per physical object).
  Stage 3  conditions     : each surviving component is ROUTED to its condition specialist via
                            config.COMPONENT_TO_CONDITION_MODEL (v_insulator -> cond_v_insulator, etc.;
                            vegetation/rust have no condition family). That specialist runs on a crop of
                            the component padded by config.CONDITION_CROP_PAD (== how cond_* was built),
                            and only in-family condition classes are kept.
  -> structured JSON + flat result.csv + 4 annotated views.

The 11 trainable weights are auto-discovered by subset name under --weights-dir (pole is reused):
    python -m inference.pipeline --image path_or_dir --weights-dir runs/
Each stage's backend is independently selectable via --backend {yolo|rfdetr|frcnn}.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2

from shared import config
from inference.backends import YoloDetector, RFDetrDetector, TorchvisionDetector, FilteredDetector
from inference.geometry import crop_with_pad, shift_detection
from data_prep.preprocess import load_oriented_bgr, clahe_image

BACKENDS = ["yolo", "rfdetr", "frcnn"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def build_detector(backend, weights, conf, imgsz, class_names, resolution=672, frcnn_min_size=2000,
                   device=None):
    """Pick a detector backend for a pipeline stage. YOLO reads class names from the model;
    RF-DETR and Faster R-CNN need the explicit class_names (they return numeric class ids).
    `device` follows CUDA -> MPS -> CPU (shared.device.select_device when None). frcnn_min_size
    MUST match the Faster R-CNN training --min-size (default 2000) or small objects are served
    at the wrong scale."""
    if backend == "rfdetr":
        return RFDetrDetector(weights, class_names, conf=conf, resolution=resolution)
    if backend == "frcnn":
        return TorchvisionDetector(weights, class_names, conf=conf, min_size=frcnn_min_size, device=device)
    return YoloDetector(weights, conf=conf, imgsz=imgsz, device=device)


def discover_weights(weights_dir, subsets):
    """Map each subset -> its trained best.pt under weights_dir, found by subset NAME anywhere in the
    tree (e.g. runs/<subset>/yolo/weights/best.pt, or the nested runs/detect/runs/<subset>/... that
    Ultralytics writes). Picks the most-recent best.pt (falls back to last.pt). Missing subsets are
    simply absent from the returned dict (the caller decides whether that's fatal)."""
    wd = Path(weights_dir)
    found = {}
    for s in subsets:
        cands = sorted(wd.glob(f"**/{s}/**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
        if not cands:
            cands = sorted(wd.glob(f"**/{s}/**/weights/last.pt"), key=lambda p: p.stat().st_mtime)
        if cands:
            found[s] = str(cands[-1])
    return found


def build_component_detector(subset, weights_dir, weights_map, backend, conf, imgsz, device=None):
    """Build the component detector for `subset`, honoring config.COMPONENT_WEIGHTS_OVERRIDE: if an
    override exists AND its weights are discoverable under weights_dir, use those weights wrapped in a
    FilteredDetector that keeps only the override's classes (e.g. comp_vegetation -> the below_1000
    detector, keep just `vegetation`). Otherwise use the slot's own weights. Returns a Detector, or
    None if no usable weights were found."""
    override = config.COMPONENT_WEIGHTS_OVERRIDE.get(subset)
    if override:
        ow = discover_weights(weights_dir, [override["weights_subset"]]).get(override["weights_subset"])
        if ow:
            inner = build_detector(backend, ow, conf, imgsz, config.SUBSET_CLASSES.get(subset, []),
                                   device=device)
            return FilteredDetector(inner, override["keep"])
    w = weights_map.get(subset)
    if w:
        return build_detector(backend, w, conf, imgsz, config.SUBSET_CLASSES.get(subset, []), device=device)
    return None


def _save_crop(crop, crop_dir, name):
    if crop_dir is None or crop.size == 0:
        return None
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    path = crop_dir / name
    cv2.imwrite(str(path), crop)
    return str(path)


def _classify_condition(comp_crop, condition_detector, component_class,
                        condition_ios=config.CONDITION_OVERLAP_IOS):
    """Stage 3: run the ROUTED condition detector on ONE component crop, keep only the in-family
    condition classes (config.COMPONENT_TO_CONDITIONS), then resolve normal-vs-damage overlaps
    (resolve_condition_overlaps). Returns (best, all_valid):
      * best = {"class","confidence"} highest-confidence surviving condition, or None
      * all_valid = every surviving in-family condition (sorted), or None if no family / no detector.
    Components with no condition family (vegetation, rust) -> (None, None)."""
    allowed = config.COMPONENT_TO_CONDITIONS.get(component_class)
    if condition_detector is None or allowed is None or comp_crop.size == 0:
        return None, None
    allowed = set(allowed)
    dets = [c for c in condition_detector.predict(comp_crop) if c.class_name in allowed]
    dets = resolve_condition_overlaps(dets, condition_ios)            # normal vs damage conflict resolution
    dets = sorted(dets, key=lambda c: c.confidence, reverse=True)
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


def _ios_xyxy(a, b):
    """Intersection over the SMALLER box's area — catches containment (a small band box inside a
    large normal box) that plain IoU misses."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def nms_components(items, iou_thresh, same_class_iou=None):
    """Class-AWARE greedy NMS so each PHYSICAL object yields ONE box. items: list of
    (comp_det_in_crop_coords, full_det). Keeps the highest-confidence detection and drops a later box
    if it overlaps a kept box by >= the relevant threshold: SAME class uses `same_class_iou` (lower ->
    aggressively removes duplicates, e.g. two boxes on one insulator), DIFFERENT classes use the
    higher `iou_thresh` (so a wire crossing a crossarm survives)."""
    same_class_iou = iou_thresh if same_class_iou is None else same_class_iou
    items = sorted(items, key=lambda t: t[1].confidence, reverse=True)
    kept = []
    for it in items:
        ok = True
        for k in kept:
            thr = same_class_iou if it[1].class_name == k[1].class_name else iou_thresh
            if _iou_xyxy(it[1].box, k[1].box) >= thr:
                ok = False
                break
        if ok:
            kept.append(it)
    return kept


def nms_detections(dets, iou_thresh):
    """Greedy IoU NMS over a plain Detection list (used for the single-class pole stage): keep the
    highest-confidence box, drop later boxes overlapping a kept one — removes duplicate poles."""
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept = []
    for d in dets:
        if all(_iou_xyxy(d.box, k.box) < iou_thresh for k in kept):
            kept.append(d)
    return kept


def resolve_condition_overlaps(dets, ios_thresh):
    """Resolve normal-vs-damage conflicts among a component's condition detections (Detection list,
    crop coords). Greedy by confidence using intersection-over-smaller (so a small band box inside a
    big normal box counts as overlapping). A box is DROPPED if it overlaps an already-kept box and
    either (a) is the same class (duplicate), or (b) has the OPPOSITE normal/damage polarity. Net:
      * normal wins (highest conf) -> overlapping damages dropped -> [normal]
      * a damage wins -> the overlapping normal dropped, ALL overlapping damages kept (band + broken)."""
    def is_normal(d):
        return d.class_name.endswith("_normal")
    kept = []
    for d in sorted(dets, key=lambda d: d.confidence, reverse=True):
        drop = False
        for k in kept:
            if _ios_xyxy(d.box, k.box) >= ios_thresh and (
                    d.class_name == k.class_name or is_normal(d) != is_normal(k)):
                drop = True
                break
        if not drop:
            kept.append(d)
    return kept


def run_pipeline(image, pole_detector, component_detectors, condition_detectors=None,
                 crop_dir=None, image_name="image.jpg", pole_pad=0.05, name_stem=None,
                 nms_iou=config.COMPONENT_NMS_IOU, condition_pad=config.CONDITION_CROP_PAD,
                 component_pad=config.COMPONENT_CROP_PAD, pole_nms_iou=config.POLE_NMS_IOU,
                 same_class_iou=config.COMPONENT_SAME_CLASS_IOU,
                 condition_ios=config.CONDITION_OVERLAP_IOS):
    """image: BGR ndarray. Returns the structured result dict.
      * component_detectors: an iterable of the component specialists; ALL run on the padded pole crop,
        their boxes are remapped to the full frame and de-duplicated with class-agnostic NMS (nms_iou;
        set >=1.0 to disable) so each object gets ONE box.
      * condition_detectors: dict {cond_subset_name: detector}. Each surviving component is routed by
        config.COMPONENT_TO_CONDITION_MODEL to its specialist (if present); that specialist sees a crop
        padded by `condition_pad` (matching how the cond_* datasets were built).
    `name_stem` overrides saved-crop filenames."""
    stem = name_stem or Path(image_name).stem
    condition_detectors = condition_detectors or {}
    result = {"image": image_name, "poles": []}
    poles = pole_detector.predict(image)
    if pole_nms_iou and pole_nms_iou < 1.0:        # drop duplicate boxes on the same pole
        poles = nms_detections(poles, pole_nms_iou)
    for pi, pole in enumerate(poles):
        pole_crop, (ox, oy) = crop_with_pad(image, pole.box, pad_frac=pole_pad)
        pole_crop_path = _save_crop(pole_crop, crop_dir, f"{stem}_pole{pi}.jpg")
        # every component specialist on the pole crop -> remap to full frame -> cross-detector NMS
        combined = [(c, shift_detection(c, ox, oy))
                    for det in component_detectors for c in det.predict(pole_crop)]
        if nms_iou and nms_iou < 1.0:
            combined = nms_components(combined, nms_iou, same_class_iou)
        comps_out = []
        for ci, (comp, full) in enumerate(combined):
            comp_crop, _ = crop_with_pad(image, full.box, pad_frac=component_pad)  # small pad, for the saved/display crop (band not clipped)
            cond_crop, _ = crop_with_pad(image, full.box, pad_frac=condition_pad)  # 0.25 pad, fed to the condition model (matches training)
            entry = {
                "class": comp.class_name,
                "confidence": comp.confidence,
                "box_crop": [int(v) for v in comp.box],
                "box_full": [int(v) for v in full.box],
                "crop_path": _save_crop(comp_crop, crop_dir, f"{stem}_pole{pi}_comp{ci}_{comp.class_name}.jpg"),
            }
            # route this component to its condition specialist (None for vegetation/rust)
            model_name = config.COMPONENT_TO_CONDITION_MODEL.get(comp.class_name)
            cond_det = condition_detectors.get(model_name)
            best, all_valid = _classify_condition(cond_crop, cond_det, comp.class_name, condition_ios)
            if all_valid is not None:               # condition stage ran for this component family
                entry["condition"] = best           # the in-family top condition (or None)
                entry["conditions"] = all_valid       # all in-family condition detections
            comps_out.append(entry)
        result["poles"].append({
            "box": [int(v) for v in pole.box],
            "confidence": pole.confidence,
            "crop_path": pole_crop_path,
            "components": comps_out,
        })
    return result


CSV_COLUMNS = ["image", "pole_index", "pole_confidence", "pole_x1", "pole_y1", "pole_x2", "pole_y2",
               "component_class", "component_confidence",
               "comp_x1", "comp_y1", "comp_x2", "comp_y2",
               "condition_class", "condition_confidence", "crop_path"]


def result_to_rows(result):
    """Flatten one pipeline result dict into CSV rows: ONE row per detected component (its routed
    condition inline), plus a component-less row for any pole with no components."""
    rows = []
    img = result["image"]
    for pi, pole in enumerate(result["poles"]):
        px = pole["box"]
        base = {"image": img, "pole_index": pi, "pole_confidence": round(pole["confidence"], 4),
                "pole_x1": px[0], "pole_y1": px[1], "pole_x2": px[2], "pole_y2": px[3]}
        comps = pole.get("components", [])
        if not comps:
            rows.append({**base, "component_class": "", "component_confidence": "",
                         "comp_x1": "", "comp_y1": "", "comp_x2": "", "comp_y2": "",
                         "condition_class": "", "condition_confidence": "",
                         "crop_path": pole.get("crop_path") or ""})
            continue
        for c in comps:
            b = c["box_full"]
            cond = c.get("condition")
            rows.append({**base,
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
    ap.add_argument("--weights-dir", required=True,
                    help="folder holding the trained runs/; each subset's best.pt is auto-discovered by name")
    # Each run is saved to its OWN folder so nothing is overwritten: <out-dir>/<input>_inference[N]/
    # with result.json, result.csv, crops/, and viz/{pole,components,conditions,all}/.
    ap.add_argument("--out-dir", default="runs/inference",
                    help="base dir; a per-run subfolder '<input>_inference' is created inside it")
    ap.add_argument("--crop-dir", default=None, help="override crop dir (default <run>/crops)")
    ap.add_argument("--out", default=None, help="override JSON path (default <run>/result.json)")
    ap.add_argument("--out-csv", default=None, help="override CSV path (default <run>/result.csv)")
    ap.add_argument("--viz-dir", default=None, help="override viz dir (default <run>/viz)")
    ap.add_argument("--no-viz", action="store_true", help="skip the 4 annotated-view images")
    ap.add_argument("--no-condition", action="store_true", help="skip the condition stage entirely")
    ap.add_argument("--pole-pad", type=float, default=config.POLE_CROP_PAD,
                    help="pole-crop padding; MUST match config.POLE_CROP_PAD used to build the comp_* datasets")
    ap.add_argument("--condition-pad", type=float, default=config.CONDITION_CROP_PAD,
                    help="padding around each component when cropping it for the condition model; "
                         "MUST match config.CONDITION_CROP_PAD used to build the cond_* datasets")
    ap.add_argument("--component-pad", type=float, default=config.COMPONENT_CROP_PAD,
                    help="padding for the SAVED component crop/thumbnail only (so edge bands aren't "
                         "clipped from view); does NOT change what the condition model is fed")
    ap.add_argument("--pole-conf", type=float, default=0.12,   # recall-leaning (Stage-1: don't miss poles)
                    help="pole-stage confidence; low favors recall")
    ap.add_argument("--comp-conf", type=float, default=0.25, help="confidence for the component detectors")
    ap.add_argument("--cond-conf", type=float, default=0.25, help="confidence for the condition detectors")
    ap.add_argument("--nms-iou", type=float, default=config.COMPONENT_NMS_IOU,
                    help="DIFFERENT-class component overlap IoU (wire crossing a crossarm kept unless above this)")
    ap.add_argument("--same-class-iou", type=float, default=config.COMPONENT_SAME_CLASS_IOU,
                    help="SAME-class component duplicate IoU (lower = drop duplicates more aggressively)")
    ap.add_argument("--pole-nms-iou", type=float, default=config.POLE_NMS_IOU,
                    help="pole duplicate IoU (two boxes on one pole -> keep highest confidence)")
    ap.add_argument("--condition-ios", type=float, default=config.CONDITION_OVERLAP_IOS,
                    help="condition normal-vs-damage overlap (intersection-over-smaller) to resolve conflicts")
    ap.add_argument("--no-nms", action="store_true", help="disable ALL de-duplication (pole + component)")
    ap.add_argument("--pole-imgsz", type=int, default=640)     # match pole training
    ap.add_argument("--comp-imgsz", type=int, default=1280)    # match component training
    ap.add_argument("--cond-imgsz", type=int, default=1280)    # match condition training
    ap.add_argument("--no-clahe", action="store_true",
                    help="skip CLAHE (only for models trained on the 'orig' variant)")
    ap.add_argument("--backend", choices=BACKENDS, default="yolo", help="detector backend for ALL stages")
    ap.add_argument("--rfdetr-resolution", type=int, default=672,
                    help="RF-DETR inference resolution (multiple of the model block_size; 672/896/1120)")
    ap.add_argument("--frcnn-min-size", type=int, default=2000,
                    help="Faster R-CNN inference min_size; MUST match training --min-size (default 2000)")
    a = ap.parse_args()

    # Auto-discover the 12 weights by subset name; pole is required, components/conditions optional.
    weights = discover_weights(a.weights_dir, config.SUBSETS)
    if "pole" not in weights:
        raise SystemExit(f"No pole weights found under {a.weights_dir} (looked for **/pole/**/weights/best.pt)")
    print("Discovered weights:")
    for s in config.SUBSETS:
        print(f"  {'OK ' if s in weights else 'MISSING'} {s:24s} {weights.get(s,'')}")

    def mk(subset, conf, imgsz):
        return build_detector(a.backend, weights[subset], conf, imgsz, config.SUBSET_CLASSES[subset],
                              a.rfdetr_resolution, a.frcnn_min_size)

    pole_det = build_detector(a.backend, weights["pole"], a.pole_conf, a.pole_imgsz,
                              config.POLE_CLASSES, a.rfdetr_resolution, a.frcnn_min_size)
    component_dets = [d for d in (build_component_detector(s, a.weights_dir, weights, a.backend,
                                                           a.comp_conf, a.comp_imgsz)
                                  for s in config.COMP_SUBSETS) if d is not None]
    condition_dets = ({} if a.no_condition
                      else {s: mk(s, a.cond_conf, a.cond_imgsz) for s in config.COND_SUBSETS if s in weights})

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
        # EXIF-orient ONCE; CLAHE is applied for inference (pole runs on it and every crop inherits
        # the CLAHE), but viz is drawn on the un-CLAHE'd `oriented` frame (identical geometry, natural
        # colors). Per-image outputs are named '<stem>_inference'.
        oriented = load_oriented_bgr(str(p))
        image = clahe_image(oriented) if not a.no_clahe else oriented
        out_stem = f"{p.stem}_inference"
        result = run_pipeline(image, pole_det, component_dets, condition_dets,
                              crop_dir=crop_dir, image_name=p.name, pole_pad=a.pole_pad,
                              name_stem=out_stem, nms_iou=(1.0 if a.no_nms else a.nms_iou),
                              condition_pad=a.condition_pad, component_pad=a.component_pad,
                              pole_nms_iou=(1.0 if a.no_nms else a.pole_nms_iou),
                              same_class_iou=a.same_class_iou, condition_ios=a.condition_ios)
        results.append(result)
        rows.extend(result_to_rows(result))
        if not a.no_viz:
            from inference.visualize import save_layers
            save_layers(oriented, result, viz_dir, out_stem)
        print(f"[{p.name}] poles={len(result['poles'])} "
              f"components={sum(len(x['components']) for x in result['poles'])}")

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(results if len(results) != 1 else results[0], indent=2))
    write_csv(rows, out_csv)
    print(f"\n== Run saved to: {run_dir} ==")
    print(f"  {len(rows)} component rows across {len(paths)} image(s)")
    print("  contents: result.json, result.csv, crops/" +
          ("" if a.no_viz else ", viz/{pole,components,conditions,all}/"))


if __name__ == "__main__":
    main()
