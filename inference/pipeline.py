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
import json
from pathlib import Path

import cv2

from inference.backends import YoloDetector, RFDetrDetector, TorchvisionDetector
from inference.geometry import crop_with_pad, shift_detection
from data_prep.preprocess import load_oriented_bgr, clahe_image

BACKENDS = ["yolo", "rfdetr", "frcnn"]


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


def _classify_condition(comp_crop, condition_detector):
    """Stage 4: run the 14-class condition detector on a single component crop."""
    if condition_detector is None or comp_crop.size == 0:
        return None
    return [{"class": c.class_name, "confidence": c.confidence,
             "box_comp": [int(v) for v in c.box]}
            for c in condition_detector.predict(comp_crop)]


def _detect_on_crop(image, pole_crop, offset, detector, crop_dir, prefix,
                    condition_detector=None):
    """Run `detector` on the pole crop, remap each detection to the full frame, save its crop,
    and (if a condition detector is given) classify each component crop's condition."""
    ox, oy = offset
    out = []
    for ci, comp in enumerate(detector.predict(pole_crop)):
        full = shift_detection(comp, ox, oy)
        comp_crop, _ = crop_with_pad(image, full.box, pad_frac=0.0)
        entry = {
            "class": comp.class_name,
            "confidence": comp.confidence,
            "box_crop": [int(v) for v in comp.box],
            "box_full": [int(v) for v in full.box],
            "crop_path": _save_crop(comp_crop, crop_dir, f"{prefix}_comp{ci}.jpg"),
        }
        conditions = _classify_condition(comp_crop, condition_detector)
        if conditions is not None:
            entry["conditions"] = conditions
        out.append(entry)
    return out


def run_pipeline(image, pole_detector, above_detector, below_detector,
                 crop_dir, image_name, pole_pad=0.05, condition_detector=None):
    """image: BGR ndarray. Returns the structured result dict. Both component detectors run
    on the padded pole crop; no pole -> no component detection. If condition_detector is
    given, each detected component also carries a 'conditions' list (stage 4)."""
    stem = Path(image_name).stem
    result = {"image": image_name, "poles": []}
    for pi, pole in enumerate(pole_detector.predict(image)):
        pole_crop, offset = crop_with_pad(image, pole.box, pad_frac=pole_pad)
        pole_crop_path = _save_crop(pole_crop, crop_dir, f"{stem}_pole{pi}.jpg")
        result["poles"].append({
            "box": [int(v) for v in pole.box],
            "confidence": pole.confidence,
            "crop_path": pole_crop_path,
            "components_above": _detect_on_crop(image, pole_crop, offset, above_detector,
                                                crop_dir, f"{stem}_pole{pi}_above",
                                                condition_detector),
            "components_below": _detect_on_crop(image, pole_crop, offset, below_detector,
                                                crop_dir, f"{stem}_pole{pi}_below",
                                                condition_detector),
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--pole-weights", required=True)
    ap.add_argument("--comp-above-weights", required=True)
    ap.add_argument("--comp-below-weights", required=True)
    ap.add_argument("--crop-dir", default="runs/inference/crops")
    ap.add_argument("--out", default="runs/inference/result.json")
    ap.add_argument("--pole-pad", type=float, default=0.05)
    ap.add_argument("--pole-conf", type=float, default=0.12,   # recall-leaning (Stage-1: don't miss poles)
                    help="pole-stage confidence; low favors recall")
    ap.add_argument("--comp-conf", type=float, default=0.25,
                    help="confidence for both component detectors")
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
    # EXIF-orient + CLAHE once on the full frame: pole runs on it, and every pole crop
    # inherits the CLAHE, so both component models also see their trained distribution.
    image = load_oriented_bgr(a.image)
    if not a.no_clahe:
        image = clahe_image(image)
    from shared import config
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
    result = run_pipeline(image, pole_det, above_det, below_det,
                          a.crop_dir, Path(a.image).name, pole_pad=a.pole_pad,
                          condition_detector=condition_det)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
