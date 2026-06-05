"""Three-stage inference: pole detect -> crop -> run BOTH the component_above_1000 and
component_below_1000 detectors on the pole crop -> remap boxes to the full frame -> crop
each component -> structured JSON. Usage:
    python -m inference.pipeline --image path.jpg \
        --pole-weights pole.pt --comp-above-weights above.pt --comp-below-weights below.pt
"""
import argparse
import json
from pathlib import Path

import cv2

from inference.backends import YoloDetector
from inference.geometry import crop_with_pad, shift_detection
from data_prep.preprocess import load_oriented_bgr, clahe_image


def _save_crop(crop, crop_dir, name):
    if crop_dir is None or crop.size == 0:
        return None
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    path = crop_dir / name
    cv2.imwrite(str(path), crop)
    return str(path)


def _detect_on_crop(image, pole_crop, offset, detector, crop_dir, prefix):
    """Run `detector` on the pole crop, remap each detection to the full frame, save its crop."""
    ox, oy = offset
    out = []
    for ci, comp in enumerate(detector.predict(pole_crop)):
        full = shift_detection(comp, ox, oy)
        comp_crop, _ = crop_with_pad(image, full.box, pad_frac=0.0)
        out.append({
            "class": comp.class_name,
            "confidence": comp.confidence,
            "box_crop": [int(v) for v in comp.box],
            "box_full": [int(v) for v in full.box],
            "crop_path": _save_crop(comp_crop, crop_dir, f"{prefix}_comp{ci}.jpg"),
        })
    return out


def run_pipeline(image, pole_detector, above_detector, below_detector,
                 crop_dir, image_name, pole_pad=0.05):
    """image: BGR ndarray. Returns the structured result dict. Both component detectors run
    on the padded pole crop; no pole -> no component detection."""
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
                                                crop_dir, f"{stem}_pole{pi}_above"),
            "components_below": _detect_on_crop(image, pole_crop, offset, below_detector,
                                                crop_dir, f"{stem}_pole{pi}_below"),
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
    a = ap.parse_args()
    # EXIF-orient + CLAHE once on the full frame: pole runs on it, and every pole crop
    # inherits the CLAHE, so both component models also see their trained distribution.
    image = load_oriented_bgr(a.image)
    if not a.no_clahe:
        image = clahe_image(image)
    pole_det = YoloDetector(a.pole_weights, conf=a.pole_conf, imgsz=a.pole_imgsz)
    above_det = YoloDetector(a.comp_above_weights, conf=a.comp_conf, imgsz=a.comp_imgsz)
    below_det = YoloDetector(a.comp_below_weights, conf=a.comp_conf, imgsz=a.comp_imgsz)
    result = run_pipeline(image, pole_det, above_det, below_det,
                          a.crop_dir, Path(a.image).name, pole_pad=a.pole_pad)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
