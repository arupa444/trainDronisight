"""Run a single YOLO model (e.g. the pole detector) on one image or a directory. Usage:
    python -m inference.infer_pole --image x.jpg --weights pole.pt [--out-csv out.csv]
Applies the same EXIF-orient + CLAHE preprocessing as training (use --no-clahe only for a
model trained on the 'orig' variant). Defaults to imgsz 640 to match pole training.
Writes JSON (per-image detections) and, with --out-csv, a flat CSV (one row per detection).
"""
import argparse
import csv
import json
from pathlib import Path

from inference.backends import YoloDetector
from data_prep.preprocess import load_oriented_bgr, clahe_image

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SOLO_CSV_COLUMNS = ["image", "class", "confidence", "x1", "y1", "x2", "y2"]


def detections_to_records(dets):
    return [{"class": d.class_name, "confidence": d.confidence,
             "box": [int(v) for v in d.box]} for d in dets]


def image_paths(arg):
    """--image may be a single file or a directory of images."""
    p = Path(arg)
    if p.is_dir():
        return sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS and not q.name.startswith("._"))
    return [p]


def run_solo(weights, image_arg, conf, imgsz, no_clahe):
    """Run one YOLO model over a file/dir. Returns (results, csv_rows)."""
    det = YoloDetector(weights, conf=conf, imgsz=imgsz)
    results, rows = [], []
    for p in image_paths(image_arg):
        img = load_oriented_bgr(str(p))
        if not no_clahe:
            img = clahe_image(img)
        recs = detections_to_records(det.predict(img))
        results.append({"image": p.name, "detections": recs})
        for r in recs:
            b = r["box"]
            rows.append({"image": p.name, "class": r["class"], "confidence": round(r["confidence"], 4),
                         "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]})
    return results, rows


def write_solo_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SOLO_CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def add_solo_args(ap, default_imgsz):
    ap.add_argument("--image", required=True, help="image file OR directory of images")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=default_imgsz)
    ap.add_argument("--no-clahe", action="store_true",
                    help="skip CLAHE (only for models trained on the 'orig' variant)")
    ap.add_argument("--out", default=None, help="write per-image detections JSON here")
    ap.add_argument("--out-csv", default=None, help="write a flat per-detection CSV here")
    return ap


def run_cli(default_imgsz):
    a = add_solo_args(argparse.ArgumentParser(), default_imgsz).parse_args()
    results, rows = run_solo(a.weights, a.image, a.conf, a.imgsz, a.no_clahe)
    out = results if len(results) != 1 else results[0]
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
    if a.out_csv:
        write_solo_csv(rows, a.out_csv)
        print(f"wrote {len(rows)} detection rows -> {a.out_csv}")
    print(json.dumps(out, indent=2))


def main():
    run_cli(default_imgsz=640)   # pole fills the frame -> 640 matches training


if __name__ == "__main__":
    main()
