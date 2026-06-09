"""Run a single component/condition YOLO model on one image/crop or a directory. Usage:
    python -m inference.infer_components --image crop.jpg --weights component.pt [--out-csv out.csv]
Generic — point --weights at any component or the condition model. Applies the same EXIF-orient
+ CLAHE preprocessing as training (--no-clahe for an 'orig'-trained model). Defaults to imgsz
1280 to match component training (thin wires). Writes JSON and, with --out-csv, a flat CSV.
"""
from inference.infer_pole import run_cli


def main():
    run_cli(default_imgsz=1280)   # components/conditions need high res


if __name__ == "__main__":
    main()
