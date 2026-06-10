"""Does ensembling the OLD unified 14-class condition classifier with each new family specialist
catch the defects the specialist misses? Runs both on every val crop and compares, per family, the
'flag defective' recall + false-alarm for: specialist-only, unified-only, and ensemble (defect if
EITHER fires). Reference for fixing 'broken/chipped insulators called normal'.

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.condition_ensemble_study
"""
import glob
import os

from shared import config

DB = os.environ.get("DRONISIGHT_DATA", "/Volumes/dronisight") + "/yolo_train_db"
UNIFIED = "/Volumes/dronisight/runs/component_classification_crop/yolo/weights/best.pt"
THRESHOLDS = [0.25, 0.35, 0.45]
IMGSZ = 1280


def is_normal(n):
    return n.endswith("_normal")


def gt_classes(label_path, names):
    return {names[int(l.split()[0])] for l in open(label_path).read().splitlines() if l.strip()}


def defect_max(res, allowed_defects):
    m = 0.0
    for b in res.boxes:
        nm = res.names[int(b.cls[0])]
        if nm in allowed_defects:
            m = max(m, float(b.conf[0]))
    return m


def main():
    from ultralytics import YOLO
    uni = YOLO(UNIFIED)
    print(f"unified classes: {len(uni.names)}\n")

    agg = {}  # strategy -> list of (gt_defective, flagged_at_T dict)
    per_subset = {}
    for s in config.COND_SUBSETS:
        names = config.SUBSET_CLASSES[s]
        defects = {n for n in names if not is_normal(n)}
        w = sorted(glob.glob(f"models/runs/{s}/**/weights/best.pt", recursive=True))
        if not w:
            continue
        spec = YOLO(w[-1])
        imgs = sorted(glob.glob(f"{DB}/{s}/images/val/clahe/*.jpg"))
        recs = []
        for img in imgs:
            lbl = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            if not os.path.exists(lbl):
                continue
            gt_def = bool({g for g in gt_classes(lbl, names) if not is_normal(g)})
            rs = spec.predict(img, imgsz=IMGSZ, conf=0.05, device="mps", verbose=False)[0]
            ru = uni.predict(img, imgsz=IMGSZ, conf=0.05, device="mps", verbose=False)[0]
            sd = defect_max(rs, defects)
            ud = defect_max(ru, defects)               # unified, restricted to THIS family's defects
            recs.append({"gt": gt_def, "spec": sd, "uni": ud, "ens": max(sd, ud)})
        per_subset[s] = recs
        agg.setdefault("rows", []).extend(recs)

    def report(label, recs):
        ndef = sum(r["gt"] for r in recs)
        nclean = len(recs) - ndef
        print(f"\n{label}  (clean={nclean}, defective={ndef})")
        print(f"  {'T':>5} | {'specialist':>20} | {'unified':>20} | {'ENSEMBLE':>20}")
        print(f"  {'':>5} | {'recall  falseAlarm':>20} | {'recall  falseAlarm':>20} | {'recall  falseAlarm':>20}")
        for T in THRESHOLDS:
            cells = []
            for k in ("spec", "uni", "ens"):
                tp = sum(r["gt"] and r[k] >= T for r in recs)
                fp = sum((not r["gt"]) and r[k] >= T for r in recs)
                rec = tp / ndef if ndef else float("nan")
                far = fp / nclean if nclean else float("nan")
                cells.append(f"{rec:>6.0%} {far:>10.0%}")
            print(f"  {T:>5.2f} | {cells[0]:>20} | {cells[1]:>20} | {cells[2]:>20}")

    report("ALL", agg["rows"])
    for s in config.COND_SUBSETS:
        if s in per_subset:
            report(s, per_subset[s])


if __name__ == "__main__":
    main()
