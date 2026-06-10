"""Empirical study of the condition normal-vs-defect conflict on the VAL crops.

For each cond_* specialist we run the model on its val/clahe crops (GT-box at 0.25 == the model's
training distribution), and compare model predictions to ground truth to answer:
  * how often does the model flag a CLEAN (GT-normal) component as defective  -> wasted operator time
  * how often does it catch a REAL defect (recall)                            -> safety
across confidence thresholds, plus per-defect-class FP/TP confidences and a normal-vs-defect margin
analysis. This tells us how to fix the defect-priority rule (threshold / margin / per-class).

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.condition_conflict_study
"""
import glob
import os

from shared import config

DB = os.environ.get("DRONISIGHT_DATA", "/Volumes/dronisight") + "/yolo_train_db"
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
IMGSZ = 1280
DETECT_CONF = 0.05   # capture everything; we threshold offline


def is_normal(name):
    return name.endswith("_normal")


def gt_classes(label_path, names):
    out = set()
    for line in open(label_path).read().splitlines():
        if line.strip():
            out.add(names[int(line.split()[0])])
    return out


def study_subset(subset):
    from ultralytics import YOLO
    names = config.SUBSET_CLASSES[subset]
    defects = [n for n in names if not is_normal(n)]
    w = sorted(glob.glob(f"models/runs/{subset}/**/weights/best.pt", recursive=True))
    if not w:
        return None
    model = YOLO(w[-1])
    imgs = sorted(glob.glob(f"{DB}/{subset}/images/val/clahe/*.jpg"))

    rows = []   # per crop: dict(gt_defective, defect_conf{cls:maxconf}, normal_conf)
    for img in imgs:
        lbl = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl):
            continue
        gt = gt_classes(lbl, names)
        gt_def = {g for g in gt if not is_normal(g)}
        res = model.predict(img, imgsz=IMGSZ, conf=DETECT_CONF, device="mps", verbose=False)[0]
        dconf, nconf = {}, 0.0
        for b in res.boxes:
            nm = res.names[int(b.cls[0])]
            c = float(b.conf[0])
            if is_normal(nm):
                nconf = max(nconf, c)
            else:
                dconf[nm] = max(dconf.get(nm, 0.0), c)
        rows.append({"gt_defective": bool(gt_def), "gt_def": gt_def, "dconf": dconf, "nconf": nconf})
    return {"subset": subset, "names": names, "defects": defects, "rows": rows}


def confusion_at(rows, T):
    """defect-priority decision = 'flag defective' iff any defect predicted with conf >= T."""
    tp = fp = fn = tn = 0
    for r in rows:
        flagged = any(c >= T for c in r["dconf"].values())
        if r["gt_defective"]:
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (fp + tn) if (fp + tn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    return tp, fp, fn, tn, rec, far, prec


def main():
    studies = []
    for s in config.COND_SUBSETS:
        st = study_subset(s)
        if st:
            studies.append(st)
            print(f"scanned {s}: {len(st['rows'])} val crops")

    print("\n" + "=" * 96)
    print("A) FLAG-DEFECTIVE confusion at each confidence threshold (defect-priority decision)")
    print("   FAR = clean components wrongly flagged defective (operator-time waste); REC = real defects caught")
    print("=" * 96)
    allrows = [r for st in studies for r in st["rows"]]
    for label, rows in [("ALL", allrows)] + [(st["subset"], st["rows"]) for st in studies]:
        ndef = sum(r["gt_defective"] for r in rows)
        nclean = len(rows) - ndef
        print(f"\n{label}  (clean={nclean}, defective={ndef})")
        print(f"  {'T':>5} {'recall':>7} {'falseAlarm':>11} {'precision':>10}   (TP/FP/FN/TN)")
        for T in THRESHOLDS:
            tp, fp, fn, tn, rec, far, prec = confusion_at(rows, T)
            print(f"  {T:>5.2f} {rec:>7.2%} {far:>11.2%} {prec:>10.2%}   ({tp}/{fp}/{fn}/{tn})")

    print("\n" + "=" * 96)
    print("B) PER-DEFECT-CLASS: confidence of the defect on crops that DO vs DON'T truly have it")
    print("   (high FP-conf = the model genuinely confuses it; low FP-conf = a threshold fixes it)")
    print("=" * 96)
    for st in studies:
        for d in st["defects"]:
            pos = [r["dconf"].get(d, 0.0) for r in st["rows"] if d in r["gt_def"]]
            neg = [r["dconf"].get(d, 0.0) for r in st["rows"] if d not in r["gt_def"]]
            pos_hit = [c for c in pos if c > 0]
            for T in (0.25, 0.40):
                fp = sum(c >= T for c in neg)
                tp = sum(c >= T for c in pos)
                print(f"  {d:28s} T={T:.2f}  TP={tp:3d}/{len(pos):3d}  FP={fp:3d}/{len(neg):3d}", end="")
            mean_fp = sum(c for c in neg if c > 0) / max(1, sum(c > 0 for c in neg))
            print(f"   mean-FP-conf={mean_fp:.2f}")

    print("\n" + "=" * 96)
    print("C) MARGIN on CONFLICT crops (both normal>=0.25 AND a defect>=0.25 predicted)")
    print("   compares normal_conf vs top defect_conf, split by ground truth")
    print("=" * 96)
    for label, rows in [("ALL", allrows)]:
        clean_margins, def_margins = [], []
        for r in rows:
            topd = max(r["dconf"].values(), default=0.0)
            if r["nconf"] >= 0.25 and topd >= 0.25:                 # a genuine conflict
                (def_margins if r["gt_defective"] else clean_margins).append(r["nconf"] - topd)

        def summ(xs):
            if not xs:
                return "n=0"
            xs = sorted(xs)
            return (f"n={len(xs)} mean(normal-defect)={sum(xs)/len(xs):+.2f} "
                    f"median={xs[len(xs)//2]:+.2f} (defect wins margin<0)")
        print(f"  conflict crops, GT CLEAN (normal right):    {summ(clean_margins)}")
        print(f"  conflict crops, GT DEFECTIVE (defect right): {summ(def_margins)}")


if __name__ == "__main__":
    main()
