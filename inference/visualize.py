"""Draw pipeline results onto the full frame as four layered views:
  pole        - just the pole box(es)
  components  - the component boxes (5 specialists, post-NMS, remapped to the full frame)
  conditions  - the per-component routed condition, drawn ON the component box
  all         - everything overlaid
All boxes are in full-frame pixel coords. The condition is shown on the component box (no separate
near-duplicate box).
"""
import cv2

# BGR colors
C_POLE = (0, 220, 0)        # green
C_COMP = (255, 160, 0)      # blue
C_COND = (0, 0, 255)        # red
LAYERS = ["pole", "components", "conditions", "all"]


def _scale(w):
    """thickness, font-scale proportional to image width (frames are ~4000 px)."""
    return max(2, round(w / 1100)), max(0.5, w / 2600)


def _draw_box(img, box, label, color):
    th, fs = _scale(img.shape[1])
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
    if not label:
        return
    (tw, tht), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    ly = max(y1, tht + bl + 2)
    cv2.rectangle(img, (x1, ly - tht - bl - 2), (x1 + tw + 2, ly), color, -1)
    cv2.putText(img, label, (x1 + 1, ly - bl), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)


def render_layers(frame_bgr, result):
    """frame_bgr: the EXIF-oriented full frame (draw on a copy). Returns {layer: annotated bgr}.
    ONE box per component (post-NMS). The condition is shown ON the component box (no separate
    near-duplicate box): the `conditions` layer relabels the component box with its condition,
    and the `all` layer labels each box with class + condition inline."""
    out = {k: frame_bgr.copy() for k in LAYERS}
    for pole in result["poles"]:
        plabel = f"pole {pole['confidence']:.2f}"
        _draw_box(out["pole"], pole["box"], plabel, C_POLE)
        _draw_box(out["all"], pole["box"], plabel, C_POLE)
        for c in pole.get("components", []):
            box = c["box_full"]
            _draw_box(out["components"], box, f"{c['class']} {c['confidence']:.2f}", C_COMP)
            cond = c.get("condition")
            if cond:
                _draw_box(out["conditions"], box, f"{cond['class']} {cond['confidence']:.2f}", C_COND)
                _draw_box(out["all"], box, f"{c['class']} | {cond['class']} {cond['confidence']:.2f}", C_COMP)
            else:
                _draw_box(out["all"], box, f"{c['class']} {c['confidence']:.2f}", C_COMP)
    return out


def save_layers(frame_bgr, result, viz_dir, stem):
    """Write the 4 layer images to <viz_dir>/<layer>/<stem>.jpg. Returns the paths."""
    from pathlib import Path
    layers = render_layers(frame_bgr, result)
    paths = {}
    for name, img in layers.items():
        d = Path(viz_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{stem}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths[name] = str(p)
    return paths
