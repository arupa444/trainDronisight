"""Draw pipeline results onto the full frame as four layered views:
  pole        - just the pole box(es)
  components  - the above+below component boxes (remapped to the full frame)
  conditions  - the per-component condition box(es), labeled with the condition class
  all         - everything overlaid
All boxes are in full-frame pixel coords. Condition boxes are stored in component-crop coords
(box_comp), so we shift them by the component's full-frame top-left to place them on the frame.
"""
import cv2

# BGR colors
C_POLE = (0, 220, 0)        # green
C_ABOVE = (255, 160, 0)     # blue
C_BELOW = (200, 0, 200)     # magenta
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


def _condition_full_box(comp):
    """Best in-family condition box remapped to the full frame, or None."""
    conds = comp.get("conditions") or []
    if not conds:
        return None
    bc = conds[0]["box_comp"]
    ox, oy = comp["box_full"][0], comp["box_full"][1]
    return [bc[0] + ox, bc[1] + oy, bc[2] + ox, bc[3] + oy]


def render_layers(frame_bgr, result):
    """frame_bgr: the EXIF-oriented full frame (draw on a copy). Returns {layer: annotated bgr}."""
    out = {k: frame_bgr.copy() for k in LAYERS}
    for pole in result["poles"]:
        plabel = f"pole {pole['confidence']:.2f}"
        _draw_box(out["pole"], pole["box"], plabel, C_POLE)
        _draw_box(out["all"], pole["box"], plabel, C_POLE)
        for grp, color in (("components_above", C_ABOVE), ("components_below", C_BELOW)):
            for c in pole.get(grp, []):
                clabel = f"{c['class']} {c['confidence']:.2f}"
                _draw_box(out["components"], c["box_full"], clabel, color)
                _draw_box(out["all"], c["box_full"], clabel, color)
                cond = c.get("condition")
                cbox = _condition_full_box(c)
                if cond and cbox is not None:
                    condlabel = f"{cond['class']} {cond['confidence']:.2f}"
                    _draw_box(out["conditions"], cbox, condlabel, C_COND)
                    _draw_box(out["all"], cbox, condlabel, C_COND)
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
