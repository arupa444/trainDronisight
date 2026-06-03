from inference.backends import Detection


def crop_with_pad(image, box, pad_frac=0.05):
    """Crop image to box (x1,y1,x2,y2) with optional padding fraction of box size.
    Returns (crop_array, (offset_x, offset_y)). Offsets are clamped to >= 0."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    pw = (x2 - x1) * pad_frac
    ph = (y2 - y1) * pad_frac
    cx1 = max(0, int(x1 - pw))
    cy1 = max(0, int(y1 - ph))
    cx2 = min(w, int(x2 + pw))
    cy2 = min(h, int(y2 + ph))
    return image[cy1:cy2, cx1:cx2], (cx1, cy1)


def shift_detection(det: Detection, off_x: int, off_y: int) -> Detection:
    """Map a detection from crop coordinates back to full-frame coordinates."""
    x1, y1, x2, y2 = det.box
    return Detection(det.class_name, det.confidence,
                     (x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y))
