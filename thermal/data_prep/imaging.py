"""Image loading + adaptive CLAHE, ported from the trainDronisight design.

The dataset stores two variants per image: `orig` (untouched) and `clahe`
(adaptive, near-identity unless the frame is backlit/clipped). Models are trained
on `clahe`, so inference MUST apply the SAME adaptive CLAHE — `clahe_image()` is
the single shared entry point for both build and serve.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_oriented_bgr(path) -> np.ndarray:
    """Load honoring EXIF orientation; return BGR uint8."""
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def profile_array(bgr: np.ndarray) -> dict:
    """Exposure / clip / backlit statistics for one BGR image."""
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    n = L.size
    highlight_clip = float((L >= 250).sum() / n)
    shadow_clip = float((L <= 5).sum() / n)
    backlit = bool(highlight_clip > 0.15 and (L < 60).mean() > 0.15)
    return {"highlight_clip": highlight_clip, "shadow_clip": shadow_clip,
            "backlit": backlit, "mean_luma": float(L.mean()), "std_luma": float(L.std())}


def clahe_params_from_profile(profile: dict):
    """Adaptive (clipLimit, tileGrid): near-identity unless backlit/clipped."""
    grid = (8, 8)
    if profile.get("backlit") or profile.get("highlight_clip", 0) > 0.2:
        clip = 3.0 if profile.get("highlight_clip", 0) > 0.35 else 2.0
    elif profile.get("shadow_clip", 0) > 0.2:
        clip = 2.0
    else:
        clip = 1.0  # effectively identity
    return clip, grid


def apply_clahe(bgr: np.ndarray, clip: float, grid) -> np.ndarray:
    """CLAHE on the LAB L-channel only; chroma (the thermal palette) untouched."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(grid)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_image(bgr: np.ndarray) -> np.ndarray:
    """Adaptive CLAHE used in BOTH data-prep and inference (train/serve parity)."""
    clip, grid = clahe_params_from_profile(profile_array(bgr))
    return apply_clahe(bgr, clip, grid)
