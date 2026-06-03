import cv2
import numpy as np
from PIL import Image, ImageOps


def load_oriented_bgr(path) -> np.ndarray:
    """Load an image honoring EXIF orientation, return BGR uint8."""
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.asarray(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def clahe_params_from_profile(profile: dict):
    """Pick (clipLimit, tileGrid) adaptively: near-identity unless backlit/clipped."""
    grid = (8, 8)
    if profile.get("backlit") or profile.get("highlight_clip", 0) > 0.2:
        clip = 3.0 if profile.get("highlight_clip", 0) > 0.35 else 2.0
    elif profile.get("shadow_clip", 0) > 0.2:
        clip = 2.0
    else:
        clip = 1.0  # effectively identity
    return clip, grid


def apply_clahe(bgr: np.ndarray, clip: float, grid) -> np.ndarray:
    """CLAHE on the LAB L-channel only; chroma untouched."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
