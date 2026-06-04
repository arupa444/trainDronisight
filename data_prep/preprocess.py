import cv2
import numpy as np
from PIL import Image, ImageOps

from data_prep.profile_images import profile_array


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


def clahe_image(bgr: np.ndarray) -> np.ndarray:
    """Apply the SAME adaptive CLAHE used in data-prep to an inference image.

    Required so a model trained on the `clahe` variant sees the same pixel
    distribution at inference; skipping this on a clahe-trained model deflates
    detection confidence. Pairs with load_oriented_bgr() for EXIF correctness.
    """
    clip, grid = clahe_params_from_profile(profile_array(bgr))
    return apply_clahe(bgr, clip, grid)
