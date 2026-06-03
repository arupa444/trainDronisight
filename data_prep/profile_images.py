import cv2
import numpy as np


def profile_array(bgr: np.ndarray) -> dict:
    """Compute exposure/clip/haze/sharpness statistics for one BGR image."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    n = L.size

    highlight_clip = float((L >= 250).sum() / n)
    shadow_clip = float((L <= 5).sum() / n)
    mean_luma = float(L.mean())
    std_luma = float(L.std())

    # dark-channel prior as a coarse haze proxy (higher == hazier)
    dark = cv2.erode(bgr.min(axis=2), np.ones((15, 15), np.uint8))
    haze = float(dark.mean() / 255.0)

    sharpness = float(cv2.Laplacian(L, cv2.CV_32F).var())

    # backlit: meaningful blown highlights AND a dark region present
    backlit = bool(highlight_clip > 0.15 and (L < 60).mean() > 0.15)

    return {
        "mean_luma": mean_luma,
        "std_luma": std_luma,
        "highlight_clip": highlight_clip,
        "shadow_clip": shadow_clip,
        "haze": haze,
        "sharpness": sharpness,
        "backlit": backlit,
    }
