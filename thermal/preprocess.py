import cv2
import numpy as np

from thermal.data_prep.imaging import clahe_image


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Enhance a colorized thermal image for the detector.

    Applies the SAME adaptive CLAHE used to build the dataset's `clahe` variant
    (LAB L-channel; per-image clip from the image profile, near-identity unless the
    frame is backlit/clipped). Using the identical transform at train and inference
    avoids train/serve skew. The palette (chroma) is left intact, so the defect
    layer's relative-heat reading on the raw image stays valid.

    Input/output: HxWx3 uint8 RGB.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("preprocess expects an HxWx3 RGB image")
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(clahe_image(bgr), cv2.COLOR_BGR2RGB)
