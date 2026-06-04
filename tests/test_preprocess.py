import numpy as np
from data_prep.preprocess import clahe_params_from_profile, apply_clahe

def test_well_exposed_image_gets_near_identity_clip():
    profile = {"backlit": False, "highlight_clip": 0.01, "shadow_clip": 0.01}
    clip, grid = clahe_params_from_profile(profile)
    assert clip <= 1.2  # near identity

def test_backlit_image_gets_stronger_clip():
    profile = {"backlit": True, "highlight_clip": 0.4, "shadow_clip": 0.3}
    clip, grid = clahe_params_from_profile(profile)
    assert clip >= 2.0

def test_apply_clahe_preserves_shape_and_dtype():
    img = np.random.randint(0, 255, (48, 64, 3), np.uint8)
    out = apply_clahe(img, clip=2.0, grid=(8, 8))
    assert out.shape == img.shape and out.dtype == np.uint8

def test_apply_clahe_increases_contrast_on_low_contrast_input():
    img = np.full((48, 64, 3), 100, np.uint8)
    img[:24] = 110  # very low contrast
    out = apply_clahe(img, clip=3.0, grid=(8, 8))
    assert out.std() >= img.std()

def test_clahe_image_runs_full_profile_pipeline():
    # inference convenience: profile -> params -> apply, in one call
    from data_prep.preprocess import clahe_image
    img = np.random.randint(0, 255, (48, 64, 3), np.uint8)
    out = clahe_image(img)
    assert out.shape == img.shape and out.dtype == np.uint8
