import numpy as np
from data_prep.profile_images import profile_array

def test_bright_image_has_high_highlight_clip():
    img = np.full((64, 64, 3), 255, np.uint8)
    p = profile_array(img)
    assert p["highlight_clip"] > 0.9
    assert p["mean_luma"] > 240

def test_dark_image_has_high_shadow_clip():
    img = np.zeros((64, 64, 3), np.uint8)
    p = profile_array(img)
    assert p["shadow_clip"] > 0.9

def test_backlit_image_flagged():
    img = np.zeros((64, 64, 3), np.uint8)
    img[:20, :, :] = 255  # bright sky band, dark below
    p = profile_array(img)
    assert p["highlight_clip"] > 0.2
    assert p["backlit"] is True

def test_profile_keys_present():
    img = np.full((32, 32, 3), 128, np.uint8)
    p = profile_array(img)
    for k in ("mean_luma", "std_luma", "highlight_clip", "shadow_clip", "haze", "sharpness", "backlit"):
        assert k in p
