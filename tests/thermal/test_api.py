import base64
import numpy as np
import cv2
from fastapi.testclient import TestClient
from thermal.schema import Detection
from thermal.colormap import build_lut, ColorToHeat
from thermal.api import create_app


class FakeDetector:
    def __init__(self, dets):
        self._dets = dets

    def detect(self, img_bgr, conf=0.25):
        return self._dets


def _make_client(dets=None):
    if dets is None:
        dets = [Detection("transformer", (0, 0, 120, 120))]
    app = create_app(FakeDetector(dets), ColorToHeat(build_lut("inferno")))
    return TestClient(app)


def _png(img_rgb):
    # encode the BGR bytes so the API's decode(BGR)->RGB round-trips back to img_rgb
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def _png_black():
    return _png(np.zeros((120, 120, 3), dtype=np.uint8))


def _png_with_hotspot():
    # mostly a mid-palette color with a small hottest-palette patch -> a real hotspot
    lut = (build_lut("inferno") * 255).astype(np.uint8)  # RGB, cold->hot
    img = np.empty((120, 120, 3), dtype=np.uint8)
    img[:] = lut[128]            # warm body
    img[50:70, 50:70] = lut[-1]  # hottest color -> high intensity patch
    return _png(img)


def test_analyze_endpoint_returns_report():
    client = _make_client()
    resp = client.post("/analyze", files={"file": ("t.png", _png_black(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "defects" in body and "calibration_ok" in body and "annotated_image_png_b64" in body
    raw = base64.b64decode(body["annotated_image_png_b64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_analyze_endpoint_flags_hotspot():
    client = _make_client()
    resp = client.post("/analyze", files={"file": ("hot.png", _png_with_hotspot(), "image/png")})
    assert resp.status_code == 200
    defects = resp.json()["defects"]
    assert len(defects) >= 1
    for d in defects:
        assert set(d) >= {"component", "bbox", "severity", "relative_delta"}
        assert d["component"] == "hotspot"
        assert isinstance(d["bbox"], list) and len(d["bbox"]) == 4


def test_analyze_invalid_image_returns_422():
    client = _make_client()
    resp = client.post("/analyze", files={"file": ("bad.png", b"not an image", "image/png")})
    assert resp.status_code == 422


def test_analyze_empty_upload_returns_422():
    client = _make_client()
    resp = client.post("/analyze", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 422
