"""End-to-end app flow with FAKE detectors (no real weights / no GPU):
upload -> poll status -> structured report. Exercises the server's job lifecycle, the
InspectionService orchestration, and the report builder. Skipped if fastapi/httpx absent."""
import io
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import cv2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from inference.backends import Detection  # noqa: E402


class _Fake:
    def __init__(self, dets): self._dets = dets
    def predict(self, image): return self._dets


@pytest.fixture
def client(tmp_path, monkeypatch):
    # point the app's run/output dir at tmp BEFORE importing the server
    monkeypatch.setenv("DRONISIGHT_APP_RUNS", str(tmp_path / "runs"))
    import importlib
    from app import server as srv
    importlib.reload(srv)

    svc = srv.service
    svc.runs_dir = tmp_path / "runs"
    # pretend weights exist + inject fakes so load_models is a no-op
    svc.weights = {s: "fake.pt" for s in __import__("shared").config.SUBSETS}
    svc._loaded = True
    svc.pole_det = _Fake([Detection("pole", 0.95, (10, 10, 180, 180))])
    svc.component_dets = [
        _Fake([Detection("v_insulator", 0.8, (5, 5, 40, 40))]),     # has a condition family
        _Fake([Detection("vegetation", 0.7, (60, 60, 120, 120))]),  # attention component, no family
    ]
    svc.condition_dets = {"cond_v_insulator": _Fake([Detection("v_insulator_broken", 0.66, (1, 1, 20, 20))])}
    # avoid disk image decode: return a plain BGR frame for any path
    monkeypatch.setattr(srv.service.__class__, "load_models", lambda self, progress=None: None)
    import app.inference_service as isvc
    monkeypatch.setattr(isvc, "load_oriented_bgr", lambda p: np.full((200, 200, 3), 120, np.uint8))
    monkeypatch.setattr(isvc, "clahe_image", lambda im: im)
    return TestClient(srv.app)


def _png_bytes():
    ok, buf = cv2.imencode(".png", np.full((40, 40, 3), 200, np.uint8))
    return buf.tobytes()


def test_health_reports_device_and_weights(client):
    h = client.get("/api/health").json()
    assert h["device"] in ("cuda", "mps", "cpu")
    assert h["ready"] is True
    assert set(h["weights"]) == set(__import__("shared").config.SUBSETS)


def test_full_flow_upload_poll_report(client):
    r = client.post("/api/analyze", files={"file": ("frame.png", _png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    status = None
    for _ in range(100):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status["status"] == "done", status

    rep = client.get(f"/api/jobs/{job_id}/result").json()
    assert rep["summary"]["poles"] == 1
    assert rep["summary"]["components"] == 2
    assert rep["summary"]["attention"] == 2          # v_insulator_broken (defect) + vegetation
    assert set(rep["viz"]) >= {"pole", "components", "conditions", "all"}
    # the v_insulator carries its routed, in-family condition flagged as a defect
    comps = {c["class"]: c for c in rep["poles"][0]["components"]}
    assert comps["v_insulator"]["condition"]["class"] == "v_insulator_broken"
    assert comps["v_insulator"]["condition"]["defect"] is True
    assert comps["v_insulator"]["attention"] is True
    # vegetation has no condition family but is itself an attention item
    assert comps["vegetation"]["condition"] is None
    assert comps["vegetation"]["attention"] is True
    # downloads point at the per-job files mount
    assert rep["downloads"]["csv"].startswith("/files/") and rep["downloads"]["csv"].endswith(".csv")


def test_rejects_non_image(client):
    r = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_unknown_job_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404
