"""FastAPI server for the DroniSight inspection app.

Run:
    DRONISIGHT_WEIGHTS=runs uvicorn app.server:app --host 0.0.0.0 --port 8000
    # or: python -m app.server   (launches uvicorn for you)

Flow:
    POST /api/analyze (multipart image)        -> {job_id}
    GET  /api/jobs/{job_id}                    -> {status, stage, percent, image_url, error}
    GET  /api/jobs/{job_id}/result             -> the structured report (when status == done)
    GET  /api/health                           -> {device, ready, weights}
Artifacts (crops, viz, csv, json) are served read-only under /files/<job_id>/...
The single-page UI is served at / (static assets under /static).
"""
import os
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference_service import InspectionService

WEIGHTS_DIR = os.environ.get("DRONISIGHT_WEIGHTS", "runs")
RUNS_BASE = Path(os.environ.get("DRONISIGHT_APP_RUNS", "app_runs")).resolve()
RUNS_BASE.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_BYTES = 80 * 1024 * 1024  # 80 MB guard

# Built once; detectors load lazily on the first analysis and stay cached.
service = InspectionService(WEIGHTS_DIR, RUNS_BASE)
executor = ThreadPoolExecutor(max_workers=1)   # serialize GPU/MPS work
JOBS = {}
LOCK = Lock()

app = FastAPI(title="DroniSight Inspection")
app.mount("/files", StaticFiles(directory=str(RUNS_BASE)), name="files")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _set(job_id, **kw):
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(**kw)


def _run_job(job_id, image_path):
    def progress(stage, percent):
        _set(job_id, stage=stage, percent=int(percent), status="running")

    try:
        run_dir = service.runs_dir / job_id
        report = service.analyze(image_path, run_dir, progress)
        _set(job_id, status="done", percent=100, stage="Done", result=report)
    except Exception as e:  # surface the message to the UI, keep a server-side traceback
        traceback.print_exc()
        _set(job_id, status="error", error=f"{type(e).__name__}: {e}")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {
        "device": service.device,
        "ready": service.has_pole,
        "weights": service.weights_status(),
        "weights_dir": service.weights_dir,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMG_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(IMG_EXTS)}")
    if not service.has_pole:
        raise HTTPException(503, f"No pole weights under {service.weights_dir}. "
                                 f"Set DRONISIGHT_WEIGHTS to your trained runs/ folder and restart.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large ({len(data)} bytes > {MAX_BYTES}).")

    job_id = uuid.uuid4().hex[:12]
    run_dir = service.runs_dir / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    image_path = run_dir / f"upload{ext}"
    image_path.write_bytes(data)

    with LOCK:
        JOBS[job_id] = {"status": "queued", "stage": "Queued", "percent": 0,
                        "image_url": f"/files/{job_id}/{image_path.name}",
                        "filename": file.filename, "result": None, "error": None}
    executor.submit(_run_job, job_id, str(image_path))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job id.")
        return {k: job[k] for k in ("status", "stage", "percent", "image_url", "filename", "error")}


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job id.")
        if job["status"] == "error":
            raise HTTPException(500, job["error"] or "Analysis failed.")
        if job["status"] != "done":
            raise HTTPException(409, f"Job not finished (status={job['status']}).")
        return JSONResponse(job["result"])


def main():
    import uvicorn
    host = os.environ.get("DRONISIGHT_HOST", "127.0.0.1")
    port = int(os.environ.get("DRONISIGHT_PORT", "8000"))
    print(f"DroniSight app -> http://{host}:{port}  (device: {service.device}, "
          f"weights: {service.weights_dir}, ready: {service.has_pole})")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
