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
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference_service import InspectionService

WEIGHTS_DIR = os.environ.get("DRONISIGHT_WEIGHTS", "runs")
RUNS_BASE = Path(os.environ.get("DRONISIGHT_APP_RUNS", "app_runs")).resolve()
RUNS_BASE.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_BYTES = int(os.environ.get("DRONISIGHT_MAX_BYTES", 80 * 1024 * 1024))  # 80 MB guard
MAX_JOBS = int(os.environ.get("DRONISIGHT_MAX_JOBS", 50))  # retain only the N most-recent finished jobs

# Built once; detectors load lazily on the first analysis and stay cached.
service = InspectionService(WEIGHTS_DIR, RUNS_BASE)
executor = ThreadPoolExecutor(max_workers=1)   # serialize GPU/MPS work
JOBS = {}
LOCK = Lock()


@asynccontextmanager
async def lifespan(app):
    yield
    executor.shutdown(wait=False, cancel_futures=True)   # drain/reclaim worker on shutdown


app = FastAPI(title="DroniSight Inspection", lifespan=lifespan)
app.mount("/files", StaticFiles(directory=str(RUNS_BASE)), name="files")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _set(job_id, **kw):
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(**kw)


def _evict_finished():
    """Keep memory + disk bounded: drop the oldest FINISHED jobs (and their run dirs) beyond
    MAX_JOBS. Never evicts a queued/running job. rmtree runs outside the lock (slow I/O)."""
    to_remove = []
    with LOCK:
        finished = [jid for jid, j in JOBS.items() if j["status"] in ("done", "error")]
        # JOBS is insertion-ordered -> `finished` is oldest-first
        overflow = len(JOBS) - MAX_JOBS
        for jid in finished[:max(0, overflow)]:
            JOBS.pop(jid, None)
            to_remove.append(jid)
    for jid in to_remove:
        shutil.rmtree(service.runs_dir / jid, ignore_errors=True)


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
async def analyze(request: Request, file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMG_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(IMG_EXTS)}")
    if not service.has_pole:
        raise HTTPException(503, f"No pole weights under {service.weights_dir}. "
                                 f"Set DRONISIGHT_WEIGHTS to your trained runs/ folder and restart.")
    # cheap early reject on the declared size, then the authoritative streamed cap below
    clen = request.headers.get("content-length")
    if clen and clen.isdigit() and int(clen) > MAX_BYTES + (1 << 20):
        raise HTTPException(413, f"File too large (> {MAX_BYTES} bytes).")

    # stream in chunks so an oversized upload is aborted WITHOUT buffering it all in RAM first
    job_id = uuid.uuid4().hex[:12]
    run_dir = service.runs_dir / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    image_path = run_dir / f"upload{ext}"
    total = 0
    try:
        with open(image_path, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)   # 1 MB
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(413, f"File too large (> {MAX_BYTES} bytes).")
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    if total == 0:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(400, "Empty upload.")

    with LOCK:
        JOBS[job_id] = {"status": "queued", "stage": "Queued", "percent": 0,
                        "image_url": f"/files/{job_id}/{image_path.name}",
                        "filename": file.filename, "result": None, "error": None}
    _evict_finished()
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
