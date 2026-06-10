# DroniSight inspection web app

Upload a drone frame → it runs the full **12-model pipeline** (pole → 5 component specialists →
NMS → 6 condition specialists) → you get an animated "Analysing…" screen → then a single report
view with the annotated frame (layer toggle), the padded component crops, an attention list, and a
structured table + CSV/JSON download.

Device is auto-selected **CUDA → MPS → CPU** (`shared/device.py`) and threaded into every detector.

## Install

```bash
source .venv/bin/activate
uv pip install -e ".[app]"        # fastapi + uvicorn + python-multipart
```

## Run

By default the app auto-finds weights in `models/` (then `runs/`, then `~/Downloads/runs`) — so with
the trained set under `models/runs/<subset>/.../weights/best.pt` you just run:

```bash
python -m app.server          # auto-resolves to models/; open http://127.0.0.1:8000
```

Override the location explicitly with `DRONISIGHT_WEIGHTS` (best.pt is auto-discovered by subset
name — pole + 5 comp_* + 6 cond_*):

```bash
DRONISIGHT_WEIGHTS=/path/to/runs python -m app.server
```

Env vars:

| var | default | meaning |
|---|---|---|
| `DRONISIGHT_WEIGHTS` | `runs` | folder searched for `**/<subset>/**/weights/best.pt` |
| `DRONISIGHT_APP_RUNS` | `app_runs` | where per-job artifacts (crops, viz, csv, json) are written + served |
| `DRONISIGHT_HOST` / `DRONISIGHT_PORT` | `127.0.0.1` / `8000` | bind address |

`/api/health` shows the device and which of the 12 weights were found; the UI greys out missing
specialists and warns if `pole` is absent.

## API

| route | purpose |
|---|---|
| `POST /api/analyze` (multipart `file`) | start a job → `{job_id}` |
| `GET /api/jobs/{id}` | `{status, stage, percent, image_url, error}` (poll this) |
| `GET /api/jobs/{id}/result` | the structured report (when `status == done`) |
| `GET /api/health` | `{device, ready, weights}` |
| `/files/<job>/…` | read-only artifacts (crops, viz layers, result.csv/json) |

Models load once on the first analysis and stay cached; jobs are serialized on a single worker so
GPU/MPS memory isn't oversubscribed.
