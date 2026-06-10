"""DroniSight inspection web app: upload an image -> full 12-model pipeline -> structured report.

Backend: FastAPI (app.server:app). Frontend: a zero-build single-page app in app/static/.
Device auto-selects CUDA -> MPS -> CPU via shared.device.select_device().
"""
