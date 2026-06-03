"""Device selection with priority CUDA -> MPS -> CPU.

torch is imported lazily so data_prep (Plan 1) has no torch dependency.
"""


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def select_device() -> str:
    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"
