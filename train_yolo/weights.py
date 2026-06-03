import warnings


def _loadable(name: str) -> bool:
    """True if Ultralytics can instantiate/download these weights."""
    try:
        from ultralytics import YOLO
        YOLO(name)
        return True
    except Exception:
        return False


def resolve_weights(preferred: str, fallback: str):
    """Return (weights_name, fell_back_bool). Try preferred (YOLO26x), else fallback."""
    if _loadable(preferred):
        return preferred, False
    warnings.warn(f"{preferred} not loadable on this Ultralytics version; "
                  f"falling back to {fallback}.")
    if _loadable(fallback):
        return fallback, True
    raise RuntimeError(f"Neither {preferred} nor {fallback} could be loaded.")
