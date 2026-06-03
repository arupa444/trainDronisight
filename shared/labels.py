"""Label source-of-truth = name-based VOC XML. Index-based YOLO .txt files are ignored."""

# All raw spellings seen in the data, mapped to a canonical kept class.
# Excluded classes (rust, om_crossarm, top_crossarm, vegetation) are intentionally
# NOT listed -> they normalize to None (ignored, never deleted from source).
_CANONICAL = {
    "pole": "pole",
    "wire": "wire",
    "h_insulator": "h_insulator",
    "v_insulator": "v_insulator",
    "crossarm_stright": "crossarm_stright",
    "crossarm_stright ": "crossarm_stright",
    "crossarmstright": "crossarm_stright",  # the mem5 stray 10th class
}


def normalize_class_name(raw: str):
    """Return canonical kept-class name, or None if the class is excluded/unknown."""
    if raw is None:
        return None
    key = raw.strip().lower()
    return _CANONICAL.get(key)
