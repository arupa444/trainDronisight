import re
from datetime import datetime

_TS = re.compile(r"DJI_(\d{14})_")


def parse_capture_time(filename: str):
    """Extract the DJI capture timestamp (YYYYMMDDHHMMSS) from a filename."""
    m = _TS.search(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")


def assign_groups(filenames, source: str, gap_seconds: int) -> dict:
    """Group consecutive frames; a gap > gap_seconds starts a new group.

    Files with no parseable timestamp each become their own group (conservative:
    never merged, so they can't leak across splits).
    """
    timed = []
    untimed = []
    for fn in filenames:
        t = parse_capture_time(fn)
        (timed if t else untimed).append((fn, t))
    timed.sort(key=lambda x: x[1])

    groups = {}
    gid = 0
    prev = None
    for fn, t in timed:
        if prev is not None and (t - prev).total_seconds() > gap_seconds:
            gid += 1
        groups[fn] = f"{source}:{gid}"
        prev = t
    for fn, _ in untimed:
        gid += 1
        groups[fn] = f"{source}:{gid}"
    return groups
