from datetime import datetime
from data_prep.grouping import parse_capture_time, assign_groups

def test_parse_dji_timestamp():
    dt = parse_capture_time("DJI_20260325112518_0159_D.JPG")
    assert dt == datetime(2026, 3, 25, 11, 25, 18)

def test_parse_returns_none_for_nonmatching():
    assert parse_capture_time("random.JPG") is None

def test_groups_split_on_time_gap():
    # two frames 5s apart, then a 10-min gap, then one more -> 2 groups
    names = [
        "DJI_20260325112518_0001_D.JPG",
        "DJI_20260325112523_0002_D.JPG",
        "DJI_20260325113523_0003_D.JPG",
    ]
    groups = assign_groups(names, source="mem2", gap_seconds=60)
    assert groups[names[0]] == groups[names[1]]
    assert groups[names[1]] != groups[names[2]]

def test_group_ids_are_namespaced_by_source():
    g = assign_groups(["DJI_20260325112518_0001_D.JPG"], source="mem2", gap_seconds=60)
    assert g["DJI_20260325112518_0001_D.JPG"].startswith("mem2:")
