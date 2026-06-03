import pytest
from data_prep.verify_dataset import assert_no_group_leakage, class_counts_from_manifest
import pandas as pd

def test_leakage_detector_passes_clean_split():
    df = pd.DataFrame([
        {"group": "mem2:0", "split": "train"},
        {"group": "mem2:0", "split": "train"},
        {"group": "mem2:1", "split": "val"},
    ])
    assert_no_group_leakage(df)  # no raise

def test_leakage_detector_raises_on_span():
    df = pd.DataFrame([
        {"group": "mem2:0", "split": "train"},
        {"group": "mem2:0", "split": "val"},
    ])
    with pytest.raises(AssertionError):
        assert_no_group_leakage(df)

def test_class_counts():
    df = pd.DataFrame([{"split": "train"}, {"split": "val"}])
    counts = class_counts_from_manifest(df)
    assert counts == {"train": 1, "val": 1}

def test_find_invalid_labels_skips_appledouble_and_flags_bad(tmp_path):
    from data_prep.verify_dataset import find_invalid_labels
    (tmp_path / "good.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "._good.txt").write_bytes(b"\x00\xb0\x05bad applebinary")  # must be skipped
    (tmp_path / "bad.txt").write_text("0 1.5 0.5 0.2 0.2\n")  # coord > 1 -> invalid
    bad = find_invalid_labels(tmp_path)
    assert any(p.endswith("bad.txt") for p in bad)
    assert not any("._good" in p for p in bad)
