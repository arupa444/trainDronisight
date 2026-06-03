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
