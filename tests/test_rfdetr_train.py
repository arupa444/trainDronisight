# tests/test_rfdetr_train.py
from unittest import mock

import pytest

from train_rf_detr import train


def test_warns_when_not_cuda_and_passes_resolution(capsys):
    with mock.patch("train_rf_detr.train.select_device", return_value="mps"), \
         mock.patch("train_rf_detr.train.build_rfdetr_view", return_value="/ds"), \
         mock.patch("train_rf_detr.train.RFDETRLarge") as M:
        train.run(subset="component_above_1000", version="clahe", epochs=1, batch=2,
                  resolution=728)
    out = capsys.readouterr().out
    assert "CUDA" in out
    assert "CLAHE preprocessing: ON" in out          # clahe variant confirmed in the banner
    M.assert_called_once_with(resolution=728)         # resolution reaches the model ctor
    kwargs = M.return_value.train.call_args.kwargs
    assert kwargs["dataset_dir"] == "/ds"
    assert kwargs["epochs"] == 1


def test_resolution_must_be_multiple_of_56():
    with pytest.raises(ValueError):
        train.run(subset="component_above_1000", version="clahe", epochs=1, batch=2,
                  resolution=700)  # 700 is not divisible by 56
