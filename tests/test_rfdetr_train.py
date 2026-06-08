# tests/test_rfdetr_train.py
from unittest import mock

import pytest

from train_rf_detr import train


def test_warns_when_not_cuda_and_passes_resolution(capsys):
    with mock.patch("train_rf_detr.train.select_device", return_value="mps"), \
         mock.patch("train_rf_detr.train.build_rfdetr_view", return_value="/ds"), \
         mock.patch("train_rf_detr.train.RFDETRLarge") as M:
        train.run(subset="component_above_1000", version="clahe", epochs=1, batch=2,
                  resolution=672)   # 672 divisible by both 32 and 56 -> valid on any RF-DETR build
    out = capsys.readouterr().out
    assert "CUDA" in out
    assert "CLAHE preprocessing: ON" in out          # clahe variant confirmed in the banner
    M.assert_called_once_with(resolution=672)         # resolution reaches the model ctor
    kwargs = M.return_value.train.call_args.kwargs
    assert kwargs["dataset_dir"] == "/ds"
    assert kwargs["epochs"] == 1


def test_resolution_must_be_multiple_of_block_size():
    # 700 is divisible by neither 32 (current build) nor 56 (older build) -> always rejected
    with pytest.raises(ValueError):
        train.run(subset="component_above_1000", version="clahe", epochs=1, batch=2,
                  resolution=700)


def test_block_size_and_default_resolution_agree():
    # guards the version-drift bug: the trainer default must be divisible by the installed
    # model's real block_size (patch_size*num_windows), not a hard-coded 56.
    import inspect
    from inference.backends import rfdetr_block_size
    block = rfdetr_block_size()
    assert isinstance(block, int) and block > 0
    default_res = inspect.signature(train.run).parameters["resolution"].default
    assert default_res % block == 0
