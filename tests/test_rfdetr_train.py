# tests/test_rfdetr_train.py
from unittest import mock
from train_rf_detr import train

def test_warns_when_not_cuda(capsys):
    with mock.patch("train_rf_detr.train.select_device", return_value="mps"), \
         mock.patch("train_rf_detr.train.build_rfdetr_view", return_value="/ds"), \
         mock.patch("train_rf_detr.train.RFDETRLarge") as M:
        train.run(subset="components", version="clahe", epochs=1, batch=2)
    assert "CUDA" in capsys.readouterr().out
    M.return_value.train.assert_called_once()
    kwargs = M.return_value.train.call_args.kwargs
    assert kwargs["dataset_dir"] == "/ds"
    assert kwargs["epochs"] == 1
