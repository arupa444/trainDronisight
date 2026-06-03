# tests/test_yolo_cli.py
from unittest import mock
from train_yolo import train_pole

def test_train_pole_builds_args_and_calls_yolo():
    fake_model = mock.MagicMock()
    with mock.patch("train_yolo.train_pole.YOLO", return_value=fake_model) as Y, \
         mock.patch("train_yolo.train_pole.resolve_weights", return_value=("yolo26x.pt", False)), \
         mock.patch("train_yolo.train_pole.select_device", return_value="cpu"):
        train_pole.run(version="clahe", epochs=1, imgsz=640, batch=2)
    Y.assert_called_once_with("yolo26x.pt")
    kwargs = fake_model.train.call_args.kwargs
    assert kwargs["device"] == "cpu"
    assert kwargs["data"].endswith("pole/data_clahe.yaml")
    assert kwargs["epochs"] == 1
