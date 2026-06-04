# tests/test_train_args.py
from shared.train_args import build_yolo_args

def test_pole_args_basic():
    a = build_yolo_args(subset="pole", data_yaml="/x/data.yaml", device="mps",
                        epochs=100, imgsz=1280, batch=8)
    assert a["data"] == "/x/data.yaml"
    assert a["device"] == "mps"
    assert a["imgsz"] == 1280

def test_orientation_priors_respected():
    a = build_yolo_args("components", "/x.yaml", "cpu", 10, 1280, 4)
    assert a["flipud"] == 0.0          # no vertical flip (poles have up-down prior)
    assert a["degrees"] <= 10          # only mild rotation

def test_components_get_stronger_scale_jitter_for_crop_gap():
    pole = build_yolo_args("pole", "/p.yaml", "cpu", 10, 1280, 4)
    comp = build_yolo_args("components", "/c.yaml", "cpu", 10, 1280, 4)
    assert comp["scale"] > pole["scale"]   # simulate zoomed-in inference crops

def test_close_mosaic_and_seed_set():
    a = build_yolo_args("pole", "/p.yaml", "cpu", 100, 1280, 4)
    assert a["close_mosaic"] >= 10
    assert "seed" in a

def test_explicit_regularization_present():
    a = build_yolo_args("components", "/c.yaml", "cpu", 10, 1280, 4)
    assert a["weight_decay"] == 0.0005   # L2
    assert a["dropout"] > 0              # head dropout
    assert a["patience"] >= 1            # early stopping

def test_mixup_only_for_multiclass():
    pole = build_yolo_args("pole", "/p.yaml", "cpu", 10, 640, 4)
    comp = build_yolo_args("components", "/c.yaml", "cpu", 10, 1280, 4)
    # mixup: extra regularizer reserved for the harder 4-class task
    assert pole["mixup"] == 0.0
    assert comp["mixup"] > 0.0

def test_no_deprecated_label_smoothing():
    # label_smoothing is deprecated in Ultralytics 8.4+; we must not pass it
    a = build_yolo_args("components", "/c.yaml", "cpu", 10, 1280, 4)
    assert "label_smoothing" not in a
