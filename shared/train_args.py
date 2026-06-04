from shared import config


def build_yolo_args(subset, data_yaml, device, epochs, imgsz, batch):
    """Build Ultralytics train kwargs with a domain-aware augmentation + regularization policy.

    Spec §6.3: poles/insulators have a strong up-down orientation prior -> no
    vertical flip, only mild rotation, no heavy blur. Spec §6.1: Model 2 is
    trained on full frames but runs on cropped pole regions, so components get a
    wider scale-jitter range to simulate the zoomed-in crop distribution.

    Anti-overfitting: the dataset is small (~750-1000 train images) relative to model
    capacity, so we lean on augmentation + early stopping + explicit weight decay,
    dropout, label smoothing (multi-class only) and mixup (the harder component task).
    The single biggest lever, though, is MODEL SIZE -- prefer yolo26m/s over yolo26x
    on this data (pass --model yolo26m.pt). Watch the train-vs-val curves Ultralytics
    plots in results.png: a widening gap = overfitting -> smaller model / fewer epochs.
    """
    is_components = subset == "components"
    return {
        "data": data_yaml,
        "device": device,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": config.SEED,
        "project": f"runs/{subset}",
        "name": "yolo",
        # augmentation
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,   # outdoor lighting variance
        "fliplr": 0.5,
        "flipud": 0.0,                                  # orientation prior
        "degrees": 10.0,                                # mild only
        "translate": 0.1,
        "scale": 0.9 if is_components else 0.5,         # crop-gap mitigation
        "mosaic": 1.0,
        "close_mosaic": 10,                             # finish on realistic images
        "copy_paste": 0.3 if is_components else 0.0,    # help scarcer component classes
        "mixup": 0.1 if is_components else 0.0,         # extra regularizer for the 4-class task
        # regularization (small data + large model -> actively guard against overfitting)
        "weight_decay": 0.0005,                         # L2 (explicit, was an implicit default)
        "dropout": 0.1,                                 # light head dropout
        # (label_smoothing intentionally omitted: deprecated in Ultralytics 8.4+;
        #  weight_decay + dropout + mixup + augmentation + early stopping cover regularization)
        # schedule
        "cos_lr": True,
        "patience": 30,                                 # early stop when val stalls
        "amp": True,
    }
