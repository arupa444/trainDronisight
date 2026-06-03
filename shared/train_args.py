from shared import config


def build_yolo_args(subset, data_yaml, device, epochs, imgsz, batch):
    """Build Ultralytics train kwargs with a domain-aware augmentation policy.

    Spec §6.3: poles/insulators have a strong up-down orientation prior -> no
    vertical flip, only mild rotation, no heavy blur. Spec §6.1: Model 2 is
    trained on full frames but runs on cropped pole regions, so components get a
    wider scale-jitter range to simulate the zoomed-in crop distribution.
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
        # schedule
        "cos_lr": True,
        "patience": 30,
        "amp": True,
    }
