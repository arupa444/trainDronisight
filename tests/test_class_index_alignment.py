"""Cross-emitter class-index alignment: YOLO label index (0-based) must equal COCO
category_id (0-based) for every class in every subset, and the FRCNN convention is
category_id + 1 (0 = background). Guards against a silent SUBSET_CLASSES reordering that
would make one model's labels disagree with another's."""
from shared import config
from shared.labels import Box, Annotation, to_yolo_line
from data_prep.emit_coco import build_coco


def test_yolo_index_equals_coco_category_id_for_every_subset():
    for subset, classes in config.SUBSET_CLASSES.items():
        # one image with exactly one box per class, in class order
        boxes = [Box(c, 1, 1, 10, 10) for c in classes]
        ann = Annotation(64, 64, boxes)
        coco = build_coco({"img.jpg": ann}, classes)
        # COCO categories are 0-based and indexed by class order
        cat_by_name = {c["name"]: c["id"] for c in coco["categories"]}
        for c in classes:
            yolo_idx = int(to_yolo_line(Box(c, 1, 1, 10, 10), 64, 64, classes).split()[0])
            assert yolo_idx == cat_by_name[c] == classes.index(c), \
                f"{subset}/{c}: yolo={yolo_idx} coco={cat_by_name[c]}"


def test_frcnn_is_coco_plus_one():
    # the torchvision detector maps model label L -> class_names[L-1] (0=background);
    # equivalently COCO category_id + 1. Confirm the inverse round-trips.
    from inference.backends import parse_torchvision_output
    import torch
    classes = config.COMPONENT_ABOVE_CLASSES
    out = {"boxes": torch.tensor([[0.0, 0, 5, 5]]),
           "scores": torch.tensor([0.9]),
           "labels": torch.tensor([2])}              # model label 2 -> class index 1
    dets = parse_torchvision_output(out, classes, conf=0.5)
    assert dets[0].class_name == classes[1]
