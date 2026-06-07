from train_faster_rcnn.eval import to_coco_dets


def test_to_coco_dets_remaps_label_xywh_and_filters():
    boxes = [[10, 20, 30, 60], [0, 0, 5, 5]]
    scores = [0.9, 0.01]
    labels = [1, 3]                          # 1-based model labels
    dets = to_coco_dets(7, boxes, scores, labels, conf=0.05)
    assert len(dets) == 1                    # low-score detection filtered out
    d = dets[0]
    assert d["image_id"] == 7
    assert d["category_id"] == 0             # 1-based label 1 -> 0-based GT category
    assert d["bbox"] == [10, 20, 20, 40]     # xyxy -> xywh
    assert d["score"] == 0.9


def test_to_coco_dets_empty_when_all_below_conf():
    assert to_coco_dets(1, [[0, 0, 1, 1]], [0.01], [2], conf=0.5) == []
