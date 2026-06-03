# tests/test_frcnn_model.py
from train_faster_rcnn.model import build_fasterrcnn

def test_head_has_num_classes_plus_background():
    model = build_fasterrcnn(num_classes=4)  # 4 components
    # cls_score out_features == num_classes + 1 (background)
    assert model.roi_heads.box_predictor.cls_score.out_features == 5
