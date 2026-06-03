import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn(num_classes: int):
    """COCO-pretrained Faster R-CNN with the head resized to num_classes+1 (background)."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model
