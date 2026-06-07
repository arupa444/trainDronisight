import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn(num_classes: int, min_size: int = 800, max_size: int = 1333):
    """COCO-pretrained Faster R-CNN with the head resized to num_classes+1 (background).

    min_size/max_size set the internal GeneralizedRCNNTransform resize. The torchvision
    default (800/1333) shrinks thin wires/small insulators to a few pixels on these large
    drone frames -> small-object AP collapses. Raise min_size (e.g. 1333) so they survive.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT", min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model
