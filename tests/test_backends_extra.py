# tests/test_backends_extra.py
import torch
from inference.backends import parse_torchvision_output, Detection

def test_parse_torchvision_output_1based_to_names():
    # 0.75 is exactly representable in float32 so tensor->list->float round-trips
    # cleanly (0.7 is not, which spuriously fails dataclass equality).
    out = {"boxes": torch.tensor([[0., 0., 5., 5.]]),
           "scores": torch.tensor([0.75]),
           "labels": torch.tensor([1])}  # label 1 -> class_names[0]
    dets = parse_torchvision_output(out, class_names=["wire", "h_insulator"], conf=0.5)
    assert dets == [Detection("wire", 0.75, (0.0, 0.0, 5.0, 5.0))]

def test_parse_drops_below_conf():
    out = {"boxes": torch.tensor([[0., 0., 5., 5.]]),
           "scores": torch.tensor([0.2]), "labels": torch.tensor([1])}
    assert parse_torchvision_output(out, ["wire"], conf=0.5) == []
