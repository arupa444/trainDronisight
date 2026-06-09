from types import SimpleNamespace
from train_yolo.eval_yolo import format_report


def test_format_report_overall_and_per_class():
    box = SimpleNamespace(map50=0.61, map=0.50, mp=0.55, mr=0.63,
                          ap_class_index=[0, 1], ap50=[0.7, 0.52], p=[0.6, 0.5], r=[0.65, 0.6])
    res = SimpleNamespace(box=box, names={0: "v_insulator", 1: "wire"})
    out = format_report(res, "w.pt", "component_above_1000", "val")
    assert "### component_above_1000 [val]" in out
    assert "mAP50=0.610" in out and "mAP50-95=0.500" in out
    assert "v_insulator" in out and "AP50=0.700" in out
    assert "wire" in out and "AP50=0.520" in out
    # one header + one overall line + one line per class
    assert len(out.splitlines()) == 4
