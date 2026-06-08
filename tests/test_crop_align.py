from shared.labels import Box, Annotation
from data_prep.crop_align import make_crops, _visible_frac, _remap_clip


def test_anchor_crops_to_pole_and_remaps_components():
    # full frame: a pole + two components on it, plus a far-away component off the pole
    ann = Annotation(1000, 1000, [
        Box("pole", 400, 100, 600, 900),
        Box("wire", 420, 200, 580, 230),          # on the pole
        Box("h_insulator", 440, 400, 560, 460),   # on the pole
        Box("wire", 50, 50, 90, 70),               # far away, off the pole crop
    ])
    crops = make_crops(ann, ["wire", "h_insulator", "v_insulator", "crossarm_stright"],
                       mode="anchor", anchor_classes=("pole",), pad_frac=0.05, min_visible=0.3)
    assert len(crops) == 1
    crop, cann = crops[0]
    # crop is the padded pole box, clipped
    assert crop[0] < 400 and crop[2] > 600
    names = sorted(b.name for b in cann.boxes)
    assert names == ["h_insulator", "wire"]        # the far wire is excluded
    # boxes are remapped to crop-local coords (origin shifted)
    x0, y0 = crop[0], crop[1]
    on_pole_wire = [b for b in cann.boxes if b.name == "wire"][0]
    assert on_pole_wire.xmin == 420 - x0 and on_pole_wire.ymin == 200 - y0
    # crop annotation dims == crop size
    assert cann.width == crop[2] - crop[0] and cann.height == crop[3] - crop[1]


def test_anchor_multiple_poles_make_multiple_crops():
    ann = Annotation(1000, 500, [
        Box("pole", 50, 50, 150, 450), Box("wire", 60, 100, 140, 120),
        Box("pole", 700, 50, 800, 450), Box("v_insulator", 710, 200, 790, 260),
    ])
    crops = make_crops(ann, ["wire", "v_insulator"], mode="anchor",
                       anchor_classes=("pole",), pad_frac=0.0, min_visible=0.3)
    assert len(crops) == 2


def test_anchor_fallback_when_no_pole():
    # no pole box -> fall back to the union of in-subset boxes (still crop-scale, not full frame)
    ann = Annotation(2000, 2000, [Box("rust", 100, 100, 140, 140), Box("vegetation", 160, 160, 220, 220)])
    crops = make_crops(ann, ["rust", "vegetation", "top_crossarm", "om_crossarm"],
                       mode="anchor", anchor_classes=("pole",), pad_frac=0.0, min_visible=0.3)
    assert len(crops) == 1
    crop, cann = crops[0]
    assert crop == (100, 100, 220, 220)            # tight union, no full-frame
    assert sorted(b.name for b in cann.boxes) == ["rust", "vegetation"]


def test_self_mode_one_crop_per_component():
    ann = Annotation(1000, 1000, [
        Box("v_insulator_normal", 100, 100, 200, 200),
        Box("h_insulator_broken", 700, 700, 800, 820),
    ])
    crops = make_crops(ann, ["v_insulator_normal", "h_insulator_broken"], mode="self",
                       pad_frac=0.1, min_visible=0.5)
    assert len(crops) == 2
    # each crop is centered on its target; the target nearly fills the crop
    for crop, cann in crops:
        assert any(b for b in cann.boxes)
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        assert cann.width == cw and cann.height == ch


def test_low_visibility_box_dropped():
    ann = Annotation(1000, 1000, [
        Box("pole", 400, 100, 600, 900),
        Box("wire", 590, 200, 900, 220),   # mostly OUTSIDE the pole crop -> < min_visible
    ])
    crops = make_crops(ann, ["wire"], mode="anchor", anchor_classes=("pole",),
                       pad_frac=0.0, min_visible=0.6)
    # the only component is <60% visible in the pole crop -> no boxes -> no crop emitted
    assert crops == []


def test_helpers():
    a = Box("x", 0, 0, 100, 100)
    assert _visible_frac(a, (0, 0, 50, 100)) == 0.5
    assert _visible_frac(a, (200, 200, 300, 300)) == 0.0
    r = _remap_clip(Box("x", 60, 60, 140, 140), (50, 50, 150, 150))
    assert (r.xmin, r.ymin, r.xmax, r.ymax) == (10, 10, 90, 90)  # clipped + shifted
