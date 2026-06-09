from shared.labels import normalize_class_name

def test_merges_all_crossarm_variants():
    for raw in ["crossarm_stright", "crossarm_Stright", "crossarmStright"]:
        assert normalize_class_name(raw) == "crossarm_stright"

def test_passes_through_kept_classes():
    for raw in ["pole", "wire", "h_insulator", "v_insulator"]:
        assert normalize_class_name(raw) == raw

def test_is_case_insensitive():
    assert normalize_class_name("H_Insulator") == "h_insulator"
    assert normalize_class_name(" WIRE ") == "wire"

def test_rare_classes_now_kept():
    # previously normalized to None; now trained as the component_below_1000 detector
    for raw in ["rust", "om_crossarm", "top_crossarm", "vegetation"]:
        assert normalize_class_name(raw) == raw

def test_condition_classes_kept_and_merged():
    # 14 condition classes pass through
    for raw in ["v_insulator_broken", "h_insulator_chip_off", "top_crossarm_normal", "cross_wire"]:
        assert normalize_class_name(raw) == raw
    # data-quality merges
    assert normalize_class_name("top_corssarm_normal") == "top_crossarm_normal"   # misspelling
    assert normalize_class_name("v_insulator_puncture") == "v_insulator_chip_off"
    assert normalize_class_name("h_insulator_puncture") == "h_insulator_chip_off"

def test_dropped_condition_labels_return_none():
    assert normalize_class_name("w") is None          # stray (8x) -> dropped
    # om_crossarm_band IS now kept (150 instances -> the om condition family)
    assert normalize_class_name("om_crossarm_band") == "om_crossarm_band"

def test_unknown_returns_none():
    assert normalize_class_name("banana") is None
    assert normalize_class_name(None) is None
