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

def test_excluded_classes_return_none():
    for raw in ["rust", "om_crossarm", "top_crossarm", "vegetation"]:
        assert normalize_class_name(raw) is None

def test_unknown_returns_none():
    assert normalize_class_name("banana") is None
