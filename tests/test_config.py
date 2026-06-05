from shared import config

def test_class_sets_are_canonical():
    assert config.POLE_CLASSES == ["pole"]
    assert config.COMPONENT_ABOVE_CLASSES == ["wire", "h_insulator", "v_insulator", "crossarm_stright"]
    assert config.COMPONENT_BELOW_CLASSES == ["vegetation", "top_crossarm", "om_crossarm", "rust"]
    assert set(config.SUBSET_CLASSES) == {"pole", "component_above_1000", "component_below_1000"}

def test_split_ratios_sum_to_one():
    assert abs(sum(config.SPLIT_RATIOS.values()) - 1.0) < 1e-9
    assert set(config.SPLIT_RATIOS) == {"train", "val", "test"}

def test_source_dirs_match_configured_folders():
    names = [p.name for p in config.SOURCE_DIRS]
    assert names == config._SOURCE_FOLDER_NAMES
    assert len(names) == 11
    assert "mem 7.1 5th june" in names and "mem10" in names

def test_seed_is_fixed():
    assert isinstance(config.SEED, int)
