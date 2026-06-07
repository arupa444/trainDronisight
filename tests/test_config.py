from shared import config

def test_class_sets_are_canonical():
    assert config.POLE_CLASSES == ["pole"]
    assert config.COMPONENT_ABOVE_CLASSES == ["wire", "h_insulator", "v_insulator", "crossarm_stright"]
    assert config.COMPONENT_BELOW_CLASSES == ["vegetation", "top_crossarm", "om_crossarm", "rust"]
    assert len(config.COMPONENT_CLASSIFICATION_CLASSES) == 14
    assert set(config.SUBSET_CLASSES) == {"pole", "component_above_1000",
                                          "component_below_1000", "component_classification"}
    # component_classification draws from the 6th-june folders, not the mem captures
    assert config.SUBSET_SOURCE_DIRS["component_classification"] == config.CONDITION_SOURCE_DIRS
    assert config.SUBSET_SOURCE_DIRS["pole"] == config.SOURCE_DIRS
    assert config.BALANCE_TARGET["component_classification"] == 400

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
