from shared import config

def test_class_sets_are_canonical():
    assert config.POLE_CLASSES == ["pole"]
    assert config.COMPONENT_ABOVE_CLASSES == ["wire", "h_insulator", "v_insulator", "crossarm_stright"]
    assert config.COMPONENT_BELOW_CLASSES == ["vegetation", "top_crossarm", "om_crossarm", "rust"]
    assert len(config.COMPONENT_CLASSIFICATION_CLASSES) == 14
    assert set(config.BASE_SUBSETS) == {"pole", "component_above_1000",
                                        "component_below_1000", "component_classification"}
    # component_classification draws from the 6th-june folders, not the mem captures
    assert config.SUBSET_SOURCE_DIRS["component_classification"] == config.CONDITION_SOURCE_DIRS
    assert config.SUBSET_SOURCE_DIRS["pole"] == config.SOURCE_DIRS
    assert config.BALANCE_TARGET["component_classification"] == 400


def test_component_to_conditions_partitions_all_14_classes():
    # every mapped condition is a real condition class; the 6 families partition all 14 exactly
    flat = [c for v in config.COMPONENT_TO_CONDITIONS.values() for c in v]
    assert set(flat) == set(config.COMPONENT_CLASSIFICATION_CLASSES)   # exact coverage
    assert len(flat) == len(config.COMPONENT_CLASSIFICATION_CLASSES)   # no overlaps (14, once each)
    # keys are real component classes (above ∪ below); vegetation/rust have NO condition family
    comp_classes = set(config.COMPONENT_ABOVE_CLASSES) | set(config.COMPONENT_BELOW_CLASSES)
    assert set(config.COMPONENT_TO_CONDITIONS) <= comp_classes
    assert "vegetation" not in config.COMPONENT_TO_CONDITIONS
    assert "rust" not in config.COMPONENT_TO_CONDITIONS
    # the naming bridge: the detector's crossarm_stright maps to straight_crossarm_* conditions
    assert config.COMPONENT_TO_CONDITIONS["crossarm_stright"] == \
        ["straight_crossarm_normal", "straight_crossarm_band"]


def test_crop_subsets_share_base_policy():
    # each <base>_crop subset mirrors its base's class list and resolves back via base_subset()
    assert config.CROP_SUBSETS == ["component_above_1000_crop", "component_below_1000_crop",
                                   "component_classification_crop"]
    for cs in config.CROP_SUBSETS:
        base = config.base_subset(cs)
        assert base + "_crop" == cs
        assert config.SUBSET_CLASSES[cs] == config.SUBSET_CLASSES[base]
    assert config.base_subset("pole") == "pole"          # non-crop passes through
    # crop modes: above/below crop to the pole anchor, condition crops to the component itself
    assert config.CROP_ALIGN["component_above_1000"][0] == "anchor"
    assert config.CROP_ALIGN["component_classification"][0] == "self"

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
