from shared import config

def test_class_sets_are_canonical():
    assert config.POLE_CLASSES == ["pole"]
    # unified component detector: 8 types incl all 3 crossarm kinds together
    assert config.COMPONENT_CLASSES == ["wire", "h_insulator", "v_insulator", "crossarm_stright",
                                        "top_crossarm", "om_crossarm", "vegetation", "rust"]
    assert set(config.SUBSETS) == {"pole", "component", "cond_v_insulator", "cond_h_insulator",
                                   "cond_straight_crossarm", "cond_top_crossarm",
                                   "cond_om_crossarm", "cond_wire"}
    assert config.SUBSET_SOURCE_DIRS["component"] == config.SOURCE_DIRS         # mem captures
    assert config.SUBSET_SOURCE_DIRS["cond_wire"] == config.CONDITION_SOURCE_DIRS  # 6th-june
    assert config.BALANCE_TARGET["component"] == 1500
    assert config.BALANCE_TARGET["cond_v_insulator"] == 400
    # om_crossarm now has a band condition class (was previously dropped)
    assert config.COND_OM_CROSSARM_CLASSES == ["om_crossarm_normal", "om_crossarm_band"]


def test_condition_specialists_route_and_partition():
    # every detector class with conditions routes to a real cond_* subset
    for comp, model in config.COMPONENT_TO_CONDITION_MODEL.items():
        assert model in config.COND_SUBSETS
        # the derived family-filter classes equal that model's class list
        assert config.COMPONENT_TO_CONDITIONS[comp] == config.SUBSET_CLASSES[model]
    # vegetation/rust have NO condition family
    assert "vegetation" not in config.COMPONENT_TO_CONDITION_MODEL
    assert "rust" not in config.COMPONENT_TO_CONDITION_MODEL
    # naming bridge: detector crossarm_stright -> straight_crossarm condition family
    assert config.COMPONENT_TO_CONDITION_MODEL["crossarm_stright"] == "cond_straight_crossarm"
    # the 6 families together cover every condition class exactly once (no overlap)
    flat = [c for s in config.COND_SUBSETS for c in config.SUBSET_CLASSES[s]]
    assert len(flat) == len(set(flat))


def test_crop_alignment_modes():
    # `component` crops to the pole (anchor); each cond_* crops to the component itself (self)
    assert config.CROP_ALIGN["component"][0] == "anchor"
    for s in config.COND_SUBSETS:
        assert config.CROP_ALIGN[s][0] == "self"
        assert config.CROP_ALIGN[s][2] == config.CONDITION_CROP_PAD
    assert "pole" not in config.CROP_ALIGN                  # pole trains on the full frame
    assert config.CROP_ALIGN["component"][2] == config.POLE_CROP_PAD

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
