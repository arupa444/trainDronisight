from shared import config

def test_class_sets_are_canonical():
    assert config.POLE_CLASSES == ["pole"]
    assert config.COMPONENT_CLASSES == ["wire", "h_insulator", "v_insulator", "crossarm_stright"]

def test_split_ratios_sum_to_one():
    assert abs(sum(config.SPLIT_RATIOS.values()) - 1.0) < 1e-9
    assert set(config.SPLIT_RATIOS) == {"train", "val", "test"}

def test_source_dirs_are_mem2_to_mem8():
    assert [p.name for p in config.SOURCE_DIRS] == [f"mem{i}" for i in range(2, 9)]

def test_seed_is_fixed():
    assert isinstance(config.SEED, int)
