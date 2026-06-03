from data_prep.build_dataset import sample_class_list, dataset_version_hash, output_key

def test_output_key_namespaces_by_source():
    assert output_key("mem3", "DJI_x") == "mem3_DJI_x"

def test_sample_class_list_pole_vs_components():
    from shared import config
    assert sample_class_list("pole") == config.POLE_CLASSES
    assert sample_class_list("components") == config.COMPONENT_CLASSES

def test_version_hash_is_stable_and_order_independent():
    a = dataset_version_hash(["mem2/a.JPG", "mem3/b.JPG"])
    b = dataset_version_hash(["mem3/b.JPG", "mem2/a.JPG"])
    assert a == b and len(a) == 12
