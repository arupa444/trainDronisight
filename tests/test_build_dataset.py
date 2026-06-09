from data_prep.build_dataset import sample_class_list, dataset_version_hash, output_key

def test_output_key_namespaces_by_source():
    assert output_key("mem3", "DJI_x") == "mem3_DJI_x"

def test_sample_class_list_for_each_subset():
    from shared import config
    assert sample_class_list("pole") == config.POLE_CLASSES
    assert sample_class_list("component") == config.COMPONENT_CLASSES
    assert sample_class_list("cond_v_insulator") == config.COND_V_INSULATOR_CLASSES

def test_version_hash_is_stable_and_order_independent():
    a = dataset_version_hash(["mem2/a.JPG", "mem3/b.JPG"])
    b = dataset_version_hash(["mem3/b.JPG", "mem2/a.JPG"])
    assert a == b and len(a) == 12

def test_yolo_label_paths_mirror_image_variants():
    # Ultralytics maps images/<split>/<variant>/x.jpg -> labels/<split>/<variant>/x.txt,
    # so labels must mirror the variant subdir, not sit flat under labels/<split>/.
    from data_prep.build_dataset import yolo_label_paths
    paths = [str(p) for p in yolo_label_paths("pole", "train", "mem2_x")]
    assert any(p.endswith("labels/train/orig/mem2_x.txt") for p in paths)
    assert any(p.endswith("labels/train/clahe/mem2_x.txt") for p in paths)
    assert len(paths) == 2

def test_clean_appledouble_removes_sidecars(tmp_path):
    from data_prep.build_dataset import clean_appledouble
    (tmp_path / "real.jpg").write_bytes(b"x")
    sub = tmp_path / "images"; sub.mkdir()
    (sub / "._real.jpg").write_bytes(b"x"); (sub / "real.jpg").write_bytes(b"x")
    removed = clean_appledouble(tmp_path)
    assert removed == 1
    assert (tmp_path / "real.jpg").exists() and not (sub / "._real.jpg").exists()
