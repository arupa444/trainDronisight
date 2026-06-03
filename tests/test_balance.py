from data_prep.balance import cap_target, select_balanced, sample_weights

def test_cap_target_is_min_kept_count():
    counts = {"wire": 3500, "h_insulator": 3000, "crossarm_stright": 1661}
    assert cap_target(counts) == 1661

def test_select_keeps_all_when_disabled():
    items = [{"name": "a", "classes": ["wire"]}, {"name": "b", "classes": ["wire"]}]
    kept = select_balanced(items, class_names=["wire"], enabled=False, seed=1)
    assert len(kept) == 2

def test_select_respects_cap_for_overrepresented_class():
    # 5 images each with one 'wire'; cap=2 -> keep 2
    items = [{"name": f"w{i}", "classes": ["wire"]} for i in range(5)]
    kept = select_balanced(items, class_names=["wire"], enabled=True, seed=1, cap=2)
    assert sum("wire" in it["classes"] for it in kept) == 2

def test_select_prioritizes_rare_class_images():
    items = [{"name": "rareimg", "classes": ["crossarm_stright"]}] + \
            [{"name": f"w{i}", "classes": ["wire"]} for i in range(5)]
    kept = select_balanced(items, class_names=["wire", "crossarm_stright"],
                           enabled=True, seed=1, cap=1)
    assert any(it["name"] == "rareimg" for it in kept)

def test_sample_weights_are_inverse_frequency():
    items = [{"classes": ["wire", "wire"]}, {"classes": ["crossarm_stright"]}]
    w = sample_weights(items, ["wire", "crossarm_stright"])
    assert w[1] > w[0]  # the rare-class image gets a higher weight
