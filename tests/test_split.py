from data_prep.split import grouped_split

def _items():
    # 10 groups per source, 2 sources, 1 item each for simplicity
    items = []
    for src in ["mem2", "mem3"]:
        for g in range(10):
            items.append({"name": f"{src}_{g}", "group": f"{src}:{g}", "source": src})
    return items

def test_no_group_spans_two_splits():
    split = grouped_split(_items(), ratios={"train": .8, "val": .15, "test": .05}, seed=1)
    group_to_split = {}
    for s in ("train", "val", "test"):
        for it in split[s]:
            group_to_split.setdefault(it["group"], s)
            assert group_to_split[it["group"]] == s

def test_is_deterministic():
    a = grouped_split(_items(), {"train": .8, "val": .15, "test": .05}, seed=7)
    b = grouped_split(_items(), {"train": .8, "val": .15, "test": .05}, seed=7)
    assert [i["name"] for i in a["train"]] == [i["name"] for i in b["train"]]

def test_every_source_appears_in_train():
    split = grouped_split(_items(), {"train": .8, "val": .15, "test": .05}, seed=1)
    srcs = {it["source"] for it in split["train"]}
    assert srcs == {"mem2", "mem3"}
