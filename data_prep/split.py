import random
from collections import defaultdict


def grouped_split(items, ratios, seed):
    """Split items into train/val/test by GROUP (never splitting a group),
    stratified per source so every location appears in train.

    items: list of dicts with keys 'group' and 'source'.
    """
    rng = random.Random(seed)
    # group -> its source (groups are source-namespaced, so unique per source)
    groups_by_source = defaultdict(list)
    members = defaultdict(list)
    for it in items:
        members[it["group"]].append(it)
    for g, its in members.items():
        groups_by_source[its[0]["source"]].append(g)

    out = {"train": [], "val": [], "test": []}
    for source, groups in groups_by_source.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        n = len(groups)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        # guarantee >=1 train group per source when possible
        n_train = max(n_train, 1) if n else 0
        buckets = {
            "train": groups[:n_train],
            "val": groups[n_train:n_train + n_val],
            "test": groups[n_train + n_val:],
        }
        for split_name, gs in buckets.items():
            for g in gs:
                out[split_name].extend(members[g])
    return out
