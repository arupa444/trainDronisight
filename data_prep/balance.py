import random
from collections import Counter


def cap_target(class_counts: dict) -> int:
    """Stable-frequency cap = the lowest kept-class instance count."""
    return min(class_counts.values())


def _counts(items, class_names):
    c = Counter()
    for it in items:
        for cls in it["classes"]:
            if cls in class_names:
                c[cls] += 1
    return {cls: c.get(cls, 0) for cls in class_names}


def select_balanced(items, class_names, enabled, seed, cap=None):
    """Greedily select images so each class's instance count approaches `cap`,
    admitting rare-class images first and never dropping an image while it still
    feeds an under-cap class. Returns the kept subset (order-independent)."""
    if not enabled:
        return list(items)
    if cap is None:
        cap = cap_target(_counts(items, class_names))

    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    # rarer images first: fewer total kept-class instances == more "specific"
    items.sort(key=lambda it: sum(c in class_names for c in it["classes"]))

    running = {cls: 0 for cls in class_names}
    kept = []
    for it in items:
        contributes = [c for c in it["classes"] if c in class_names and running[c] < cap]
        if not contributes:
            continue
        kept.append(it)
        for c in it["classes"]:
            if c in class_names:
                running[c] += 1
    return kept


def sample_weights(items, class_names) -> list:
    """Per-image inverse-frequency weight (for 'train on all data + weighted sampling').

    The weight is driven by the image's RAREST contained class (max inverse
    frequency) so that an image gets a high weight whenever it contains a rare
    class, regardless of how many common-class instances it also carries.
    """
    totals = _counts(items, class_names)
    inv = {cls: (1.0 / n if n else 0.0) for cls, n in totals.items()}
    weights = []
    for it in items:
        contribs = [inv.get(c, 0.0) for c in it["classes"] if c in class_names]
        weights.append(max(contribs) if contribs else 0.0)
    return weights
