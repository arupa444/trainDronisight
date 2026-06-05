"""Drop re-annotated duplicate images that appear in more than one source folder.

Some flights were annotated twice (e.g. `mem7` and `mem 7.1 5th june` share 161 image
stems; 157 are byte-identical). For each configured (primary, secondary) folder pair, an
image present in BOTH (matched by filename stem) with an identical annotation hash is a
true duplicate -> keep the primary copy, drop the secondary. Different annotations for the
same stem are kept (they carry distinct source-namespaced keys downstream).
"""
from shared.labels import annotation_hash


def drop_duplicate_annotations(parsed, pairs):
    """parsed: {image_path: (sample, annotation)} where sample has `.source`.
    pairs: list of (primary_source, secondary_source) folder-name tuples.
    Returns (kept_parsed, n_dropped)."""
    primaries = {p for p, _ in pairs}
    secondary_to_primary = {sec: pri for pri, sec in pairs}

    # stem -> annotation hash, per primary source
    prim_hashes = {p: {} for p in primaries}
    for path, (s, ann) in parsed.items():
        if s.source in primaries:
            prim_hashes[s.source][path.stem] = annotation_hash(ann.boxes)

    kept, dropped = {}, 0
    for path, (s, ann) in parsed.items():
        pri = secondary_to_primary.get(s.source)
        if pri is not None:
            h = prim_hashes.get(pri, {}).get(path.stem)
            if h is not None and h == annotation_hash(ann.boxes):
                dropped += 1
                continue  # exact duplicate of the primary copy
        kept[path] = (s, ann)
    return kept, dropped
