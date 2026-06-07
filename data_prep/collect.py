from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    image: Path
    xml: Path
    source: str  # mem folder name


def collect_samples(source_dirs) -> list:
    """Find every image that has a matching .xml across the given source dirs.

    The label is normally a sibling (same dir); falls back to the PARENT dir by stem to
    handle folders like 6thMem2AllTeam1 where the XMLs sit one level above the images.
    """
    samples = []
    for d in source_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for img in sorted(d.iterdir()):
            if img.name.startswith("._"):  # macOS AppleDouble sidecar, not real data
                continue
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            xml = img.with_suffix(".xml")
            if not xml.exists():
                alt = img.parent.parent / f"{img.stem}.xml"   # XML one level up
                if alt.exists():
                    xml = alt
            if xml.exists():
                samples.append(Sample(image=img, xml=xml, source=d.name))
    return samples
