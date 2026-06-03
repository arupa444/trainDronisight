from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    image: Path
    xml: Path
    source: str  # mem folder name


def collect_samples(source_dirs) -> list:
    """Find every image that has a sibling .xml across the given source dirs."""
    samples = []
    for d in source_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for img in sorted(d.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            xml = img.with_suffix(".xml")
            if xml.exists():
                samples.append(Sample(image=img, xml=xml, source=d.name))
    return samples
