from pathlib import Path
from data_prep.collect import collect_samples

def test_pairs_only_images_with_xml(tmp_path):
    d = tmp_path / "mem2"
    d.mkdir()
    (d / "a.JPG").write_bytes(b"x")
    (d / "a.xml").write_text("<annotation/>")
    (d / "b.JPG").write_bytes(b"x")          # no xml -> skipped
    (d / "c.xml").write_text("<annotation/>")  # no image -> skipped
    samples = collect_samples([d])
    assert [s.image.name for s in samples] == ["a.JPG"]
    assert samples[0].xml.name == "a.xml"
    assert samples[0].source == "mem2"
