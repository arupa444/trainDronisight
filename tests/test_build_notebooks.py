# tests/test_build_notebooks.py
import nbformat
from pathlib import Path
from notebooks.build_notebooks import build_all, NOTEBOOKS

def test_build_all_writes_valid_notebooks(tmp_path):
    paths = build_all(out_dir=tmp_path)
    assert len(paths) == len(NOTEBOOKS)
    for p in paths:
        nb = nbformat.read(p, as_version=4)   # raises if invalid
        nbformat.validate(nb)
        assert len(nb.cells) >= 3

def test_each_notebook_has_a_gpu_check_and_install():
    from notebooks.build_notebooks import NOTEBOOKS
    for spec in NOTEBOOKS.values():
        joined = "\n".join(spec)
        assert "nvidia-smi" in joined or "torch.cuda" in joined
        assert "uv pip install" in joined
