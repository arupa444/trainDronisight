# tests/test_colab_utils.py
from pathlib import Path
from unittest import mock
from notebooks import colab_utils

def test_drive_db_path():
    p = colab_utils.drive_db_zip("yolo_train_db", drive_root="/content/drive/MyDrive/dronisight")
    assert p == "/content/drive/MyDrive/dronisight/yolo_train_db.zip"

def test_ensure_dataset_unzips_when_missing(tmp_path):
    zip_path = tmp_path / "yolo_train_db.zip"
    zip_path.write_bytes(b"PK")  # pretend zip exists
    dest = tmp_path / "data"
    with mock.patch.object(colab_utils, "_unzip") as uz:
        out = colab_utils.ensure_dataset(str(zip_path), str(dest))
    uz.assert_called_once_with(str(zip_path), str(dest))
    assert out == str(dest)

def test_ensure_dataset_skips_when_present(tmp_path):
    dest = tmp_path / "data" / "yolo_train_db"
    dest.mkdir(parents=True)
    (dest / "marker").write_text("x")  # already unzipped
    with mock.patch.object(colab_utils, "_unzip") as uz:
        colab_utils.ensure_dataset(str(tmp_path / "yolo_train_db.zip"),
                                   str(tmp_path / "data"), expect_subdir="yolo_train_db")
    uz.assert_not_called()

def test_repo_clone_command():
    cmd = colab_utils.repo_setup_cmd("https://github.com/u/trainDronisight.git", "/content/repo")
    assert "git clone" in cmd and "/content/repo" in cmd


def test_drive_runs_dir():
    assert colab_utils.drive_runs_dir("/d/dronisight") == "/d/dronisight/runs"


def test_save_runs_to_drive_copies_tree(tmp_path):
    runs = tmp_path / "runs"
    (runs / "pole" / "yolo" / "weights").mkdir(parents=True)
    (runs / "pole" / "yolo" / "weights" / "best.pt").write_bytes(b"W")
    drive = tmp_path / "drive"
    dest = colab_utils.save_runs_to_drive(local_runs=str(runs), drive_root=str(drive))
    assert dest == f"{drive}/runs"
    assert (drive / "runs" / "pole" / "yolo" / "weights" / "best.pt").read_bytes() == b"W"


def test_restore_runs_from_drive_copies_back(tmp_path):
    drive = tmp_path / "drive"
    (drive / "runs" / "components" / "yolo" / "weights").mkdir(parents=True)
    (drive / "runs" / "components" / "yolo" / "weights" / "best.pt").write_bytes(b"X")
    local = tmp_path / "repo" / "runs"
    n = colab_utils.restore_runs_from_drive(drive_root=str(drive), local_runs=str(local))
    assert n == 1
    assert (local / "components" / "yolo" / "weights" / "best.pt").read_bytes() == b"X"


def test_restore_runs_from_drive_noop_when_absent(tmp_path):
    assert colab_utils.restore_runs_from_drive(drive_root=str(tmp_path / "nope"),
                                               local_runs=str(tmp_path / "runs")) == 0
