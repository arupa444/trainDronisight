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
