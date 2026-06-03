"""Helpers used inside the Colab notebooks. Pure/path logic is unit-tested;
the Colab-only side effects (drive.mount) live in tiny wrappers."""
import zipfile
from pathlib import Path


def drive_db_zip(db_name: str, drive_root="/content/drive/MyDrive/dronisight") -> str:
    return f"{drive_root}/{db_name}.zip"


def _unzip(zip_path: str, dest: str):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def ensure_dataset(zip_path: str, dest: str, expect_subdir: str = None) -> str:
    """Unzip the DB to fast local storage if it isn't already there."""
    check = Path(dest) / expect_subdir if expect_subdir else Path(dest)
    if check.exists() and any(check.iterdir()):
        return dest
    Path(dest).mkdir(parents=True, exist_ok=True)
    _unzip(zip_path, dest)
    return dest


def repo_setup_cmd(repo_url: str, dest="/content/repo") -> str:
    return f"git clone {repo_url} {dest} || (cd {dest} && git pull)"


def mount_drive():  # pragma: no cover (Colab-only)
    from google.colab import drive
    drive.mount("/content/drive")
