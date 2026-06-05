"""Helpers used inside the Colab notebooks. Pure/path logic is unit-tested;
the Colab-only side effects (drive.mount) live in tiny wrappers."""
import shutil
import zipfile
from pathlib import Path

DRIVE_ROOT = "/content/drive/MyDrive/dronisight"


def drive_db_zip(db_name: str, drive_root=DRIVE_ROOT) -> str:
    return f"{drive_root}/{db_name}.zip"


def drive_runs_dir(drive_root=DRIVE_ROOT) -> str:
    """Where trained outputs (weights, plots, results.csv) are mirrored on Drive."""
    return f"{drive_root}/runs"


def _copy_tree(src, dst) -> int:
    """Copy every file under src into dst (merging), preserving structure. Returns count."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return 0
    n = 0
    for p in src.rglob("*"):
        if p.is_file():
            out = Path(dst) / p.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            n += 1
    return n


def save_runs_to_drive(local_runs="runs", drive_root=DRIVE_ROOT) -> str:
    """Persist the local runs/ tree to Drive so weights survive the ephemeral runtime."""
    dest = drive_runs_dir(drive_root)
    _copy_tree(local_runs, dest)
    return dest


def restore_runs_from_drive(drive_root=DRIVE_ROOT, local_runs="runs") -> int:
    """Pull a previously-saved runs/ tree from Drive into the local session (for inference
    in a fresh runtime). Returns the number of files restored (0 if nothing on Drive)."""
    return _copy_tree(drive_runs_dir(drive_root), local_runs)


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
