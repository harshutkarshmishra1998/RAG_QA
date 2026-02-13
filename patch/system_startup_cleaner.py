from pathlib import Path
import shutil

def _clear_directory_contents(dir_path: Path):
    if not dir_path.exists():
        return

    for item in dir_path.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"Failed to remove {item}: {e}")


def clean_data_directories(project_root: Path):
    """
    Cleans contents of:
        - storage/
        - uploaded_files/

    Keeps directories themselves.
    Runs once at app startup or when manually triggered.
    """
    storage = project_root / "storage"
    uploads = project_root / "uploaded_files"

    _clear_directory_contents(storage)
    _clear_directory_contents(uploads)

    print("Startup cleanup completed.")