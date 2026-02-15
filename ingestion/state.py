import json
from pathlib import Path
from typing import Dict
from pydantic import TypeAdapter

# STORAGE = Path("storage")
# SOURCES = STORAGE / "sources.jsonl"
# DOCUMENTS = STORAGE / "documents.jsonl"
# UNITS = STORAGE / "content_units.jsonl"

# STORAGE.mkdir(exist_ok=True)
# SOURCES.touch(exist_ok=True)
# DOCUMENTS.touch(exist_ok=True)
# UNITS.touch(exist_ok=True)

# Paths

BASE_DIR_NEW = Path(__file__).resolve().parents[1]  # stable path (very important)

STORAGE = BASE_DIR_NEW / "storage"
ARTIFACT_DIR = STORAGE / "artifacts"
IMAGE_DIR = ARTIFACT_DIR / "images"

SOURCES = STORAGE / "sources.jsonl"
DOCUMENTS = STORAGE / "documents.jsonl"
UNITS = STORAGE / "content_units.jsonl"

# Initialization (idempotent, safe to call anytime)

def ensure_storage():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for file in (SOURCES, DOCUMENTS, UNITS):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch(exist_ok=True)

ensure_storage()

def _load_index(path: Path, key: str) -> Dict[str, dict]:
    ensure_storage()
    index = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                index[obj[key]] = obj
    return index


def load_source_index() -> Dict[str, dict]:
    ensure_storage()
    return _load_index(SOURCES, "file_hash")


def append_jsonl(path: Path, obj: dict):
    ensure_storage()
    json_str = TypeAdapter(dict).dump_json(obj).decode("utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(json_str + "\n")