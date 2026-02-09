import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """
    Compute SHA-256 hash of raw binary data.

    Use cases:
    - PDF file bytes
    - Image bytes
    - Any binary artifact

    Why:
    - Guarantees idempotent ingestion
    - Any byte-level change produces a new hash
    - Safe for reload and deduplication

    DO NOT:
    - Convert bytes to string before hashing
    """
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """
    Compute SHA-256 hash of textual data.

    Input MUST be:
    - Unicode-normalized
    - Whitespace-normalized
    - UTF-8 encodable

    Use cases:
    - Document-level content hashing
    - Chunk hashing
    - ContentUnit hashing

    Why:
    - Stable across restarts
    - Independent of memory/layout
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def content_hash(unit_type: str, extraction_method: str, content: Any) -> str:
    base = f"{unit_type}|{extraction_method}|{stable_json(content)}"
    return sha256_text(base)