import hashlib
import json
from schema.ingestion_schema import UnitType, ExtractionMethod


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(unit_type: UnitType, method: ExtractionMethod, content) -> str:
    payload = {
        "unit_type": unit_type.value,
        "method": method.value,
        "content": content,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()