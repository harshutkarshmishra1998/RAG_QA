from pathlib import Path
import json
from typing import Dict, List

from schema.ingestion_schema import Source, Document, ContentUnit
from ingestion.state import SOURCES, DOCUMENTS, UNITS


def _load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_sources() -> Dict[str, Source]:
    """
    Load all sources indexed by source_id
    """
    sources = {}
    for raw in _load_jsonl(SOURCES):
        src = Source.model_validate(raw)
        sources[src.source_id] = src
    return sources


def load_documents() -> Dict[str, Document]:
    """
    Load all documents indexed by doc_id
    """
    documents = {}
    for raw in _load_jsonl(DOCUMENTS):
        doc = Document.model_validate(raw)
        documents[doc.doc_id] = doc
    return documents


def load_content_units() -> Dict[str, List[ContentUnit]]:
    """
    Load content units grouped by doc_id
    """
    grouped: Dict[str, List[ContentUnit]] = {}

    for raw in _load_jsonl(UNITS):
        unit = ContentUnit.model_validate(raw)
        grouped.setdefault(unit.doc_id, []).append(unit)

    # Ensure deterministic ordering
    for units in grouped.values():
        units.sort(key=lambda u: u.order_index)

    return grouped