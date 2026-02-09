import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from schema.ingestion_schema import Source, Document, ContentUnit

from ingestion.state import SOURCES, DOCUMENTS, UNITS


def _load(path: Path, model):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = model.model_validate_json(line)
                out[getattr(obj, model.__fields__.keys().__iter__().__next__())] = obj
    return out


def load_sources() -> Dict[str, Source]:
    return _load(SOURCES, Source)


def load_documents() -> Dict[str, Document]:
    return _load(DOCUMENTS, Document)


def load_content_units() -> Dict[str, List[ContentUnit]]:
    units = defaultdict(list)
    with UNITS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                u = ContentUnit.model_validate_json(line)
                units[u.doc_id].append(u)
    return units