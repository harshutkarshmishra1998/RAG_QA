import json
from pathlib import Path
import faiss
import numpy as np

from clean_normalize.clean_normalize import clean_content_units_file
from chunk.chunk import chunk_content_units
from embedding.faiss_embedder import _embed_texts

# CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE = PROJECT_ROOT / "storage"

UNITS_FILE = STORAGE / "content_units.jsonl"
CLEANED_FILE = STORAGE / "content_units_cleaned.jsonl"
CHUNKED_FILE = STORAGE / "content_units_cleaned_chunked.jsonl"
FAISS_FILE = STORAGE / "content_units.faiss"
STATE_FILE = STORAGE / "pipeline_state.json"

EMBEDDING_DIM = 3072


# STATE MANAGEMENT

def load_state():
    if not STATE_FILE.exists():
        return {
            "processed_doc_ids": [],
            "processed_chunk_ids": []
        }

    state = json.loads(STATE_FILE.read_text())

    # Ensure keys always exist
    state.setdefault("processed_doc_ids", [])
    state.setdefault("processed_chunk_ids", [])

    return state


def save_state(state):
    STORAGE.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# INCREMENTAL CLEAN

def clean_new_docs():

    state = load_state()

    if not UNITS_FILE.exists():
        return []

    with UNITS_FILE.open("r", encoding="utf-8") as f:
        units = [json.loads(line) for line in f]

    new_units = [
        u for u in units
        if u["doc_id"] not in state["processed_doc_ids"]
    ]

    if not new_units:
        return []

    temp_input = STORAGE / "temp_new_units.jsonl"
    with temp_input.open("w", encoding="utf-8") as f:
        for u in new_units:
            f.write(json.dumps(u) + "\n")

    cleaned_path = clean_content_units_file(temp_input)

    # Append to master cleaned file
    with CLEANED_FILE.open("a", encoding="utf-8") as master:
        with cleaned_path.open("r", encoding="utf-8") as cleaned:
            master.write(cleaned.read())

    return list({u["doc_id"] for u in new_units})


# INCREMENTAL CHUNK

def chunk_new_docs(new_doc_ids):

    if not new_doc_ids:
        return []

    with CLEANED_FILE.open("r", encoding="utf-8") as f:
        cleaned_units = [json.loads(line) for line in f]

    new_units = [
        u for u in cleaned_units
        if u["doc_id"] in new_doc_ids
    ]

    if not new_units:
        return []

    temp_input = STORAGE / "temp_new_cleaned.jsonl"
    with temp_input.open("w", encoding="utf-8") as f:
        for u in new_units:
            f.write(json.dumps(u) + "\n")

    chunked_path = chunk_content_units(temp_input)

    # Append to master chunk file
    with CHUNKED_FILE.open("a", encoding="utf-8") as master:
        with chunked_path.open("r", encoding="utf-8") as chunks:
            master.write(chunks.read())

    with chunked_path.open("r", encoding="utf-8") as f:
        return [json.loads(line)["chunk_id"] for line in f]


# INCREMENTAL EMBED

def embed_new_chunks(new_chunk_ids):

    if not new_chunk_ids:
        return

    with CHUNKED_FILE.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    new_chunks = [
        c for c in chunks
        if c["chunk_id"] in new_chunk_ids
    ]

    if not new_chunks:
        return

    texts = [c["text"] for c in new_chunks]
    vectors = _embed_texts(texts)

    if FAISS_FILE.exists():
        index = faiss.read_index(str(FAISS_FILE))
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

    index.add(vectors)  # type: ignore
    faiss.write_index(index, str(FAISS_FILE))

# TEMP FILE CLEANUP

def cleanup_temp_files():

    deleted_files = []

    for file in STORAGE.glob("temp_new_*.jsonl"):
        try:
            file.unlink()
            deleted_files.append(file.name)
        except Exception:
            pass

    return deleted_files

# MASTER PIPELINE

def run_incremental_pipeline():

    state = load_state()

    new_doc_ids = clean_new_docs()
    new_chunk_ids = chunk_new_docs(new_doc_ids)
    embed_new_chunks(new_chunk_ids)

    # Update state
    state["processed_doc_ids"].extend(new_doc_ids)
    state["processed_chunk_ids"].extend(new_chunk_ids)

    save_state(state)

    # Cleanup temp files
    deleted = cleanup_temp_files()

    return {
        "new_docs": new_doc_ids,
        "new_chunks": new_chunk_ids,
        "deleted_temp_files": deleted
    }