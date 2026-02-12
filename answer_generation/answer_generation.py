import api_keys
import json
import hashlib
import ast
import re
from pathlib import Path
from typing import Dict, Any, List

from groq import Groq


# =====================================================
# CONFIG
# =====================================================

GROQ_MODEL = "llama-3.1-8b-instant"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_PATH = PROJECT_ROOT / "storage"

MEMORY_FILE = STORAGE_PATH / "retrieval_results.jsonl"
OUTPUT_FILE = STORAGE_PATH / "final_answers.json"

CONTENT_UNIT_FILE = STORAGE_PATH / "content_units.jsonl"
CHUNK_MAP_FILE = STORAGE_PATH / "content_units_cleaned_chunked.jsonl"
DOCUMENTS_FILE = STORAGE_PATH / "documents.jsonl"
SOURCES_FILE = STORAGE_PATH / "sources.jsonl"

client = Groq()

MAX_CHUNKS_INITIAL = 3
MAX_CHUNKS_EXPANDED = 6


# =====================================================
# FILE HELPERS
# =====================================================

def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_json_array_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            json.dump([], f)


def load_json_array(path: Path):
    ensure_json_array_file(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
    except json.JSONDecodeError:
        return []


def save_json_array(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


# =====================================================
# LOAD CONTENT UNITS
# =====================================================

def load_content_unit_index(path: Path):
    index = {}
    for row in load_jsonl(path):
        index[str(row["unit_id"])] = row
    print("Loaded content units:", len(index))
    return index


CONTENT_UNIT_INDEX = load_content_unit_index(CONTENT_UNIT_FILE)


# =====================================================
# LOAD CHUNK → UNIT MAP
# =====================================================

def load_chunk_unit_map(path: Path):
    mapping = {}
    for row in load_jsonl(path):
        mapping[str(row["chunk_id"])] = row.get("unit_ids", [])
    print("Loaded chunk→unit map:", len(mapping))
    return mapping


CHUNK_UNIT_MAP = load_chunk_unit_map(CHUNK_MAP_FILE)


# =====================================================
# LOAD DOCUMENT REGISTRY
# =====================================================

def load_document_registry(path: Path):
    registry = {}
    for row in load_jsonl(path):
        doc_id = row.get("doc_id") or row.get("id")
        source_id = (
            row.get("source_id")
            or row.get("source")
            or row.get("source_ref")
        )
        registry[doc_id] = source_id
    print("Loaded documents:", len(registry))
    return registry


DOCUMENT_REGISTRY = load_document_registry(DOCUMENTS_FILE)


# =====================================================
# LOAD SOURCE REGISTRY
# =====================================================

# 

def load_source_registry(path: Path):
    registry = {}

    for row in load_jsonl(path):
        source_id = row.get("source_id") or row.get("id")

        uri = row.get("source_uri")

        if uri:
            filename = Path(uri).stem
        else:
            filename = None

        registry[source_id] = filename

    print("Loaded sources:", len(registry))
    return registry


SOURCE_REGISTRY = load_source_registry(SOURCES_FILE)


# =====================================================
# RESOLVE DOCUMENT NAME
# =====================================================

def resolve_document_name(doc_id: str) -> str:
    source_id = DOCUMENT_REGISTRY.get(doc_id)
    if not source_id:
        return doc_id
    filename = SOURCE_REGISTRY.get(source_id)
    return filename or doc_id


# =====================================================
# TEXT NORMALIZATION
# =====================================================

def repair_chunk_text(text):
    if text is None:
        return ""

    if isinstance(text, list):
        text = " ".join(str(t) for t in text)

    elif isinstance(text, str):
        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    text = " ".join(str(t) for t in parsed)
            except Exception:
                pass

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s([,.;:])", r"\1", text)
    return text.strip()


def chunk_is_usable(text: str) -> bool:
    if not text or len(text) < 40:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio > 0.5


# =====================================================
# FUZZY MATCH SCORE
# =====================================================

def text_overlap_score(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


# =====================================================
# PROVENANCE RECONSTRUCTION
# =====================================================

def get_chunk_provenance(chunk):
    prov = {}

    unit_ids = chunk.get("unit_ids")
    if not unit_ids:
        unit_ids = CHUNK_UNIT_MAP.get(str(chunk.get("chunk_id")), [])

    # fuzzy recovery
    if not unit_ids:
        chunk_text = repair_chunk_text(chunk.get("text", "")).lower()
        best_uid = None
        best_score = 0

        for uid, unit in CONTENT_UNIT_INDEX.items():
            unit_text = repair_chunk_text(unit.get("content", "")).lower()
            score = text_overlap_score(chunk_text, unit_text)
            if score > best_score:
                best_score = score
                best_uid = uid

        if best_score > 0.30:
            unit_ids = [best_uid]

    if not unit_ids:
        return prov

    for uid in unit_ids:
        unit = CONTENT_UNIT_INDEX.get(str(uid))
        if not unit:
            continue

        doc_id = unit.get("doc_id")
        doc = resolve_document_name(doc_id)
        page = unit.get("page_start")

        if doc not in prov:
            prov[doc] = {"pages": set()}

        if page is not None:
            prov[doc]["pages"].add(page)

    return prov


def merge_provenance(bundle):
    merged = {}
    for item in bundle:
        prov = get_chunk_provenance(item["chunk"])
        for doc, data in prov.items():
            if doc not in merged:
                merged[doc] = {"pages": set()}
            merged[doc]["pages"].update(data["pages"])
    return merged


def format_page_ranges(pages):
    if not pages:
        return "unknown pages"
    pages = sorted(pages)
    ranges = []
    start = pages[0]
    prev = pages[0]

    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = p
            prev = p
    ranges.append((start, prev))

    return ", ".join(
        str(a) if a == b else f"{a}-{b}" for a, b in ranges
    )


def build_reference_list(bundle):
    merged = merge_provenance(bundle)
    refs = []
    i = 1
    for doc in sorted(merged.keys()):
        pages = format_page_ranges(merged[doc]["pages"])
        refs.append(f"[{i}] {doc} — pages {pages}")
        i += 1
    return refs


# =====================================================
# RETRIEVAL SORTING
# =====================================================

def get_all_chunks_sorted(record):
    all_chunks = []
    for sub in record.get("subqueries", []):
        for ch in sub.get("retrieved_chunks", []):
            all_chunks.append({
                "chunk": ch,
                "score": ch.get("rrf_score", 0.0)
            })
    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    return all_chunks


# =====================================================
# CONTEXT BUILD
# =====================================================

def build_context(bundle):
    blocks = []
    for item in bundle:
        text = repair_chunk_text(item["chunk"].get("text", ""))
        if chunk_is_usable(text):
            blocks.append(text)
    return "\n\n".join(blocks)


# =====================================================
# LLM CALL
# =====================================================

def llm_answer(query: str, context: str):

    system = """
You are a retrieval-grounded QA system.
Answer ONLY from context.
"""

    user = f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0,
        max_tokens=600,
    )

    return resp.choices[0].message.content.strip() #type: ignore


# =====================================================
# SELF HEALING
# =====================================================

def self_healing_answer(query, record):
    ranked = get_all_chunks_sorted(record)

    print("Attempt 1: top chunks")
    bundle = ranked[:MAX_CHUNKS_INITIAL]
    context = build_context(bundle)

    if context:
        return llm_answer(query, context), bundle

    print("Recovery: expanding chunks")
    bundle = ranked[:MAX_CHUNKS_EXPANDED]
    context = build_context(bundle)

    if context:
        return llm_answer(query, context), bundle

    return "No usable evidence found.", []


# =====================================================
# MAIN PIPELINE
# =====================================================

def generate_answer_from_last_entry():

    records = load_jsonl(MEMORY_FILE)
    record = records[-1]

    query_id = record.get("query_id")
    query = record.get("original_query")

    print("Processing query:", query)

    answer, bundle = self_healing_answer(query, record)

    refs = build_reference_list(bundle)
    if refs:
        answer += "\n\nREFERENCES\n"
        for r in refs:
            answer += r + "\n"

    chunk_ids = [str(x["chunk"].get("chunk_id")) for x in bundle]

    result = {
        "query_id": query_id,
        "original_query": query,
        "used_chunk_ids": chunk_ids,
        "answer": answer
    }

    dedup_id = stable_hash(query_id, "|".join(chunk_ids))
    result["dedup_id"] = dedup_id

    existing = load_json_array(OUTPUT_FILE)

    updated = False
    for i, item in enumerate(existing):
        if item.get("dedup_id") == dedup_id:
            existing[i] = result
            updated = True
            print("♻ Updated existing answer")
            break

    if not updated:
        existing.append(result)
        print("✅ Stored new answer")

    save_json_array(OUTPUT_FILE, existing)
    # print("\n===== FULL RAW SOURCE RECORD =====")
    # for row in load_jsonl(SOURCES_FILE):
    #     print(json.dumps(row, indent=2))
    #     break
    # print("=================================\n")
    return result