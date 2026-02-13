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

def load_source_registry(path: Path):
    registry = {}

    for row in load_jsonl(path):
        source_id = row.get("source_id") or row.get("id")
        uri = row.get("source_uri")
        filename = Path(uri).stem if uri else None
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

    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


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
# SUBQUERY EXTRACTION
# =====================================================

# def extract_subqueries(record):
#     subs = []

#     if record.get("subqueries"):
#         for i, sub in enumerate(record["subqueries"], 1):
#             q = (
#                 sub.get("subquery")
#                 or sub.get("query")
#                 or sub.get("question")
#                 or f"Subquestion {i}"
#             )
#             subs.append({
#                 "question": q,
#                 "chunks": sub.get("retrieved_chunks", [])
#             })
#     else:
#         subs.append({
#             "question": record["original_query"],
#             "chunks": []
#         })

#     return subs

def extract_subqueries(record):

    subs = record.get("subqueries", [])

    if not subs:
        return [{
            "question": record["original_query"],
            "chunks": []
        }]

    # try to detect actual stored subquery text
    has_real_questions = False
    questions = []

    for sub in subs:
        for k, v in sub.items():
            if isinstance(v, str) and len(v.split()) > 5:
                questions.append(v.strip())
                has_real_questions = True
                break
        else:
            questions.append(None)

    # if no real subquery text stored → split original query
    if not has_real_questions:

        original = record["original_query"]

        split_questions = [
            q.strip()
            for q in re.split(r"\?\s+|\?$", original)
            if q.strip()
        ]

        # ensure count alignment
        while len(split_questions) < len(subs):
            split_questions.append(original)

        questions = split_questions[:len(subs)]

    result = []

    for i, sub in enumerate(subs):
        result.append({
            "question": questions[i],
            "chunks": sub.get("retrieved_chunks", [])
        })

    return result


# =====================================================
# CONTEXT BUILD (PER SUBQUERY)
# =====================================================

# def build_context(chunks):
#     blocks = []
#     for ch in chunks:
#         text = repair_chunk_text(ch.get("text", ""))
#         if chunk_is_usable(text):
#             blocks.append(text)
#     return "\n\n".join(blocks)

def build_context(chunks):

    usable_blocks = []
    raw_blocks = []

    for ch in chunks:
        text = repair_chunk_text(ch.get("text", ""))

        if not text:
            continue

        raw_blocks.append(text)

        if chunk_is_usable(text):
            usable_blocks.append(text)

    # prefer clean usable chunks
    if usable_blocks:
        return "\n\n".join(usable_blocks)

    # fallback if filtering removed everything
    if raw_blocks:
        print("⚠ using raw chunks (filter removed all)")
        return "\n\n".join(raw_blocks)

    return ""



# =====================================================
# STRUCTURED LLM ANSWER (PER QUESTION)
# =====================================================

# def llm_answer_single(question, context):

#     system = """
# You are a retrieval-grounded QA system.

# Answer ONLY the specific question provided.
# Use ONLY the context.

# Rules:
# 1. Answer completely and precisely
# 2. If context insufficient say: Insufficient evidence in provided context
# 3. Do not include unrelated information
# 4. Do not merge with other questions
# 5. Be clear and structured
# """

#     user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"

#     resp = client.chat.completions.create(
#         model=GROQ_MODEL,
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "user", "content": user}
#         ],
#         temperature=0,
#         max_tokens=600,
#     )

#     return resp.choices[0].message.content.strip()

def llm_answer_single(question, context):

    system = """
You are a retrieval-grounded QA system.

Use the provided context to answer the question.

Rules:
1. Base your answer strictly on information contained in the context
2. You MAY synthesize, summarize, or reorganize information from multiple context passages
3. The answer does NOT need to appear as an exact sentence match
4. If context provides partial information, answer with what is supported
5. Only say "Insufficient evidence" if the context truly contains no relevant information
6. Be precise and factual
"""

    user = f"""QUESTION:
{question}

CONTEXT:
{context}
"""

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
# SELF HEALING PER SUBQUERY
# =====================================================

# def answer_subquery_with_self_healing(question, chunks):

#     ranked = sorted(
#         [{"chunk": c, "score": c.get("rrf_score", 0.0)} for c in chunks],
#         key=lambda x: x["score"],
#         reverse=True
#     )

#     bundle = ranked[:MAX_CHUNKS_INITIAL]
#     context = build_context([b["chunk"] for b in bundle])

#     if not context:
#         bundle = ranked[:MAX_CHUNKS_EXPANDED]
#         context = build_context([b["chunk"] for b in bundle])

#     if not context:
#         return "No usable evidence found.", []

#     answer = llm_answer_single(question, context)
#     return answer, bundle

def answer_subquery_with_self_healing(question, chunks, record):

    ranked = sorted(
        [{"chunk": c, "score": c.get("rrf_score", 0.0)} for c in chunks],
        key=lambda x: x["score"],
        reverse=True
    )

    # attempt 1
    bundle = ranked[:MAX_CHUNKS_INITIAL]
    context = build_context([b["chunk"] for b in bundle])

    # attempt 2
    if not context:
        bundle = ranked[:MAX_CHUNKS_EXPANDED]
        context = build_context([b["chunk"] for b in bundle])

    # final rescue → global retrieval
    if not context:
        print("⚠ fallback to global top chunks")
        global_ranked = get_all_chunks_sorted(record)
        bundle = global_ranked[:MAX_CHUNKS_EXPANDED]
        context = build_context([b["chunk"] for b in bundle])

    if not context:
        return "No usable evidence found.", []

    answer = llm_answer_single(question, context)
    return answer, bundle

# =====================================================
# MAIN PIPELINE
# =====================================================

def generate_answer_from_last_entry():

    records = load_jsonl(MEMORY_FILE)
    record = records[-1]

    query_id = record.get("query_id")
    query = record.get("original_query")

    subqueries = extract_subqueries(record)

    structured_blocks = []
    full_bundle = []

    for i, sub in enumerate(subqueries, 1):

        print(f"Processing subquery {i}")

        answer, bundle = answer_subquery_with_self_healing(
            sub["question"],
            sub["chunks"],
            record
        )

        structured_blocks.append(
            f"QUESTION {i}:\n{sub['question']}\n\nANSWER:\n{answer}"
        )

        full_bundle.extend(bundle)

    final_answer = "\n\n" + ("\n\n" + "="*60 + "\n\n").join(structured_blocks)

    refs = build_reference_list(full_bundle)
    if refs:
        final_answer += "\n\nREFERENCES\n"
        for r in refs:
            final_answer += r + "\n"

    chunk_ids = [str(x["chunk"].get("chunk_id")) for x in full_bundle]

    result = {
        "query_id": query_id,
        "original_query": query,
        "used_chunk_ids": chunk_ids,
        "answer": final_answer
    }

    dedup_id = stable_hash(query_id, "|".join(chunk_ids), str("v1"))
    result["dedup_id"] = dedup_id

    # existing = load_json_array(OUTPUT_FILE)

    # updated = False
    # for i, item in enumerate(existing):
    #     if item.get("dedup_id") == dedup_id:
    #         existing[i] = result
    #         updated = True
    #         break

    # if not updated:
    #     existing.append(result)

    # save_json_array(OUTPUT_FILE, existing)

    data = load_json_array(OUTPUT_FILE)

    # last-entry-only dedupe
    if data and data[-1].get("dedup_id") == dedup_id:
        data[-1] = result
    else:
        data.append(result)

    save_json_array(OUTPUT_FILE, data)

    return result