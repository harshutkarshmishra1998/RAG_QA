import api_keys
import json
import hashlib
import ast
import re
from pathlib import Path
from typing import Dict, List

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
# FILE IO
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


def load_json_array(path: Path):
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except:
        return []


def save_json_array(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def stable_hash(*parts):
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


# =====================================================
# LOAD INDICES
# =====================================================

CONTENT_UNIT_INDEX = {str(r["unit_id"]): r for r in load_jsonl(CONTENT_UNIT_FILE)}

CHUNK_UNIT_MAP = {
    str(r["chunk_id"]): r.get("unit_ids", [])
    for r in load_jsonl(CHUNK_MAP_FILE)
}

DOCUMENT_REGISTRY = {
    r.get("doc_id"): r.get("source_id")
    for r in load_jsonl(DOCUMENTS_FILE)
}

def clean_filename(uri):
    if not uri:
        return None
    return Path(uri).stem

SOURCE_REGISTRY = {
    r.get("source_id"): clean_filename(r.get("source_uri"))
    for r in load_jsonl(SOURCES_FILE)
}

def resolve_document_name(doc_id):
    source_id = DOCUMENT_REGISTRY.get(doc_id)
    return SOURCE_REGISTRY.get(source_id, doc_id)


# =====================================================
# TEXT NORMALIZATION
# =====================================================

def repair_text(text):
    if text is None:
        return ""

    if isinstance(text, list):
        text = " ".join(map(str, text))

    if isinstance(text, str):
        stripped = text.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    text = " ".join(map(str, parsed))
            except:
                pass

    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


def tokenize(t):
    if not isinstance(t, str):
        t = repair_text(t)
    return set(re.findall(r"\b\w+\b", t.lower()))


def overlap(a, b):
    A = tokenize(a)
    B = tokenize(b)
    if not A:
        return 0
    return len(A & B) / len(A)


def chunk_is_usable(text):
    text = repair_text(text)
    if len(text) < 40:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio > 0.5


# =====================================================
# PROVENANCE
# =====================================================

def get_chunk_units(chunk):
    unit_ids = chunk.get("unit_ids")
    if unit_ids:
        return unit_ids
    return CHUNK_UNIT_MAP.get(str(chunk.get("chunk_id")), [])


def fuzzy_unit_match(chunk_text):
    chunk_text = repair_text(chunk_text)
    best = None
    best_score = 0

    for uid, unit in CONTENT_UNIT_INDEX.items():
        score = overlap(chunk_text, unit.get("content"))
        if score > best_score:
            best_score = score
            best = uid

    return [best] if best_score > 0.3 else []


def get_chunk_provenance(chunk):
    unit_ids = get_chunk_units(chunk)

    if not unit_ids:
        unit_ids = fuzzy_unit_match(chunk.get("text"))

    prov = {}

    for uid in unit_ids:
        unit = CONTENT_UNIT_INDEX.get(str(uid))
        if not unit:
            continue

        doc = resolve_document_name(unit.get("doc_id"))
        page = unit.get("page_start")

        prov.setdefault(doc, set())
        if page is not None:
            prov[doc].add(page)

    return prov


def merge_provenance(bundle):
    merged = {}
    for item in bundle:
        prov = get_chunk_provenance(item["chunk"])
        for doc, pages in prov.items():
            merged.setdefault(doc, set()).update(pages)
    return merged


def page_ranges(pages):
    if not pages:
        return "unknown"
    pages = sorted(pages)
    ranges = []
    s = pages[0]
    p = pages[0]

    for x in pages[1:]:
        if x == p + 1:
            p = x
        else:
            ranges.append((s, p))
            s = x
            p = x
    ranges.append((s, p))

    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def build_reference_list(bundle):
    merged = merge_provenance(bundle)
    refs = []
    doc_to_ref = {}
    i = 1

    for doc in sorted(merged):
        label = f"[{i}]"
        refs.append(f"{label} {doc} — pages {page_ranges(merged[doc])}")
        doc_to_ref[doc] = label
        i += 1

    return refs, doc_to_ref


# =====================================================
# RETRIEVAL
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

def build_context(bundle, doc_to_ref):
    blocks = []

    for item in bundle:
        chunk = item["chunk"]
        text = repair_text(chunk.get("text"))

        if not chunk_is_usable(text):
            continue

        prov = get_chunk_provenance(chunk)
        labels = [doc_to_ref[d] for d in prov if d in doc_to_ref]

        if labels:
            blocks.append(text + "\nSOURCE " + " ".join(labels))

    return "\n\n".join(blocks)


# =====================================================
# LLM
# =====================================================

def llm_answer(query, context):

    system = """
You are a strictly evidence-grounded QA system.

Rules:
1. Every factual statement MUST have a citation.
2. Do NOT write any uncited claim.
3. Do NOT add explanatory filler without citation.
4. If a sentence has no evidence — remove it.
5. Prefer concise factual sentences.
6. Multiple claims in one sentence → multiple citations.
Use ONLY citation numbers provided in context.
Never invent citations.
"""

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip() #type: ignore


# =====================================================
# EVIDENCE METRICS
# =====================================================

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def compute_evidence_metrics(answer, context):
    sentences = split_sentences(answer)
    context_lower = context.lower()

    supported = 0
    confidence_scores = []

    for s in sentences:
        if "[" in s:
            supported += 1
            confidence_scores.append(overlap(s, context_lower))

    coverage = supported / len(sentences) if sentences else 0
    confidence = sum(confidence_scores)/len(confidence_scores) if confidence_scores else 0

    return {
        "sentence_count": len(sentences),
        "supported_sentences": supported,
        "citation_coverage": round(coverage, 3),
        "avg_evidence_confidence": round(confidence, 3)
    }

def enforce_evidence_grounding(answer, metrics):

    if metrics["supported_sentences"] == 0:
        return "Insufficient evidence in retrieved context to answer this question."

    return answer


# =====================================================
# SELF HEALING RETRIEVAL
# =====================================================

def self_healing_answer(query, record):
    ranked = get_all_chunks_sorted(record)
    return ranked[:MAX_CHUNKS_INITIAL]


# =====================================================
# SUBQUERY EXTRACTION (FIX)
# =====================================================

def extract_subqueries(record):

    subs = record.get("subqueries", [])

    if not subs:
        return [{
            "question": record["original_query"],
            "chunks": []
        }]

    questions = []
    has_real = False

    for sub in subs:
        q = None
        for k, v in sub.items():
            if isinstance(v, str) and len(v.split()) > 5:
                q = v.strip()
                has_real = True
                break
        questions.append(q)

    if not has_real:
        original = record["original_query"]
        split_q = [
            x.strip()
            for x in re.split(r"\?\s+|\?$", original)
            if x.strip()
        ]
        while len(split_q) < len(subs):
            split_q.append(original)
        questions = split_q[:len(subs)]

    result = []
    for i, sub in enumerate(subs):
        result.append({
            "question": questions[i],
            "chunks": sub.get("retrieved_chunks", [])
        })

    return result


# =====================================================
# SUBQUERY SELF HEALING (FIX)
# =====================================================

def self_healing_subquery_bundle(question, chunks, record):

    ranked = sorted(
        [{"chunk": c, "score": c.get("rrf_score", 0.0)} for c in chunks],
        key=lambda x: x["score"],
        reverse=True
    )

    bundle = ranked[:MAX_CHUNKS_INITIAL]

    if not bundle:
        bundle = ranked[:MAX_CHUNKS_EXPANDED]

    if not bundle:
        bundle = self_healing_answer(question, record)

    return bundle


# =====================================================
# MAIN PIPELINE (FIXED)
# =====================================================

# def generate_answer_from_last_entry():

#     records = load_jsonl(MEMORY_FILE)
#     record = records[-1]

#     query_id = record["query_id"]
#     original_query = record["original_query"]

#     print("Processing query:", original_query)

#     subqueries = extract_subqueries(record)

#     final_blocks = []
#     full_bundle = []

#     for i, sub in enumerate(subqueries, 1):

#         print(f"Processing subquery {i}")

#         bundle = self_healing_subquery_bundle(
#             sub["question"],
#             sub["chunks"],
#             record
#         )

#         refs, doc_to_ref = build_reference_list(bundle)
#         context = build_context(bundle, doc_to_ref)

#         answer = llm_answer(sub["question"], context)
#         metrics = compute_evidence_metrics(answer, context)

#         block = (
#             f"QUESTION {i}:\n{sub['question']}\n\n"
#             f"ANSWER:\n{answer}\n\n"
#             f"EVIDENCE METRICS:\n{json.dumps(metrics, indent=2)}"
#         )

#         if refs:
#             block += "\n\nREFERENCES\n" + "\n".join(refs)

#         final_blocks.append(block)
#         full_bundle.extend(bundle)

#     final_answer = "\n\n" + ("\n\n" + "="*60 + "\n\n").join(final_blocks)

#     result = {
#         "query_id": query_id,
#         "original_query": original_query,
#         "used_chunk_ids": [c["chunk"]["chunk_id"] for c in full_bundle],
#         "answer": final_answer
#     }

#     result["dedup_id"] = stable_hash(query_id, str(result["used_chunk_ids"]))

#     data = load_json_array(OUTPUT_FILE)

#     replaced = False
#     for i, item in enumerate(data):
#         if item.get("dedup_id") == result["dedup_id"]:
#             data[i] = result
#             replaced = True
#             print("♻ Updated existing answer")
#             break

#     if not replaced:
#         data.append(result)
#         print("✅ Stored new answer")

#     save_json_array(OUTPUT_FILE, data)
#     return result

def generate_answer_from_last_entry():

    records = load_jsonl(MEMORY_FILE)
    record = records[-1]

    query_id = record["query_id"]
    original_query = record["original_query"]

    print("Processing query:", original_query)

    subqueries = extract_subqueries(record)

    final_blocks = []
    full_bundle = []
    subquery_results = []

    for i, sub in enumerate(subqueries, 1):

        print(f"Processing subquery {i}")

        bundle = self_healing_subquery_bundle(
            sub["question"],
            sub["chunks"],
            record
        )

        refs, doc_to_ref = build_reference_list(bundle)
        context = build_context(bundle, doc_to_ref)

        answer = llm_answer(sub["question"], context)
        metrics = compute_evidence_metrics(answer, context)

        # answer = llm_answer(sub["question"], context)
        # metrics = compute_evidence_metrics(answer, context)

        # answer = enforce_evidence_grounding(answer, metrics)

        # clean display block (NO METRICS HERE)
        block = (
            f"QUESTION {i}:\n{sub['question']}\n\n"
            f"ANSWER:\n{answer}"
        )

        if refs:
            block += "\n\nREFERENCES\n" + "\n".join(refs)

        final_blocks.append(block)
        full_bundle.extend(bundle)

        # structured storage (THIS is where metrics go)
        subquery_results.append({
            "index": i,
            "question": sub["question"],
            "answer": answer,
            "used_chunk_ids": [b["chunk"]["chunk_id"] for b in bundle],
            "evidence_metrics": metrics,
            "references": refs
        })

    final_answer = "\n\n" + ("\n\n" + "="*60 + "\n\n").join(final_blocks)

    # optional aggregate metrics (recommended)
    all_metrics = [s["evidence_metrics"] for s in subquery_results]
    if all_metrics:
        aggregate_metrics = {
            "avg_citation_coverage": round(
                sum(m["citation_coverage"] for m in all_metrics)/len(all_metrics), 3
            ),
            "avg_evidence_confidence": round(
                sum(m["avg_evidence_confidence"] for m in all_metrics)/len(all_metrics), 3
            ),
            "total_supported_sentences": sum(m["supported_sentences"] for m in all_metrics),
            "total_sentences": sum(m["sentence_count"] for m in all_metrics)
        }
    else:
        aggregate_metrics = {}

    result = {
        "query_id": query_id,
        "original_query": original_query,
        "used_chunk_ids": [c["chunk"]["chunk_id"] for c in full_bundle],
        "answer": final_answer,

        # NEW STRUCTURED FIELDS
        "subquery_results": subquery_results,
        "aggregate_evidence_metrics": aggregate_metrics
    }

    result["dedup_id"] = stable_hash(query_id, str(result["used_chunk_ids"]), str("v2"))

    # data = load_json_array(OUTPUT_FILE)

    # replaced = False
    # for i, item in enumerate(data):
    #     if item.get("dedup_id") == result["dedup_id"]:
    #         data[i] = result
    #         replaced = True
    #         print("♻ Updated existing answer")
    #         break

    # if not replaced:
    #     print("✅ Stored new answer")
    #     data.append(result)

    # save_json_array(OUTPUT_FILE, data)

    data = load_json_array(OUTPUT_FILE)

    # last-entry-only dedupe
    if data and data[-1].get("dedup_id") == result["dedup_id"]:
        data[-1] = result
        print("♻ Updated last answer")
    else:
        print("✅ Stored new answer")
        data.append(result)

    save_json_array(OUTPUT_FILE, data)

    return result
