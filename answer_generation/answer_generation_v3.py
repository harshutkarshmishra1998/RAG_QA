import api_keys
import json
import hashlib
import re
import ast
from pathlib import Path
from typing import List, Dict, Any

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
MAX_EVIDENCE_SNIPPETS = 3


# =====================================================
# SAFE FILE IO
# =====================================================

def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_json_array(path: Path):
    if not path.exists():
        return []
    try:
        txt = path.read_text(encoding="utf-8").strip()
        if not txt:
            return []
        return json.loads(txt)
    except json.JSONDecodeError:
        print("⚠ Corrupted JSON detected — resetting")
        return []


def save_json_array(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def stable_hash(*parts):
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


# =====================================================
# LOAD INDICES
# =====================================================

CONTENT_UNIT_INDEX = {str(r["unit_id"]): r for r in load_jsonl(CONTENT_UNIT_FILE)}
CHUNK_UNIT_MAP = {str(r["chunk_id"]): r.get("unit_ids", []) for r in load_jsonl(CHUNK_MAP_FILE)}
DOCUMENT_REGISTRY = {r.get("doc_id"): r.get("source_id") for r in load_jsonl(DOCUMENTS_FILE)}

def clean_filename(uri):
    return Path(uri).stem if uri else None

SOURCE_REGISTRY = {r.get("source_id"): clean_filename(r.get("source_uri")) for r in load_jsonl(SOURCES_FILE)}

def resolve_document_name(doc_id):
    return SOURCE_REGISTRY.get(DOCUMENT_REGISTRY.get(doc_id), doc_id)


# =====================================================
# TEXT NORMALIZATION (ROBUST)
# =====================================================

def flatten_tokens(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(flatten_tokens(item))
        return out
    return [str(x)]


# def repair_text(x):
#     if x is None:
#         return ""

#     if isinstance(x, str):
#         s = x.strip()
#         if s.startswith("[") and s.endswith("]"):
#             try:
#                 parsed = ast.literal_eval(s)
#                 tokens = flatten_tokens(parsed)
#             except:
#                 tokens = [s]
#         else:
#             tokens = [s]
#     else:
#         tokens = flatten_tokens(x)

#     text = " ".join(tokens)
#     text = re.sub(r"\s+([,.;:!?])", r"\1", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

def repair_text(x):
    """
    Converts OCR / token arrays into clean readable text.
    Removes token quotes, commas, empty tokens, and artifacts.
    """

    if x is None:
        return ""

    # -------- flatten structure --------
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                tokens = flatten_tokens(parsed)
            except:
                tokens = [s]
        else:
            tokens = [s]
    else:
        tokens = flatten_tokens(x)

    # -------- join tokens --------
    text = " ".join(tokens)

    # -------- remove quoted token formatting --------
    # 'FDIC' → FDIC
    text = re.sub(r"'([^']+)'", r"\1", text)

    # remove empty quoted tokens ''
    text = text.replace("''", " ")

    # -------- remove comma token separators --------
    # word , word → word word
    text = re.sub(r"\s*,\s*", " ", text)

    # -------- normalize dashes --------
    text = text.replace(" — ", " — ")
    text = re.sub(r"\s*-\s*", " - ", text)

    # -------- remove duplicate punctuation --------
    text = re.sub(r"([,.;:!?])\1+", r"\1", text)

    # -------- remove bracket artifacts --------
    text = text.replace("[", "").replace("]", "")

    # -------- collapse whitespace --------
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_chunk_for_display(text, max_len=220):
    text = repair_text(text)
    text = re.sub(r'\b\d+(\.\d+)*\b\s*', '', text)
    text = re.sub(r'[-_=]{3,}', ' ', text)
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."

    return text


def tokenize(t):
    return set(re.findall(r"\b\w+\b", repair_text(t).lower()))


def overlap(a, b):
    A = tokenize(a)
    B = tokenize(b)
    if not A:
        return 0
    return len(A & B) / len(A)


# =====================================================
# SCHEMA-ADAPTIVE CHUNK TEXT EXTRACTION
# =====================================================

def get_chunk_display_text(chunk: Dict[str, Any]):
    candidates = [
        chunk.get("text"),
        chunk.get("content"),
        chunk.get("raw_text"),
        chunk.get("page_text"),
        chunk.get("lines"),
        chunk.get("blocks"),
        chunk
    ]

    for c in candidates:
        if not c:
            continue

        cleaned = normalize_chunk_for_display(c)

        if "[" in cleaned and "'" in cleaned:
            continue

        if len(cleaned) > 20:
            return cleaned

    return normalize_chunk_for_display(chunk)


# =====================================================
# PROVENANCE
# =====================================================

def fuzzy_unit_match(chunk_text):
    best = None
    best_score = 0

    for uid, unit in CONTENT_UNIT_INDEX.items():
        score = overlap(chunk_text, unit.get("content"))
        if score > best_score:
            best_score = score
            best = uid

    return [best] if best_score > 0.3 else []


def get_chunk_provenance(chunk):
    unit_ids = chunk.get("unit_ids") or CHUNK_UNIT_MAP.get(str(chunk.get("chunk_id")), [])
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


# =====================================================
# REFERENCES + HUMAN READABLE EVIDENCE
# =====================================================

def page_ranges(pages):
    pages = sorted(p for p in pages if p is not None)
    if not pages:
        return "unknown"

    start = prev = pages[0]
    ranges = []

    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = prev = p

    ranges.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def extract_evidence_snippets(bundle, doc):
    snippets = []
    for item in bundle:
        chunk = item["chunk"]
        prov = get_chunk_provenance(chunk)
        if doc not in prov:
            continue
        cleaned = get_chunk_display_text(chunk)
        snippets.append(cleaned)
    return snippets[:MAX_EVIDENCE_SNIPPETS]


def build_reference_list(bundle):
    merged = {}

    for item in bundle:
        for doc, pages in get_chunk_provenance(item["chunk"]).items():
            merged.setdefault(doc, set()).update(pages)

    refs = []
    doc_to_ref = {}
    i = 1

    for doc in sorted(merged):
        label = f"[{i}]"
        ref = f"{label} {doc} — pages {page_ranges(merged[doc])}"

        snippets = extract_evidence_snippets(bundle, doc)
        if snippets:
            ref += "\n    Evidence:"
            for s in snippets:
                ref += f"\n      - {s}"

        refs.append(ref)
        doc_to_ref[doc] = label
        i += 1

    return refs, doc_to_ref


# =====================================================
# CLAIM SEGMENTATION
# =====================================================

def segment_claims(answer):
    lines = answer.split("\n")
    claims = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r"^(\d+\.|\-|\*)\s+", line):
            claims.append(line)
            continue

        parts = re.split(r'(?<=[.!?])\s+', line)
        claims.extend(p.strip() for p in parts if p.strip())

    return claims


# =====================================================
# CLAIM → EVIDENCE ATTRIBUTION
# =====================================================

def find_best_chunk_for_claim(claim, bundle):
    best_chunk = None
    best_score = 0

    for item in bundle:
        chunk_text = repair_text(item["chunk"].get("text"))
        score = overlap(claim, chunk_text)

        if score > best_score:
            best_score = score
            best_chunk = item["chunk"]

    return best_chunk


def attribute_claims_to_refs(claims, bundle, doc_to_ref):
    attributed = []

    for claim in claims:
        chunk = find_best_chunk_for_claim(claim, bundle)

        if not chunk:
            attributed.append((claim, ["[unverified]"]))
            continue

        prov = get_chunk_provenance(chunk)
        refs = [doc_to_ref[d] for d in prov if d in doc_to_ref]
        attributed.append((claim, sorted(set(refs))))

    return attributed


def inject_citations(attributed_claims):
    return "\n".join(f"{claim} {''.join(refs)}" for claim, refs in attributed_claims)


# =====================================================
# LLM ANSWER
# =====================================================

def llm_generate_answer(query, context):

    system = """
Answer the question using the provided context.
Explain clearly and completely.
Do not include citations.
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
# CLAIM-LEVEL METRICS
# =====================================================

def compute_evidence_metrics(answer):
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    claim_lines = []
    for l in lines:
        if l.startswith("REFERENCES"):
            break
        claim_lines.append(l)

    total = len(claim_lines)
    supported = sum(1 for l in claim_lines if "[" in l and "unverified" not in l)

    return {
        "claim_count": total,
        "supported_claims": supported,
        "citation_coverage": round(supported/total, 3) if total else 0
    }


# =====================================================
# MAIN PIPELINE
# =====================================================

def generate_answer_from_last_entry():

    record = load_jsonl(MEMORY_FILE)[-1]
    query = record["original_query"]
    query_id = record["query_id"]

    print("Processing query:", query)

    chunks = []
    for s in record.get("subqueries", []):
        for c in s.get("retrieved_chunks", []):
            chunks.append({"chunk": c, "score": c.get("rrf_score", 0)})

    chunks.sort(key=lambda x: x["score"], reverse=True)
    bundle = chunks[:MAX_CHUNKS_INITIAL]

    refs, doc_to_ref = build_reference_list(bundle)

    context = "\n\n".join(
        repair_text(item["chunk"].get("text")) +
        "\nSOURCE " +
        " ".join(doc_to_ref[d] for d in get_chunk_provenance(item["chunk"]) if d in doc_to_ref)
        for item in bundle
    )

    raw_answer = llm_generate_answer(query, context)

    claims = segment_claims(raw_answer)
    attributed = attribute_claims_to_refs(claims, bundle, doc_to_ref)
    grounded_answer = inject_citations(attributed)

    metrics = compute_evidence_metrics(grounded_answer)

    if refs:
        grounded_answer += "\n\nREFERENCES\n" + "\n".join(refs)

    result = {
        "query_id": query_id,
        "original_query": query,
        "used_chunk_ids": [c["chunk"]["chunk_id"] for c in bundle],
        "answer": grounded_answer,
        "evidence_metrics": metrics
    }

    result["dedup_id"] = stable_hash(query_id, str(result["used_chunk_ids"]))

    # data = load_json_array(OUTPUT_FILE)
    # data.append(result)
    # save_json_array(OUTPUT_FILE, data)

    # print("✅ Stored deterministic grounded answer")
    data = load_json_array(OUTPUT_FILE)

    existing_ids = {item.get("dedup_id") for item in data}

    if result["dedup_id"] in existing_ids:
        print("⚠ Duplicate result — not stored")
    else:
        data.append(result)
        save_json_array(OUTPUT_FILE, data)
        print("✅ Stored deterministic grounded answer")

    return result