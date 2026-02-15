import api_keys
import os
import json
import re
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from openai import OpenAI
from groq import Groq


# CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_PATH = PROJECT_ROOT / "storage"
MEMORY_FILE = STORAGE_PATH / "query_memory.json"

FAISS_INDEX_PATH = STORAGE_PATH / "content_unit.faiss"
CONTENT_FILE = STORAGE_PATH / "content_units_cleaned_chunked.jsonl"

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

GROQ_MODEL = "llama-3.1-8b-instant"

DRIFT_THRESHOLD = 0.84
MAX_MULTI_QUERIES = 3

openai_client = OpenAI()
groq_client = Groq()


# UTILITIES

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    return text


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# EMBEDDING

def _embed(text: str) -> np.ndarray:
    r = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )

    vec = np.array(r.data[0].embedding, dtype=np.float32)

    if vec.shape[0] != EMBEDDING_DIM:
        raise ValueError("Embedding dimension mismatch.")

    return vec


def _validate_semantic_drift(original: str, candidate: str, threshold: float = DRIFT_THRESHOLD) -> bool:
    vec1 = _embed(original)
    vec2 = _embed(candidate)
    similarity = _cosine_similarity(vec1, vec2)
    return similarity >= threshold


# LLM JSON WRAPPER

def _groq_json_call(prompt: str) -> Dict:

    system_message = """
You MUST output valid JSON only.
Do NOT include explanations.
Do NOT include reasoning.
Do NOT include <think> blocks.
If you include anything else, it is an error.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()  # type: ignore
    content = _strip_think(content)

    parsed = _safe_json_loads(content)

    if parsed is None:
        raise ValueError("Invalid JSON returned from Groq.")

    return parsed


# STRUCTURAL HELPERS

def _is_question(text: str) -> bool:
    return text.strip().endswith("?")


def _preserve_question_shape(original: str, candidate: str) -> bool:
    return _is_question(original) == _is_question(candidate)


# DECOMPOSITION (V3)

def _deterministic_split(query: str) -> List[str]:
    parts = re.split(r"\?\s*", query)
    parts = [p.strip() for p in parts if p.strip()]

    results = []
    for p in parts:
        if not p.endswith("?"):
            p = p + "?"
        results.append(p)

    return results

def _split_interrogative_conjunction(query: str) -> List[str]:
    """
    Split patterns like:
    What ... and what ...
    How ... and how ...
    """

    pattern = re.compile(
        r"\b(and|or)\s+(what|how|when|why|where|who)\b",
        re.IGNORECASE
    )

    match = pattern.search(query)

    if not match:
        return [query]

    split_index = match.start()

    first = query[:split_index].strip()
    second = query[match.end() - len(match.group(2)):].strip()

    if not first.endswith("?"):
        first += "?"
    if not second.endswith("?"):
        second += "?"

    return [first, second]


# def _decompose_query_v3(query: str) -> List[str]:

#     deterministic = _deterministic_split(query)

#     if len(deterministic) > 1:
#         return deterministic

#     lowered = query.lower()
#     conjunction_triggers = [
#         " and ", " or ", " but ", " yet ",
#         " also ", " additionally ", " moreover ",
#         " furthermore ", " as well as ",
#         " then ", " after that ", " next ",
#         " followed by ",
#         " compare ", " contrast ",
#         " difference between ", " differences between ",
#         " explain and ", " describe and ",
#         " analyze and ", " evaluate and ",
#         " discuss and ", " and moreover "
#     ]

#     if not any(t in lowered for t in conjunction_triggers):
#         return [query]

#     prompt = f"""
# Split into fully independent questions.
# Each must:
# - Be complete
# - End with '?' or should have a conjuction which will help to split the query
# - Preserve original meaning
# - Contain only one request

# Return JSON:
# {{"queries": ["q1", "q2"]}}

# Query:
# {query}
# """

#     try:
#         result = _groq_json_call(prompt)
#         queries = result.get("queries", [])

#         if not isinstance(queries, list) or len(queries) < 2:
#             return [query]

#         cleaned = []
#         for q in queries:
#             q = q.strip()
#             if not q.endswith("?"):
#                 q += "?"
#             cleaned.append(q)

#         return cleaned

#     except Exception:
#         return [query]


def _decompose_query_v3(query: str) -> List[str]:

    # Step 1: split by '?'
    base = _deterministic_split(query)

    final = []

    for part in base:
        split_parts = _split_interrogative_conjunction(part)
        final.extend(split_parts)

    return final


# ENHANCEMENT (V3)

def _enhance_query_v3(query: str) -> str:

    prompt = f"""
Rewrite the question for retrieval clarity.
Do NOT:
- Change meaning
- Convert to statement
- Add concepts
- Remove '?'

Return JSON:
{{"enhanced": "rewritten question"}}

Query:
{query}
"""

    try:
        result = _groq_json_call(prompt)
        enhanced = result.get("enhanced", "").strip()

        if not enhanced:
            return query

        if not _preserve_question_shape(query, enhanced):
            return query

        if not _validate_semantic_drift(query, enhanced):
            return query

        return enhanced

    except Exception:
        return query


# RETRIEVAL

def _light_retrieve(query: str, top_k: int = 3) -> List[str]:

    if not FAISS_INDEX_PATH.exists() or not CONTENT_FILE.exists():
        return []

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    chunks = []
    with open(CONTENT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    query_vec = _embed(query)
    query_vec = np.expand_dims(query_vec, axis=0)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx]["text"])

    return results


# MULTI QUERY (V3)

def _generate_multi_queries_v3(enhanced_query: str, context_chunks: List[str]) -> List[str]:

    context_preview = "\n\n".join(context_chunks[:3])

    prompt = f"""
Generate up to {MAX_MULTI_QUERIES} alternative retrieval questions.
Rules:
- Preserve meaning
- End with '?'
- Do NOT introduce new concepts

Return JSON:
{{"queries": ["q1", "q2", "q3"]}}

Query:
{enhanced_query}

Context:
{context_preview}
"""

    try:
        result = _groq_json_call(prompt)
        raw = result.get("queries", [])

        if not isinstance(raw, list):
            return []

        valid = []

        for q in raw:
            q = q.strip()

            if not q.endswith("?"):
                continue

            if not _validate_semantic_drift(enhanced_query, q):
                continue

            valid.append(q)

        unique = list(dict.fromkeys(valid))
        return unique[:MAX_MULTI_QUERIES]

    except Exception:
        return []


# MEMORY #

def _load_memory() -> Dict:
    if not MEMORY_FILE.exists():
        return {"queries": []}

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_memory(data: Dict):
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    temp = MEMORY_FILE.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    os.replace(temp, MEMORY_FILE)


# MAIN PIPELINE #

def process_user_query(user_query: str) -> Dict:

    normalized = _normalize_query(user_query)
    decomposed = _decompose_query_v3(normalized)

    processed_subqueries = []

    for sub in decomposed:

        clean_sub = _normalize_query(sub)
        enhanced = _enhance_query_v3(clean_sub)

        context = _light_retrieve(enhanced)
        multi_queries = _generate_multi_queries_v3(enhanced, context)

        processed_subqueries.append({
            "subquery_id": _hash_text(clean_sub),
            "original_subquery": clean_sub,
            "enhanced_query": enhanced,
            "multi_queries": multi_queries
        })

    query_id = _hash_text(normalized)
    memory = _load_memory()

    new_entry = {
        "id": query_id,
        "original_query": normalized,
        "decomposed": processed_subqueries
    }

    # existing_index = next(
    #     (i for i, q in enumerate(memory["queries"]) if q["id"] == query_id),
    #     None
    # )

    # if existing_index is not None:
    #     memory["queries"][existing_index] = new_entry
    # else:
    #     memory["queries"].append(new_entry)

    # sequential dedupe (compare only with last entry)
    if memory["queries"]:
        if memory["queries"][-1]["id"] == query_id:
            memory["queries"][-1] = new_entry
        else:
            memory["queries"].append(new_entry)
    else:
        memory["queries"].append(new_entry)

    _save_memory(memory)

    return new_entry