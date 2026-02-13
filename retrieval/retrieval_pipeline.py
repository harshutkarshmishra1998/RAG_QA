import api_keys
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import faiss
from openai import OpenAI

# ================= CONFIG ================= #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_PATH = PROJECT_ROOT / "storage"

MEMORY_FILE = STORAGE_PATH / "query_memory.json"
FAISS_INDEX_PATH = STORAGE_PATH / "content_units.faiss"
CONTENT_FILE = STORAGE_PATH / "content_units_cleaned_chunked.jsonl"

OUTPUT_FILE = STORAGE_PATH / "retrieval_results.jsonl"

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

TOP_K = 5
SIM_THRESHOLD = 0.70
RRF_K = 60

openai_client = OpenAI()


# ================= UTILITIES ================= #

def _embed(text: str) -> np.ndarray:
    r = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    vec = np.array(r.data[0].embedding, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


def _cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _load_latest_query() -> Dict:
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)

    if not memory["queries"]:
        raise ValueError("Query memory is empty.")

    return memory["queries"][-1]


def _load_chunks():
    chunks = []
    with open(CONTENT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


# ================= RETRIEVAL CORE ================= #

def retrieve_latest_query_chunks():

    latest = _load_latest_query()
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    chunks = _load_chunks()

    output = {
        "query_id": latest["id"],
        "original_query": latest["original_query"],
        "subqueries": []
    }

    for sub in latest["decomposed"]:

        all_queries = [
            sub["original_subquery"],
            sub["enhanced_query"],
            *sub.get("multi_queries", [])
        ]

        query_vectors = [(q, _embed(q)) for q in all_queries if q]

        per_query_results = {}

        for q_text, q_vec in query_vectors:
            q_vec = np.expand_dims(q_vec, axis=0)
            scores, indices = index.search(q_vec, TOP_K)

            retrieved = []

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(chunks):
                    continue

                # if score < SIM_THRESHOLD:
                #     continue

                retrieved.append({
                    "chunk_id": int(idx),
                    "score": float(score)
                })

            per_query_results[q_text] = retrieved

        # ---------- RRF Fusion ---------- #

        fusion_scores = {}

        for q_text, results in per_query_results.items():
            for rank, item in enumerate(results):
                chunk_id = item["chunk_id"]
                rrf_score = 1 / (RRF_K + rank)

                if chunk_id not in fusion_scores:
                    fusion_scores[chunk_id] = {
                        "rrf_score": 0.0,
                        "source_queries": set()
                    }

                fusion_scores[chunk_id]["rrf_score"] += rrf_score
                fusion_scores[chunk_id]["source_queries"].add(q_text)

        # ---------- Sort ---------- #

        sorted_chunks = sorted(
            fusion_scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True
        )

        final_chunks = []

        for chunk_id, data in sorted_chunks:
            chunk_id = int(chunk_id)
            chunk_data = chunks[chunk_id]

            final_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_data.get("text", ""),
                "metadata": chunk_data.get("metadata", {}),
                "rrf_score": round(data["rrf_score"], 6),
                "source_queries": list(data["source_queries"])
            })

        output["subqueries"].append({
            "subquery_id": sub["subquery_id"],
            "queries_used": all_queries,
            "retrieved_chunks": final_chunks
        })

    # ---------- Save as JSONL ---------- #

    STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    # # If file doesn't exist → create and write
    # if not OUTPUT_FILE.exists():
    #     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    #         f.write(json.dumps(output) + "\n")
    # else:
    #     # Check for duplicate query_id
    #     is_duplicate = False

    #     with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    #         for line in f:
    #             try:
    #                 existing = json.loads(line)
    #                 if existing.get("query_id") == output["query_id"]:
    #                     is_duplicate = True
    #                     break
    #             except Exception:
    #                 continue

    #     # Append only if not duplicate
    #     if not is_duplicate:
    #         with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    #             f.write(json.dumps(output, default=_json_safe) + "\n")

    # If file doesn't exist → create and write
    if not OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(output, default=_json_safe) + "\n")

    else:
        # ---------- read ONLY last entry ----------
        last_entry = None

        with open(OUTPUT_FILE, "rb") as f:
            try:
                f.seek(-2, 2)
                while f.read(1) != b"\n":
                    f.seek(-2, 1)
            except OSError:
                f.seek(0)

            last_line = f.readline().decode("utf-8").strip()

            if last_line:
                try:
                    last_entry = json.loads(last_line)
                except Exception:
                    last_entry = None

        # # ---------- sequential dedupe ----------
        # if last_entry and last_entry.get("query_id") == output["query_id"]:
        #     # overwrite last line
        #     with open(OUTPUT_FILE, "rb+") as f:
        #         f.seek(-len(last_line) - 1, 2)
        #         f.truncate()
        #         f.write(json.dumps(output, default=_json_safe).encode("utf-8") + b"\n")
        # else:
        #     with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        #         f.write(json.dumps(output, default=_json_safe) + "\n")

        if last_entry and last_entry.get("query_id") == output["query_id"]:
            # load all valid lines
            valid_lines = []
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        valid_lines.append(obj)
                    except json.JSONDecodeError:
                        continue

            # replace last entry
            if valid_lines:
                valid_lines[-1] = output
            else:
                valid_lines.append(output)

            # rewrite file cleanly
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for obj in valid_lines:
                    f.write(json.dumps(obj, default=_json_safe) + "\n")

        else:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(output, default=_json_safe) + "\n")


    return output
