import api_keys
import json
from pathlib import Path
import faiss
import numpy as np
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K = 5


def embed(text: str) -> np.ndarray:
    client = OpenAI()
    r = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    v = np.array(r.data[0].embedding, dtype=np.float32)
    faiss.normalize_L2(v.reshape(1, -1))
    return v


def load_chunks(jsonl_path: Path):
    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(obj.get("text", ""))
    return chunks


def run_query(query: str, faiss_path: str, jsonl_path: str):
    index = faiss.read_index(faiss_path)
    chunks = load_chunks(Path(jsonl_path))

    qv = embed(query)
    D, I = index.search(qv.reshape(1, -1), TOP_K)

    print(f"\nQUERY: {query}")
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        print(f"\nRank {rank} | score={score:.4f}")
        print(chunks[idx][:300])


if __name__ == "__main__":
    run_query(
        query="What are the high-level process steps performed after a covered institution fails under Part 370?",
        faiss_path="storage/content_units.faiss",
        jsonl_path="storage/content_units_cleaned_chunked.jsonl",
    )

# What are the high-level process steps performed after a covered institution fails under Part 370?
# What activities must be completed within 24 hours after a bank failure according to Part 370?
# What is described in the High-Level Process at Failure section?