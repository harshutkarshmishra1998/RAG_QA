import api_keys
import json
import random
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K = 5
SAMPLES = 10


def load_texts(jsonl_path: Path):
    texts = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if text:
                texts.append(text)
    return texts


def embed(texts):
    client = OpenAI()
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = np.array([e.embedding for e in resp.data], dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def verify(
    faiss_path: str,
    jsonl_path: str,
):
    index = faiss.read_index(faiss_path)
    texts = load_texts(Path(jsonl_path))

    samples = random.sample(texts, min(SAMPLES, len(texts)))
    query_vecs = embed(samples)

    scores = []

    for i, q in enumerate(query_vecs):
        D, I = index.search(q.reshape(1, -1), TOP_K)
        top_score = float(D[0][0])
        scores.append(top_score)

        print(f"\nSample {i+1}")
        print(f"Top cosine similarity: {top_score:.4f}")
        print(f"Retrieved indices   : {I[0]}")

    print("\n==== SUMMARY ====")
    print(f"Avg top-1 similarity: {sum(scores)/len(scores):.4f}")
    print(f"Min top-1 similarity: {min(scores):.4f}")
    print(f"Max top-1 similarity: {max(scores):.4f}")

if __name__ == "__main__":
    verify(
        faiss_path="storage/content_units.faiss",
        jsonl_path="storage/content_units_cleaned_chunked.jsonl",
    )