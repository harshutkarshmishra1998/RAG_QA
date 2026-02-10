import api_keys
import json
import random
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-large"
K = 5
SAMPLES = 20


def embed(texts):
    client = OpenAI()
    r = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    v = np.array([e.embedding for e in r.data], dtype=np.float32)
    faiss.normalize_L2(v)
    return v


def load_texts(path: Path):
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("text"):
                texts.append(obj["text"])
    return texts


def recall_at_k():
    index = faiss.read_index("storage/content_units.faiss")
    texts = load_texts(Path("storage/content_units_cleaned_chunked.jsonl"))

    sampled_ids = random.sample(range(len(texts)), min(SAMPLES, len(texts)))
    sampled_texts = [texts[i] for i in sampled_ids]

    qvecs = embed(sampled_texts)

    hits = 0
    for true_id, qv in zip(sampled_ids, qvecs):
        _, I = index.search(qv.reshape(1, -1), K)
        if true_id in I[0]:
            hits += 1

    recall = hits / len(sampled_ids)
    print(f"Recall@{K}: {recall:.2f}")


if __name__ == "__main__":
    recall_at_k()