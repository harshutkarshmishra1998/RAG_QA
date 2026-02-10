import api_keys
import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1-mini"
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


def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(obj["text"])
    return chunks


def answer(question: str):
    index = faiss.read_index("storage/content_units.faiss")
    chunks = load_chunks(Path("storage/content_units_cleaned_chunked.jsonl"))

    qv = embed(question)
    D, I = index.search(qv.reshape(1, -1), TOP_K)

    context = []
    for rank, idx in enumerate(I[0], 1):
        context.append(f"[{rank}] {chunks[idx]}")

    prompt = f"""
Answer the question using ONLY the context below.
Cite sources using [number] notation.

Question:
{question}

Context:
{chr(10).join(context)}
"""

    client = OpenAI()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return resp.choices[0].message.content


if __name__ == "__main__":
    print(
        answer(
            "What are the high-level process steps after a covered institution fails under Part 370?"
        )
    )