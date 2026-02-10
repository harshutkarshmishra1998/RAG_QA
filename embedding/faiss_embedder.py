import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072


def _load_texts(jsonl_path: Path) -> List[str]:
    texts = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            text = obj.get("text", "").strip()

            if not text:
                continue

            texts.append(text)

    if not texts:
        raise ValueError("No valid text chunks found to embed.")

    return texts


def _embed_texts(texts: List[str]) -> np.ndarray:
    client = OpenAI()

    embeddings: List[List[float]] = []

    BATCH_SIZE = 96  # safe, conservative

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        embeddings.extend(e.embedding for e in response.data)

    vectors = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    return vectors


def build_faiss_db(jsonl_path: str) -> Path:
    """
    Builds a FAISS vector database from a JSONL file of content units.

    Args:
        jsonl_path: Path to chunked JSONL file

    Returns:
        Path to the created FAISS index file
    """
    jsonl_path = Path(jsonl_path) #type: ignore

    if not jsonl_path.exists(): #type: ignore
        raise FileNotFoundError(jsonl_path)

    texts = _load_texts(jsonl_path) #type: ignore
    vectors = _embed_texts(texts)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors) #type: ignore

    output_path = jsonl_path.parent / "content_units.faiss" #type: ignore
    faiss.write_index(index, str(output_path))

    return output_path