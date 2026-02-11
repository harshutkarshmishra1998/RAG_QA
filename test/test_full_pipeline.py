"""
End-to-end test:
PDF → Ingestion → Clean → Chunk → Embed → FAISS

Run:
    python test/test_full_pipeline.py path/to/file.pdf
"""

import sys
import json
from pathlib import Path
import faiss

from ingestion.pdf_ingestion import ingest_pdf
from clean_normalize.clean_normalize import clean_content_units_file
from chunk.chunk import chunk_content_units
from embedding.faiss_embedder import build_faiss_db


# ---------------- CONFIG ---------------- #

STORAGE = Path("storage")
UNITS_FILE = STORAGE / "content_units.jsonl"
CLEANED_FILE = STORAGE / "content_units_cleaned.jsonl"
CHUNKED_FILE = STORAGE / "content_units_cleaned_chunked.jsonl"
FAISS_FILE = STORAGE / "content_units.faiss"


def main():

    if len(sys.argv) != 2:
        print("Usage: python test/test_full_pipeline.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).resolve()

    if not pdf_path.exists():
        print("ERROR: File not found.")
        sys.exit(1)

    print("\n=== FULL PIPELINE TEST START ===\n")

    # --------------------------------------------------------
    # 1. INGEST
    # --------------------------------------------------------
    print("1️⃣ Running ingestion...")
    result = ingest_pdf(str(pdf_path))
    assert result.document.doc_id.startswith("doc_")
    print(f"   ✔ Document created: {result.document.doc_id}")

    # --------------------------------------------------------
    # 2. CLEAN
    # --------------------------------------------------------
    print("2️⃣ Running cleaning...")
    cleaned_path = clean_content_units_file(UNITS_FILE)
    assert cleaned_path.exists()
    print(f"   ✔ Cleaned file: {cleaned_path.name}")

    # --------------------------------------------------------
    # 3. CHUNK
    # --------------------------------------------------------
    print("3️⃣ Running chunking...")
    chunked_path = chunk_content_units(cleaned_path)
    assert chunked_path.exists()
    print(f"   ✔ Chunked file: {chunked_path.name}")

    # Count chunks
    with chunked_path.open("r", encoding="utf-8") as f:
        chunk_count = sum(1 for _ in f)

    assert chunk_count > 0, "No chunks created."
    print(f"   ✔ Chunks created: {chunk_count}")

    # --------------------------------------------------------
    # 4. EMBED + FAISS
    # --------------------------------------------------------
    print("4️⃣ Building FAISS index...")
    faiss_path = build_faiss_db(chunked_path) #type: ignore
    assert faiss_path.exists()
    print(f"   ✔ FAISS file: {faiss_path.name}")

    # Load FAISS and verify dimension
    index = faiss.read_index(str(faiss_path))
    assert index.d == 3072, "Embedding dimension mismatch"
    assert index.ntotal > 0, "FAISS index has no vectors"

    print(f"   ✔ FAISS vectors stored: {index.ntotal}")

    # --------------------------------------------------------
    # 5. Alignment Check
    # --------------------------------------------------------
    print("5️⃣ Verifying alignment...")

    with chunked_path.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    assert index.ntotal == len(chunks), \
        "Vector count does not match chunk count."

    print("   ✔ Vector-to-chunk alignment verified.")

    print("\n🎉 FULL PIPELINE TEST PASSED\n")


if __name__ == "__main__":
    main()