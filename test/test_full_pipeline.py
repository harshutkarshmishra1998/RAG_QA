"""
Incremental End-to-End Test:
PDF → Ingestion → Incremental Clean → Chunk → Embed → FAISS Append
"""

import sys
from pathlib import Path
import faiss

# Fix module path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(PROJECT_ROOT))

from ingestion.pdf_ingestion import ingest_pdf
from pipeline_incremental.pipeline_incremental import run_incremental_pipeline

STORAGE = PROJECT_ROOT / "storage"
FAISS_FILE = STORAGE / "content_units.faiss"
STATE_FILE = STORAGE / "pipeline_state.json"


def main():

    if len(sys.argv) != 2:
        print("Usage: python -m test.test_full_pipeline <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).resolve()

    if not pdf_path.exists():
        print("ERROR: File not found.")
        sys.exit(1)

    print("\n=== INCREMENTAL PIPELINE TEST START ===\n")

    # 1️⃣ Ingest
    print("1️⃣ Running ingestion...")
    result = ingest_pdf(str(pdf_path))
    print(f"   ✔ Document created: {result.document.doc_id}")

    # 2️⃣ Incremental Pipeline (Clean → Chunk → Embed → FAISS Append)
    print("2️⃣ Running incremental processing...")
    pipeline_result = run_incremental_pipeline()

    print(f"   ✔ New Docs Processed: {pipeline_result['new_docs']}")
    print(f"   ✔ New Chunks Created: {len(pipeline_result['new_chunks'])}")

    # 3️⃣ Validate FAISS
    if FAISS_FILE.exists():
        index = faiss.read_index(str(FAISS_FILE))
        print(f"   ✔ FAISS Total Vectors: {index.ntotal}")
    else:
        print("   ❌ FAISS index missing!")

    print("\n🎉 INCREMENTAL PIPELINE TEST COMPLETED\n")


if __name__ == "__main__":
    main()