"""
End-to-end test for PDF ingestion.

This script:
- Takes a PDF file
- Runs full ingestion (native text, tables, OCR, images)
- Prints a structured summary
- Verifies basic invariants

Run:
    python test/test_pdf_ingestion.py path/to/file.pdf
"""

import sys
from pathlib import Path
from collections import Counter

from ingestion.pdf_ingestion import ingest_pdf
from schema.ingestion_schema import UnitType, ExtractionMethod


def main() -> None:
    # Input validation
    if len(sys.argv) != 2:
        print("Usage: python test/test_pdf_ingestion.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).resolve()

    if not pdf_path.exists():
        print(f"ERROR: File does not exist: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print("ERROR: Input file must be a PDF")
        sys.exit(1)

    # Run ingestion
    print(f"\n📄 Ingesting PDF: {pdf_path}\n")

    try:
        result = ingest_pdf(str(pdf_path))
    except Exception as e:
        print("❌ Ingestion failed:")
        raise

    # Basic assertions (sanity checks)
    assert result.source.source_id.startswith("src_")
    assert result.document.doc_id.startswith("doc_")
    assert len(result.units) > 0, "No content units were extracted"

    # Summarize results
    type_counter = Counter()
    method_counter = Counter()

    for unit in result.units:
        type_counter[unit.unit_type.value] += 1
        method_counter[unit.extraction_method.value] += 1

    print("✅ Ingestion successful\n")

    print("Source:")
    print(f"  source_id   : {result.source.source_id}")
    print(f"  file_hash  : {result.source.file_hash[:12]}…")

    print("\nDocument:")
    print(f"  doc_id     : {result.document.doc_id}")
    print(f"  doc_hash   : {result.document.doc_hash[:12]}…")
    print(f"  version    : {result.document.version}")

    print("\nContent Units Summary:")
    for k, v in sorted(type_counter.items()):
        print(f"  {k:>10}: {v}")

    print("\nExtraction Methods Summary:")
    for k, v in sorted(method_counter.items()):
        print(f"  {k:>10}: {v}")

    # Show a few samples (safe, non-verbose)
    print("\nSample Units:")
    for unit in result.units:
        print(
            f"- [{unit.unit_type.value}/{unit.extraction_method.value}] "
            f"page={unit.page_start} order={unit.order_index}"
        )

    print("\n🎉 Test completed successfully.")


if __name__ == "__main__":
    main()