import fitz
from pathlib import Path
from schema.ingestion_schema import Source, Document, IngestionResult, SourceType
from ingestion_v2.state import append_jsonl, load_source_index, SOURCES, DOCUMENTS, UNITS
from ingestion_v2.hashing import sha256_bytes, sha256_text
from ingestion_v2.base import assign_order

from ingestion_v2.extractors.native_text import extract_native_text
from ingestion_v2.extractors.native_tables import extract_native_tables
from ingestion_v2.extractors.images import extract_images
from ingestion_v2.extractors.vision_router import route_from_clip
from ingestion_v2.extractors.vision_caption import extract_vision_caption
from ingestion_v2.extractors.ocr_text import extract_ocr_text_from_image
from ingestion_v2.extractors.ocr_tables import extract_ocr_table_from_image


def ingest_pdf(pdf_path: str) -> IngestionResult:
    path = Path(pdf_path).resolve()
    file_hash = sha256_bytes(path.read_bytes())

    if file_hash in load_source_index():
        raise RuntimeError("Already ingested")

    source = Source(
        source_type=SourceType.PDF,
        source_uri=str(path),
        file_hash=file_hash
    )
    append_jsonl(SOURCES, source.model_dump())

    doc = fitz.open(path)
    units = []
    order = 0

    for extractor in [
        extract_native_text,
        lambda d: extract_images(d, Path("storage/artifacts/images")),
    ]:
        new = extractor(doc)
        order = assign_order(new, order)
        units.extend(new)

    for img in [u for u in units if u.unit_type.name == "IMAGE"]:
        caption, interp = extract_vision_caption(img)
        order = assign_order([caption], order)
        units.append(caption)

        route = route_from_clip(interp)
        if route == "text_ocr":
            text = extract_ocr_text_from_image(img.content)
            if text:
                order += 1
                units.append(caption.model_copy(update={
                    "unit_type": caption.unit_type.TEXT,
                    "extraction_method": caption.extraction_method.OCR,
                    "content": text,
                    "order_index": order
                }))

        if route == "table_ocr":
            table = extract_ocr_table_from_image(img.content)
            if table:
                order += 1
                units.append(caption.model_copy(update={
                    "unit_type": caption.unit_type.TABLE,
                    "extraction_method": caption.extraction_method.OCR,
                    "content": table,
                    "order_index": order
                }))

    tables = extract_native_tables(str(path))
    order = assign_order(tables, order)
    units.extend(tables)

    document = Document(
        source_id=source.source_id,
        doc_hash=sha256_text(" ".join(
            u.content for u in units if isinstance(u.content, str)
        ))
    )
    append_jsonl(DOCUMENTS, document.model_dump())

    final_units = []
    for u in units:
        fu = u.model_copy(update={"doc_id": document.doc_id})
        append_jsonl(UNITS, fu.model_dump())
        final_units.append(fu)

    return IngestionResult(source=source, document=document, units=final_units)