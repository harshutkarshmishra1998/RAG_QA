# import fitz
# from pathlib import Path
# from schema.ingestion_schema import Source, Document, IngestionResult, SourceType
# from ingestion_v2.state import append_jsonl, load_source_index, SOURCES, DOCUMENTS, UNITS
# from ingestion_v2.hashing import sha256_bytes, sha256_text
# from ingestion_v2.base import assign_order

# from ingestion_v2.extractors.native_text import extract_native_text
# from ingestion_v2.extractors.native_tables import extract_native_tables
# from ingestion_v2.extractors.images import extract_images
# from ingestion_v2.extractors.vision_router import route_from_clip
# from ingestion_v2.extractors.vision_caption import extract_vision_caption
# from ingestion_v2.extractors.ocr_text import extract_ocr_text_from_image
# from ingestion_v2.extractors.ocr_tables import extract_ocr_table_from_image


# def ingest_pdf(pdf_path: str) -> IngestionResult:
#     path = Path(pdf_path).resolve()
#     file_hash = sha256_bytes(path.read_bytes())

#     if file_hash in load_source_index():
#         raise RuntimeError("Already ingested")

#     source = Source(
#         source_type=SourceType.PDF,
#         source_uri=str(path),
#         file_hash=file_hash
#     )
#     append_jsonl(SOURCES, source.model_dump())

#     doc = fitz.open(path)
#     units = []
#     order = 0

#     for extractor in [
#         extract_native_text,
#         lambda d: extract_images(d, Path("storage/artifacts/images")),
#     ]:
#         new = extractor(doc)
#         order = assign_order(new, order)
#         units.extend(new)

#     for img in [u for u in units if u.unit_type.name == "IMAGE"]:
#         caption, interp = extract_vision_caption(img)
#         order = assign_order([caption], order)
#         units.append(caption)

#         route = route_from_clip(interp)
#         if route == "text_ocr":
#             text = extract_ocr_text_from_image(img.content)
#             if text:
#                 order += 1
#                 units.append(caption.model_copy(update={
#                     "unit_type": caption.unit_type.TEXT,
#                     "extraction_method": caption.extraction_method.OCR,
#                     "content": text,
#                     "order_index": order
#                 }))

#         if route == "table_ocr":
#             table = extract_ocr_table_from_image(img.content)
#             if table:
#                 order += 1
#                 units.append(caption.model_copy(update={
#                     "unit_type": caption.unit_type.TABLE,
#                     "extraction_method": caption.extraction_method.OCR,
#                     "content": table,
#                     "order_index": order
#                 }))

#     tables = extract_native_tables(str(path))
#     order = assign_order(tables, order)
#     units.extend(tables)

#     document = Document(
#         source_id=source.source_id,
#         doc_hash=sha256_text(" ".join(
#             u.content for u in units if isinstance(u.content, str)
#         ))
#     )
#     append_jsonl(DOCUMENTS, document.model_dump())

#     final_units = []
#     for u in units:
#         fu = u.model_copy(update={"doc_id": document.doc_id})
#         append_jsonl(UNITS, fu.model_dump())
#         final_units.append(fu)

#     return IngestionResult(source=source, document=document, units=final_units)

from pathlib import Path
import fitz  # PyMuPDF

from schema.ingestion_schema import (
    Source,
    Document,
    IngestionResult,
    SourceType,
    UnitType,
    ExtractionMethod,
    ContentUnit,
)

from ingestion.state import (
    append_jsonl,
    load_source_index,
    SOURCES,
    DOCUMENTS,
    UNITS,
)

from ingestion_v2.hashing import sha256_bytes, sha256_text, content_hash
from ingestion_v2.base import assign_order

# -----------------------------
# Reusable extractors
# -----------------------------
from ingestion_v2.extractors.native_text import extract_native_text
from ingestion_v2.extractors.native_tables import extract_native_tables
from ingestion_v2.extractors.images import extract_images

from ingestion_v2.extractors.vision_caption import extract_vision_caption
from ingestion_v2.extractors.vision_router import route_from_clip

from ingestion_v2.extractors.ocr_text import extract_ocr_text_from_image
from ingestion_v2.extractors.ocr_tables import extract_ocr_table_from_image


# ============================================================
# PDF INGESTION (CLIP-routed, schema-safe)
# ============================================================

def ingest_pdf(pdf_path: str) -> IngestionResult:
    pdf_path = Path(pdf_path).resolve() #type: ignore

    if not pdf_path.exists(): #type: ignore
        raise FileNotFoundError(pdf_path)

    # --------------------------------------------------------
    # 1. Source de-duplication (reload safe)
    # --------------------------------------------------------
    file_hash = sha256_bytes(pdf_path.read_bytes()) #type: ignore
    source_index = load_source_index()

    if file_hash in source_index:
        raise RuntimeError("PDF already ingested")

    source = Source(
        source_type=SourceType.PDF,
        source_uri=str(pdf_path),
        file_hash=file_hash,
    )
    append_jsonl(SOURCES, source.model_dump())

    # --------------------------------------------------------
    # 2. Open PDF
    # --------------------------------------------------------
    doc = fitz.open(pdf_path)

    units: list[ContentUnit] = []
    order_index = 0

    # --------------------------------------------------------
    # 3. Native text extraction
    # --------------------------------------------------------
    native_text_units = extract_native_text(doc)
    order_index = assign_order(native_text_units, order_index)
    units.extend(native_text_units)

    # --------------------------------------------------------
    # 4. Image extraction (artifacts only)
    # --------------------------------------------------------
    image_dir = Path("storage/artifacts/images")
    image_units = extract_images(doc, image_dir)
    order_index = assign_order(image_units, order_index)
    units.extend(image_units)

    # --------------------------------------------------------
    # 5. CLIP routing + OCR fallback
    # --------------------------------------------------------
    for img_unit in image_units:

        # ---- 5.1 CLIP semantic caption (always)
        caption_unit, interp = extract_vision_caption(img_unit)
        order_index += 1
        caption_unit.order_index = order_index
        units.append(caption_unit)

        # ---- 5.2 Routing decision
        route = route_from_clip(interp)

        # ---- 5.3 Confident routes
        if route == "text_ocr":
            text = extract_ocr_text_from_image(img_unit.content)
            if text:
                order_index += 1
                units.append(ContentUnit(
                    doc_id="TEMP",
                    unit_type=UnitType.TEXT,
                    extraction_method=ExtractionMethod.OCR,
                    content=text,
                    content_hash=content_hash(
                        UnitType.TEXT,
                        ExtractionMethod.OCR,
                        text.lower()
                    ),
                    page_start=img_unit.page_start,
                    page_end=img_unit.page_end,
                    order_index=order_index,
                ))

        elif route == "table_ocr":
            table = extract_ocr_table_from_image(img_unit.content)
            if table:
                order_index += 1
                units.append(ContentUnit(
                    doc_id="TEMP",
                    unit_type=UnitType.TABLE,
                    extraction_method=ExtractionMethod.OCR,
                    content=table,
                    content_hash=content_hash(
                        UnitType.TABLE,
                        ExtractionMethod.OCR,
                        table
                    ),
                    page_start=img_unit.page_start,
                    page_end=img_unit.page_end,
                    order_index=order_index,
                ))

        # ---- 5.4 CLIP unsure → OCR fallback (MAX RECALL)
        elif route == "fallback_ocr":

            # OCR text
            text = extract_ocr_text_from_image(img_unit.content)
            if text:
                order_index += 1
                units.append(ContentUnit(
                    doc_id="TEMP",
                    unit_type=UnitType.TEXT,
                    extraction_method=ExtractionMethod.OCR,
                    content=text,
                    content_hash=content_hash(
                        UnitType.TEXT,
                        ExtractionMethod.OCR,
                        text.lower()
                    ),
                    page_start=img_unit.page_start,
                    page_end=img_unit.page_end,
                    order_index=order_index,
                ))

            # OCR table (best-effort)
            table = extract_ocr_table_from_image(img_unit.content)
            if table:
                order_index += 1
                units.append(ContentUnit(
                    doc_id="TEMP",
                    unit_type=UnitType.TABLE,
                    extraction_method=ExtractionMethod.OCR,
                    content=table,
                    content_hash=content_hash(
                        UnitType.TABLE,
                        ExtractionMethod.OCR,
                        table
                    ),
                    page_start=img_unit.page_start,
                    page_end=img_unit.page_end,
                    order_index=order_index,
                ))

        # ---- 5.5 ignore → nothing else
        else:
            pass

    # --------------------------------------------------------
    # 6. Native table extraction (PDF structure)
    # --------------------------------------------------------
    native_table_units = extract_native_tables(str(pdf_path))
    order_index = assign_order(native_table_units, order_index)
    units.extend(native_table_units)

    # --------------------------------------------------------
    # 7. Create Document
    # --------------------------------------------------------
    text_for_hash = " ".join(
        u.content for u in units if isinstance(u.content, str)
    )

    document = Document(
        source_id=source.source_id,
        doc_hash=sha256_text(text_for_hash),
    )
    append_jsonl(DOCUMENTS, document.model_dump())

    # --------------------------------------------------------
    # 8. Persist content units
    # --------------------------------------------------------
    final_units: list[ContentUnit] = []

    for u in units:
        fu = u.model_copy(update={"doc_id": document.doc_id})
        append_jsonl(UNITS, fu.model_dump())
        final_units.append(fu)

    # --------------------------------------------------------
    # 9. Return ingestion result
    # --------------------------------------------------------
    return IngestionResult(
        source=source,
        document=document,
        units=final_units,
    )