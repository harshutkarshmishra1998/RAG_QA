from pathlib import Path
from typing import List
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import os
import shutil

from schema.ingestion_schema import (
    Source, Document, ContentUnit,
    SourceType, UnitType, ExtractionMethod, IngestionResult
)
from ingestion.hashing import (
    sha256_bytes, sha256_text, content_hash
)
from ingestion.state import (
    load_source_index, append_jsonl,
    SOURCES, DOCUMENTS, UNITS
)


ARTIFACT_DIR = Path("storage/artifacts")
IMAGE_DIR = ARTIFACT_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Runtime dependency resolution (executed at import time)
# Assumes `.env` has already been loaded by caller
# ============================================================

def _resolve_poppler_path() -> str | None:
    """
    Resolve Poppler binaries.

    Priority:
    1. POPPLER_PATH env var (directory)
    2. PATH lookup (pdftoppm)

    Returns:
        Directory path or None (use PATH)
    """
    poppler_path = os.getenv("POPPLER_PATH")

    if poppler_path:
        if not os.path.isdir(poppler_path):
            raise RuntimeError(f"Invalid POPPLER_PATH: {poppler_path}")
        return poppler_path

    if shutil.which("pdftoppm"):
        return None

    raise RuntimeError(
        "Poppler not found. Set POPPLER_PATH or add Poppler to PATH."
    )


def _resolve_tesseract() -> None:
    """
    Resolve Tesseract executable.

    Priority:
    1. TESSERACT_PATH env var (full path)
    2. PATH lookup
    """
    tesseract_path = os.getenv("TESSERACT_PATH")

    if tesseract_path:
        if not os.path.isfile(tesseract_path):
            raise RuntimeError(f"Invalid TESSERACT_PATH: {tesseract_path}")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return

    if shutil.which("tesseract"):
        return

    raise RuntimeError(
        "Tesseract not found. Set TESSERACT_PATH or add it to PATH."
    )


# Resolve once (fail fast)
POPPLER_PATH = _resolve_poppler_path()
_resolve_tesseract()


# ============================================================
# Helpers
# ============================================================

def normalize_text(text: str) -> str:
    return " ".join(text.split()).lower()


# ============================================================
# Main Ingestion Function
# ============================================================

def ingest_pdf(pdf_path: str) -> IngestionResult:
    pdf_path = Path(pdf_path).resolve() #type: ignore
    file_bytes = pdf_path.read_bytes() #type: ignore
    file_hash = sha256_bytes(file_bytes)

    # --------------------------------------------------------
    # Stage 1: Source Gate (Idempotency)
    # --------------------------------------------------------
    source_index = load_source_index()
    if file_hash in source_index:
        raise RuntimeError("PDF already ingested (file hash exists)")

    source = Source(
        source_type=SourceType.PDF,
        source_uri=str(pdf_path),
        file_hash=file_hash
    )
    append_jsonl(SOURCES, source.model_dump())

    units: List[ContentUnit] = []
    order_index = 0
    all_text_for_doc_hash: List[str] = []

    # --------------------------------------------------------
    # Stage 2: Native Text + Images (PyMuPDF)
    # --------------------------------------------------------
    doc = fitz.open(pdf_path)

    for page_no, page in enumerate(doc, start=1): #type: ignore
        # ---- Native text
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue

            norm = normalize_text(text)
            all_text_for_doc_hash.append(norm)
            order_index += 1

            units.append(ContentUnit(
                doc_id="TEMP",
                unit_type=UnitType.TEXT,
                extraction_method=ExtractionMethod.NATIVE,
                content=text,
                content_hash=content_hash(
                    UnitType.TEXT, ExtractionMethod.NATIVE, norm
                ),
                page_start=page_no,
                page_end=page_no,
                order_index=order_index
            ))

        # ---- Images (Vision placeholder)
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]

            image_path = IMAGE_DIR / f"{source.source_id}_p{page_no}_{img_idx}.png"
            image_path.write_bytes(img_bytes)

            order_index += 1
            units.append(ContentUnit(
                doc_id="TEMP",
                unit_type=UnitType.IMAGE,
                extraction_method=ExtractionMethod.VISION,
                content=str(image_path),
                content_hash=sha256_bytes(img_bytes),
                page_start=page_no,
                page_end=page_no,
                order_index=order_index
            ))

    # --------------------------------------------------------
    # Stage 3: Native Tables (pdfplumber)
    # --------------------------------------------------------
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue

                norm_table = [[cell or "" for cell in row] for row in table]
                order_index += 1

                units.append(ContentUnit(
                    doc_id="TEMP",
                    unit_type=UnitType.TABLE,
                    extraction_method=ExtractionMethod.NATIVE,
                    content=norm_table,
                    content_hash=content_hash(
                        UnitType.TABLE, ExtractionMethod.NATIVE, norm_table
                    ),
                    page_start=page_no,
                    page_end=page_no,
                    order_index=order_index
                ))

    # --------------------------------------------------------
    # Stage 4: OCR (Text + Tables)
    # --------------------------------------------------------
    for page_no, page in enumerate(doc, start=1): #type: ignore
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes()))

        ocr_text = pytesseract.image_to_string(img).strip()
        if ocr_text:
            norm = normalize_text(ocr_text)
            all_text_for_doc_hash.append(norm)
            order_index += 1

            units.append(ContentUnit(
                doc_id="TEMP",
                unit_type=UnitType.TEXT,
                extraction_method=ExtractionMethod.OCR,
                content=ocr_text,
                content_hash=content_hash(
                    UnitType.TEXT, ExtractionMethod.OCR, norm
                ),
                page_start=page_no,
                page_end=page_no,
                order_index=order_index,
                confidence=None
            ))

        # OCR tables (best-effort)
        try:
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            rows = {}
            for i, txt in enumerate(ocr_data["text"]):
                if txt.strip():
                    row = ocr_data["block_num"][i]
                    rows.setdefault(row, []).append(txt)

            if rows:
                table = list(rows.values())
                order_index += 1

                units.append(ContentUnit(
                    doc_id="TEMP",
                    unit_type=UnitType.TABLE,
                    extraction_method=ExtractionMethod.OCR,
                    content=table,
                    content_hash=content_hash(
                        UnitType.TABLE, ExtractionMethod.OCR, table
                    ),
                    page_start=page_no,
                    page_end=page_no,
                    order_index=order_index
                ))
        except Exception:
            pass  # OCR tables are best-effort

    # --------------------------------------------------------
    # Stage 5: Document Hash + Persistence
    # --------------------------------------------------------
    doc_hash = sha256_text(" ".join(all_text_for_doc_hash))

    document = Document(
        source_id=source.source_id,
        doc_hash=doc_hash
    )
    append_jsonl(DOCUMENTS, document.model_dump())

    finalized_units = []
    for unit in units:
        final_unit = unit.model_copy(update={"doc_id": document.doc_id})
        append_jsonl(UNITS, final_unit.model_dump())
        finalized_units.append(final_unit)

    return IngestionResult(
        source=source,
        document=document,
        units=finalized_units
    )