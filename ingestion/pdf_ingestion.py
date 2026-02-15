from pathlib import Path
from typing import List
import fitz
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
from ingestion.hashing import sha256_bytes, sha256_text, content_hash
from ingestion.state import load_source_index, append_jsonl, SOURCES, DOCUMENTS, UNITS
from ingestion.vision_interpretation import interpret_image


# Artifacts

# ARTIFACT_DIR = Path("storage/artifacts")
# IMAGE_DIR = ARTIFACT_DIR / "images"
# IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Paths

BASE_DIR_NEW = Path(__file__).resolve().parents[1]  # stable path (very important)

STORAGE = BASE_DIR_NEW / "storage"
ARTIFACT_DIR = STORAGE / "artifacts"
IMAGE_DIR = ARTIFACT_DIR / "images"

SOURCES = STORAGE / "sources.jsonl"
DOCUMENTS = STORAGE / "documents.jsonl"
UNITS = STORAGE / "content_units.jsonl"

# Initialization (idempotent, safe to call anytime)

def ensure_storage():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for file in (SOURCES, DOCUMENTS, UNITS):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch(exist_ok=True)

ensure_storage()

# Runtime checks

def _require_binary(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found in PATH")


_require_binary("tesseract")


# Helpers

def normalize_text(text: str) -> str:
    return " ".join(text.split()).lower()


# Main

def ingest_pdf(pdf_path: str) -> IngestionResult:
    pdf_path = Path(pdf_path).resolve() #type: ignore
    file_bytes = pdf_path.read_bytes() #type: ignore
    file_hash = sha256_bytes(file_bytes)

    # --------------------------------------------------------
    # Source gate
    # --------------------------------------------------------
    if file_hash in load_source_index():
        raise RuntimeError("PDF already ingested")

    source = Source(
        source_type=SourceType.PDF,
        source_uri=str(pdf_path),
        file_hash=file_hash
    )
    append_jsonl(SOURCES, source.model_dump())

    units: List[ContentUnit] = []
    order_index = 0
    doc_text_accumulator: List[str] = []

    doc = fitz.open(pdf_path)

    # --------------------------------------------------------
    # Native text + images
    # --------------------------------------------------------
    for page_no, page in enumerate(doc, start=1): #type: ignore
        # ---- Native text
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if not text:
                continue

            norm = normalize_text(text)
            doc_text_accumulator.append(norm)
            order_index += 1

            units.append(ContentUnit(
                doc_id="TEMP",
                unit_type=UnitType.TEXT,
                extraction_method=ExtractionMethod.NATIVE,
                content=text,
                content_hash=content_hash(UnitType.TEXT, ExtractionMethod.NATIVE, norm),
                page_start=page_no,
                page_end=page_no,
                order_index=order_index
            ))

        # ---- Images + CLIP
        for img_idx, img in enumerate(page.get_images(full=True)):
            img_bytes = doc.extract_image(img[0])["image"]

            image_path = IMAGE_DIR / f"{source.source_id}_p{page_no}_{img_idx}.png"
            image_path.write_bytes(img_bytes)

            order_index += 1
            image_unit = ContentUnit(
                doc_id="TEMP",
                unit_type=UnitType.IMAGE,
                extraction_method=ExtractionMethod.VISION,
                content=str(image_path),
                content_hash=sha256_bytes(img_bytes),
                page_start=page_no,
                page_end=page_no,
                order_index=order_index
            )
            units.append(image_unit)

            # ---- CLIP semantic interpretation
            interp = interpret_image(image_path) #type: ignore

            order_index += 1
            units.append(ContentUnit(
                doc_id="TEMP",
                unit_type=UnitType.TEXT,
                extraction_method=ExtractionMethod.VISION,
                content=interp["summary"],
                content_hash=content_hash(
                    UnitType.TEXT, ExtractionMethod.VISION, interp["summary"].lower()
                ),
                page_start=page_no,
                page_end=page_no,
                order_index=order_index,
                metadata={
                    "derived_from_image": str(image_path),
                    "clip_label": interp["clip_label"],
                    "clip_confidence": interp["clip_confidence"],
                    "image_type": interp["image_type"]
                }
            ))

    # --------------------------------------------------------
    # Native tables
    # --------------------------------------------------------
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables() or []:
                if not table:
                    continue

                norm_table = [[cell or "" for cell in row] for row in table]
                order_index += 1

                units.append(ContentUnit(
                    doc_id="TEMP",
                    unit_type=UnitType.TABLE,
                    extraction_method=ExtractionMethod.NATIVE,
                    content=norm_table,
                    content_hash=content_hash(UnitType.TABLE, ExtractionMethod.NATIVE, norm_table),
                    page_start=page_no,
                    page_end=page_no,
                    order_index=order_index
                ))

    # --------------------------------------------------------
    # OCR text + OCR tables
    # --------------------------------------------------------
    for page_no, page in enumerate(doc, start=1): #type: ignore
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes()))

        ocr_text = pytesseract.image_to_string(img).strip()
        if ocr_text:
            norm = normalize_text(ocr_text)
            doc_text_accumulator.append(norm)
            order_index += 1

            units.append(ContentUnit(
                doc_id="TEMP",
                unit_type=UnitType.TEXT,
                extraction_method=ExtractionMethod.OCR,
                content=ocr_text,
                content_hash=content_hash(UnitType.TEXT, ExtractionMethod.OCR, norm),
                page_start=page_no,
                page_end=page_no,
                order_index=order_index
            ))

        # OCR table heuristic
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        rows = {}
        for i, txt in enumerate(data["text"]):
            if txt.strip():
                rows.setdefault(data["block_num"][i], []).append(txt)

        if rows:
            table = list(rows.values())
            order_index += 1

            units.append(ContentUnit(
                doc_id="TEMP",
                unit_type=UnitType.TABLE,
                extraction_method=ExtractionMethod.OCR,
                content=table,
                content_hash=content_hash(UnitType.TABLE, ExtractionMethod.OCR, table),
                page_start=page_no,
                page_end=page_no,
                order_index=order_index
            ))

    # --------------------------------------------------------
    # Document + persistence
    # --------------------------------------------------------
    document = Document(
        source_id=source.source_id,
        doc_hash=sha256_text(" ".join(doc_text_accumulator))
    )
    append_jsonl(DOCUMENTS, document.model_dump())

    final_units = []
    for u in units:
        fu = u.model_copy(update={"doc_id": document.doc_id})
        append_jsonl(UNITS, fu.model_dump())
        final_units.append(fu)

    return IngestionResult(
        source=source,
        document=document,
        units=final_units
    )