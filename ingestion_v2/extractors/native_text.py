import fitz
from schema.ingestion_schema import ContentUnit, UnitType, ExtractionMethod
from ingestion.hashing import content_hash


def extract_native_text(doc: fitz.Document):
    units = []
    for page_no, page in enumerate(doc, start=1): #type: ignore
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if not text:
                continue
            units.append(ContentUnit(
                doc_id="TEMP",
                unit_type=UnitType.TEXT,
                extraction_method=ExtractionMethod.NATIVE,
                content=text,
                content_hash=content_hash(UnitType.TEXT, ExtractionMethod.NATIVE, text),
                page_start=page_no,
                page_end=page_no,
                order_index=0
            ))
    return units