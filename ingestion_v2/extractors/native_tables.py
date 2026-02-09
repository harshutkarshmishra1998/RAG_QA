import pdfplumber
from schema.ingestion_schema import ContentUnit, UnitType, ExtractionMethod
from ingestion.hashing import content_hash


def extract_native_tables(pdf_path: str):
    units = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables() or []:
                if not table:
                    continue
                units.append(ContentUnit(
                    doc_id="TEMP",
                    unit_type=UnitType.TABLE,
                    extraction_method=ExtractionMethod.NATIVE,
                    content=table,
                    content_hash=content_hash(UnitType.TABLE, ExtractionMethod.NATIVE, table),
                    page_start=page_no,
                    page_end=page_no,
                    order_index=0
                ))
    return units