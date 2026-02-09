import fitz
from pathlib import Path
from schema.ingestion_schema import ContentUnit, UnitType, ExtractionMethod
from ingestion.hashing import sha256_bytes


def extract_images(doc: fitz.Document, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    units = []

    for page_no, page in enumerate(doc, start=1): #type: ignore
        for idx, img in enumerate(page.get_images(full=True)):
            raw = doc.extract_image(img[0])["image"]
            path = out_dir / f"p{page_no}_{idx}.png"
            path.write_bytes(raw)

            units.append(ContentUnit(
                doc_id="TEMP",
                unit_type=UnitType.IMAGE,
                extraction_method=ExtractionMethod.VISION,
                content=str(path),
                content_hash=sha256_bytes(raw),
                page_start=page_no,
                page_end=page_no,
                order_index=0
            ))
    return units