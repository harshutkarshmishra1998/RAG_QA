# from pathlib import Path
# from schema.ingestion_schema import ContentUnit, UnitType, ExtractionMethod
# from ingestion.hashing import content_hash
# from ingestion.vision_interpretation import interpret_image


# def extract_vision_caption(image_unit):
#     interp = interpret_image(Path(image_unit.content))
#     return ContentUnit(
#         doc_id="TEMP",
#         unit_type=UnitType.TEXT,
#         extraction_method=ExtractionMethod.VISION,
#         content=interp["summary"],
#         content_hash=content_hash(UnitType.TEXT, ExtractionMethod.VISION, interp["summary"]),
#         page_start=image_unit.page_start,
#         page_end=image_unit.page_end,
#         order_index=0,
#         metadata={
#             "derived_from_image": image_unit.content,
#             "clip_label": interp["clip_label"],
#             "clip_confidence": interp["clip_confidence"],
#             "image_type": interp["image_type"],
#         }
#     ), interp

# ingestion/extractors/vision_caption.py

from pathlib import Path
from schema.ingestion_schema import ContentUnit, UnitType, ExtractionMethod
from ingestion.hashing import content_hash
from ingestion.vision_interpretation import interpret_image


def extract_vision_caption(image_unit):
    interp = interpret_image(Path(image_unit.content))

    unit = ContentUnit(
        doc_id="TEMP",
        unit_type=UnitType.TEXT,
        extraction_method=ExtractionMethod.VISION,
        content=interp["summary"],
        content_hash=content_hash(
            UnitType.TEXT,
            ExtractionMethod.VISION,
            interp["summary"].lower()
        ),
        page_start=image_unit.page_start,
        page_end=image_unit.page_end,
        order_index=0,
        metadata={
            "derived_from_image": image_unit.content,
            "clip_label": interp["clip_label"],
            "clip_confidence": interp["clip_confidence"],
            "image_type": interp["image_type"],
        }
    )

    return unit, interp