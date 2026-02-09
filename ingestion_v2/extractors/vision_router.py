# def route_from_clip(interp: dict) -> str:
#     if interp["clip_confidence"] < 0.35:
#         return "text_ocr"

#     label = interp["clip_label"].lower()

#     if "table" in label:
#         return "table_ocr"
#     if "text" in label or "scanned" in label:
#         return "text_ocr"
#     if "chart" in label or "graph" in label or "diagram" in label:
#         return "caption"
#     return "ignore"

# ingestion/extractors/vision_router.py

from typing import Literal, Dict

Route = Literal[
    "caption",
    "text_ocr",
    "table_ocr",
    "fallback_ocr",
    "ignore",
]


def route_from_clip(interp: Dict) -> Route:
    """
    Decide downstream processing based on CLIP output.

    Rules:
    - High confidence → trust CLIP
    - Medium/low confidence → OCR fallback
    """

    confidence = interp["clip_confidence"]
    label = interp["clip_label"].lower()

    # -------------------------
    # CLIP unsure → OCR fallback
    # -------------------------
    if confidence < 0.55:
        return "fallback_ocr"

    # -------------------------
    # Confident routing
    # -------------------------
    if "table" in label:
        return "table_ocr"

    if "scanned" in label or "text" in label:
        return "text_ocr"

    if "chart" in label or "graph" in label or "diagram" in label:
        return "caption"

    return "fallback_ocr"