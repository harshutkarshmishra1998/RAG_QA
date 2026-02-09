def route_from_clip(interp: dict) -> str:
    if interp["clip_confidence"] < 0.35:
        return "text_ocr"

    label = interp["clip_label"].lower()

    if "table" in label:
        return "table_ocr"
    if "text" in label or "scanned" in label:
        return "text_ocr"
    if "chart" in label or "graph" in label or "diagram" in label:
        return "caption"
    return "ignore"