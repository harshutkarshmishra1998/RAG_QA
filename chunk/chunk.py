import json
from pathlib import Path
from typing import Dict, List


def chunk_content_units(
    input_path: str | Path,
    *,
    min_tokens: int = 120,
    target_tokens: int = 450,
    max_tokens: int = 800,
    clip_conf_threshold: float = 0.25,
) -> Path:
    """
    Block-aware, heading-anchored chunking for cleaned content units.

    Args:
        input_path: Path to content_units.jsonl
        min_tokens: Minimum chunk size before allowing a split
        target_tokens: Soft target chunk size
        max_tokens: Hard chunk cap
        clip_conf_threshold: Vision unit confidence gate

    Returns:
        Path to generated *_chunked.jsonl file
    """

    input_path = Path(input_path)
    output_path = input_path.with_name(
        input_path.stem + "_chunked" + input_path.suffix
    )

    # -------------------------
    # Helpers
    # -------------------------

    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def is_heading(text: str) -> bool:
        t = text.strip()
        if not t or len(t) > 120:
            return False

        alpha = sum(c.isalpha() for c in t)
        upper = sum(c.isupper() for c in t)
        ratio = upper / alpha if alpha else 0.0

        return t.isupper() or ratio > 0.6 or t[:2].isdigit()

    def should_keep_unit(unit: Dict) -> bool:
        text = unit.get("content", "").strip()
        if not text:
            return False

        if unit.get("unit_type") == "image":
            return unit.get("clip_confidence", 0.0) >= clip_conf_threshold

        return True

    # -------------------------
    # Load + filter units
    # -------------------------

    units: List[Dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            if should_keep_unit(u):
                units.append(u)

    units.sort(key=lambda x: x["order_index"])

    # -------------------------
    # Build heading blocks
    # -------------------------

    blocks = []
    current_block = []
    current_heading = None

    for u in units:
        text = u["content"]
        if is_heading(text):
            if current_block:
                blocks.append({
                    "heading": current_heading,
                    "units": current_block
                })
            current_heading = text.strip()
            current_block = [u]
        else:
            current_block.append(u)

    if current_block:
        blocks.append({
            "heading": current_heading,
            "units": current_block
        })

    # -------------------------
    # Chunk blocks
    # -------------------------

    chunks = []
    chunk_index = 0

    current_units = []
    current_tokens = 0
    current_heading = None

    def flush_chunk():
        nonlocal chunk_index, current_units, current_tokens

        if not current_units:
            return

        pages = [
            u.get("page_start")
            for u in current_units
            if u.get("page_start") is not None
        ]

        chunk = {
            "chunk_id": f"chunk_{chunk_index:06d}",
            "doc_id": current_units[0]["doc_id"],
            "chunk_index": chunk_index,
            "section_heading": current_heading,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "unit_ids": [u["unit_id"] for u in current_units],
            "text": "\n".join(u["content"] for u in current_units),
            "token_count": current_tokens,
        }

        chunks.append(chunk)
        chunk_index += 1
        current_units = []
        current_tokens = 0

    for block in blocks:
        block_heading = block["heading"]
        block_units = block["units"]

        block_text = "\n".join(u["content"] for u in block_units)
        block_tokens = estimate_tokens(block_text)

        # Force-split oversized blocks
        if block_tokens > max_tokens:
            for u in block_units:
                u_tokens = estimate_tokens(u["content"])

                if current_tokens + u_tokens > max_tokens:
                    flush_chunk()

                if not current_units:
                    current_heading = block_heading

                current_units.append(u)
                current_tokens += u_tokens

            flush_chunk()
            continue

        # Normal block flow
        if (
            current_tokens + block_tokens > target_tokens
            and current_tokens >= min_tokens
        ):
            flush_chunk()

        if not current_units:
            current_heading = block_heading

        for u in block_units:
            current_units.append(u)
            current_tokens += estimate_tokens(u["content"])

    flush_chunk()

    # -------------------------
    # Write output
    # -------------------------

    with output_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return output_path