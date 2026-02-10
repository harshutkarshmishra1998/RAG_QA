import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def chunk_content_units(
    input_path: str | Path,
    *,
    min_tokens: int = 120,
    target_tokens: int = 450,
    max_tokens: int = 800,
    clip_conf_threshold: float = 0.25,
) -> Path:
    """
    V2 Semantic Chunker:
    - heading confidence scoring
    - semantic closure awareness
    - paragraph-aware splitting
    """

    input_path = Path(input_path)
    output_path = input_path.with_name(
        input_path.stem + "_chunked" + input_path.suffix
    )

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def heading_score(text: str) -> float:
        t = text.strip()
        if not t or len(t) > 140:
            return 0.0

        score = 0.0
        alpha = sum(c.isalpha() for c in t)
        upper = sum(c.isupper() for c in t)

        if t.isupper():
            score += 0.4
        if alpha and upper / alpha > 0.6:
            score += 0.3
        if re.match(r"^\d+(\.\d+)*\s", t):
            score += 0.3
        if not t.endswith("."):
            score += 0.1

        return min(score, 1.0)

    def is_heading(text: str) -> bool:
        return heading_score(text) >= 0.6

    def semantic_type(text: str) -> str:
        t = text.lower()

        if any(k in t for k in ["defined as", "refers to", "means that"]):
            return "definition"
        if any(k in t for k in ["step", "procedure", "process", "how to"]):
            return "procedure"
        if any(k in t for k in ["shall", "must", "required to"]):
            return "policy_rule"
        if re.match(r"^[-•\d]+\s", t):
            return "list"
        return "explanation"

    def should_keep_unit(unit: Dict) -> bool:
        text = unit.get("content", "").strip()
        if not text:
            return False
        if unit.get("unit_type") == "image":
            return unit.get("clip_confidence", 0.0) >= clip_conf_threshold
        return True

    # -------------------------------------------------
    # Load units
    # -------------------------------------------------

    units: List[Dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            if should_keep_unit(u):
                units.append(u)

    units.sort(key=lambda x: x["order_index"])

    # -------------------------------------------------
    # Build semantic blocks
    # -------------------------------------------------

    blocks = []
    current_units = []
    current_heading = None

    for u in units:
        text = u["content"]

        if is_heading(text):
            if current_units:
                blocks.append({
                    "heading": current_heading,
                    "units": current_units
                })
            current_heading = text.strip()
            current_units = [u]
        else:
            current_units.append(u)

    if current_units:
        blocks.append({
            "heading": current_heading,
            "units": current_units
        })

    # -------------------------------------------------
    # Chunk with semantic closure
    # -------------------------------------------------

    chunks = []
    chunk_index = 0

    active_units: List[Dict] = []
    active_tokens = 0
    active_heading: Optional[str] = None
    active_semantic: Optional[str] = None

    def flush_chunk(reason: str):
        nonlocal chunk_index, active_units, active_tokens, active_heading, active_semantic

        if not active_units:
            return

        pages = [
            u.get("page_start")
            for u in active_units
            if u.get("page_start") is not None
        ]

        text = "\n".join(u["content"] for u in active_units)

        chunks.append({
            "chunk_id": f"chunk_{chunk_index:06d}",
            "doc_id": active_units[0]["doc_id"],
            "chunk_index": chunk_index,
            "section_heading": active_heading,
            "semantic_type": active_semantic,
            "closure_reason": reason,
            "page_start": min(pages) if pages else None, #type: ignore
            "page_end": max(pages) if pages else None, #type: ignore
            "unit_ids": [u["unit_id"] for u in active_units],
            "text": text,
            "token_count": active_tokens,
            "semantic_density": len(active_units) / max(active_tokens, 1),
        })

        chunk_index += 1
        active_units = []
        active_tokens = 0
        active_heading = None
        active_semantic = None

    for block in blocks:
        heading = block["heading"]

        for u in block["units"]:
            u_text = u["content"]
            u_tokens = estimate_tokens(u_text)
            u_semantic = semantic_type(u_text)

            if not active_units:
                active_heading = heading
                active_semantic = u_semantic

            # Semantic closure trigger
            if (
                active_semantic != u_semantic
                and active_tokens >= min_tokens
            ):
                flush_chunk("semantic_boundary")
                active_heading = heading
                active_semantic = u_semantic

            # Token hard cap
            if active_tokens + u_tokens > max_tokens:
                flush_chunk("max_tokens")

            active_units.append(u)
            active_tokens += u_tokens

            # Natural closure
            if (
                active_tokens >= target_tokens
                and active_semantic in {"definition", "procedure", "list"}
            ):
                flush_chunk("semantic_complete")

    flush_chunk("end_of_document")

    # -------------------------------------------------
    # Write output
    # -------------------------------------------------

    with output_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return output_path