from pathlib import Path
from typing import Dict, Any, List

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# CLIP setup
# ============================================================

MODEL_NAME = "./clip-vit-base-patch32"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = CLIPModel.from_pretrained(MODEL_NAME, local_files_only=True)
_processor = CLIPProcessor.from_pretrained(MODEL_NAME, local_files_only=True)

_model.to(_device) #type: ignore
_model.eval()


# ============================================================
# Controlled semantic labels (NO hallucination)
# ============================================================

CANDIDATE_LABELS: List[str] = [
    "a graph showing trends over time",
    "a bar chart comparing numeric values",
    "a line chart",
    "a table of numeric data",
    "a technical diagram",
    "a neural network architecture diagram",
    "a flowchart",
    "a scanned document page",
    "a generic image"
]


# ============================================================
# Public API
# ============================================================

def interpret_image(image_path: Path) -> Dict[str, Any]:
    """
    Semantic interpretation using CLIP.

    Returns ONLY what CLIP is good at:
    - image category
    - semantic description
    - confidence

    Does NOT extract numbers or text.
    """

    image = Image.open(image_path).convert("RGB")

    inputs = _processor(
        text=CANDIDATE_LABELS,
        images=image,
        return_tensors="pt", #type: ignore
        padding=True #type: ignore
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    idx = int(probs.argmax())
    label = CANDIDATE_LABELS[idx]
    confidence = float(probs[idx])

    # Route-friendly type
    if "chart" in label or "graph" in label:
        image_type = "graph"
    elif "table" in label:
        image_type = "table"
    elif "diagram" in label or "flowchart" in label:
        image_type = "diagram"
    else:
        image_type = "other"

    return {
        "summary": f"The image appears to be {label}.",
        "clip_label": label,
        "clip_confidence": round(confidence, 4),
        "image_type": image_type
    }