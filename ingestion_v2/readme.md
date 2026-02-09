PDF
 ├─ Native text extraction
 ├─ Image extraction
 │    └─ CLIP semantic review
 │         ├─ Confident → route intelligently
 │         └─ Unsure → OCR fallback
 ├─ Native table extraction

v2 — Aggressive pruning

Pros
    Clean, minimal
    Almost no OCR hallucinations
    Faster indexing, cheaper embeddings

Cons
    Deletes real tables
    Loses scanned but meaningful content
    Blind to image-heavy PDFs

This version assumes documents are clean & text-native.