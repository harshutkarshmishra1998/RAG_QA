PDF
 ├─ Native text extraction
 ├─ Image extraction
 │    └─ CLIP image caption (always)
 ├─ OCR text on ALL images
 ├─ OCR tables on ALL images
 └─ Native table extraction

v1 — Maximal extraction

Pros
    Highest recall
    Captures scanned tables & figures
    Best for compliance, audits, dense PDFs

Cons
    OCR noise present
    Table duplication risk
    Heavier chunk filtering required downstream

This version assumes the retriever is smart.