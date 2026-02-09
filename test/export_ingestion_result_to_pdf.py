from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table as PdfTable,
    TableStyle,
    Image as PdfImage
)
from reportlab.lib import colors

from ingestion.reload import (
    load_sources, load_documents, load_content_units
)
from schema.ingestion_schema import IngestionResult, UnitType


# ============================================================
# PDF Export
# ============================================================

def export_ingestion_result_to_pdf(
    result: IngestionResult,
    output_path: Path
) -> None:
    styles = getSampleStyleSheet()
    elements: list[Any] = []

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------
    elements.append(Paragraph("<b>RAG Ingestion Verification Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------
    elements.append(Paragraph("<b>Source</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Source ID: {result.source.source_id}", styles["Normal"]))
    elements.append(Paragraph(f"Source Type: {result.source.source_type}", styles["Normal"]))
    elements.append(Paragraph(f"Source URI: {result.source.source_uri}", styles["Normal"]))
    elements.append(Paragraph(f"File Hash: {result.source.file_hash}", styles["Normal"]))
    elements.append(Paragraph(f"Ingested At: {result.source.ingested_at}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # --------------------------------------------------------
    # Document
    # --------------------------------------------------------
    elements.append(Paragraph("<b>Document</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Document ID: {result.document.doc_id}", styles["Normal"]))
    elements.append(Paragraph(f"Source ID: {result.document.source_id}", styles["Normal"]))
    elements.append(Paragraph(f"Document Hash: {result.document.doc_hash}", styles["Normal"]))
    elements.append(Paragraph(f"Version: {result.document.version}", styles["Normal"]))
    elements.append(Paragraph(f"Created At: {result.document.created_at}", styles["Normal"]))
    elements.append(PageBreak())

    # --------------------------------------------------------
    # Content Units
    # --------------------------------------------------------
    elements.append(Paragraph("<b>Content Units</b>", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    for unit in result.units:
        elements.append(Paragraph(
            f"<b>Unit {unit.order_index}</b> "
            f"[{unit.unit_type.value} | {unit.extraction_method.value}]",
            styles["Heading3"]
        ))

        elements.append(Paragraph(f"Unit ID: {unit.unit_id}", styles["Normal"]))
        elements.append(Paragraph(f"Page: {unit.page_start}", styles["Normal"]))
        elements.append(Paragraph(f"Content Hash: {unit.content_hash}", styles["Normal"]))

        # ---- TEXT
        if unit.unit_type == UnitType.TEXT:
            text = unit.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(text, styles["Normal"]))

        # ---- TABLE
        elif unit.unit_type == UnitType.TABLE:
            table_data = unit.content
            if isinstance(table_data, list) and table_data:
                pdf_table = PdfTable(table_data, repeatRows=1)
                pdf_table.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ]))
                elements.append(Spacer(1, 6))
                elements.append(pdf_table)

        # ---- IMAGE
        elif unit.unit_type == UnitType.IMAGE:
            img_path = Path(unit.content)
            elements.append(Paragraph(f"Image Path: {img_path}", styles["Normal"]))
            if img_path.exists():
                try:
                    elements.append(Spacer(1, 6))
                    elements.append(PdfImage(str(img_path), width=300, preserveAspectRatio=True)) #type: ignore
                except Exception:
                    elements.append(Paragraph("⚠ Image could not be rendered", styles["Italic"]))

        elements.append(Spacer(1, 16))

    # --------------------------------------------------------
    # Build PDF
    # --------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    doc.build(elements)


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    sources = load_sources()
    documents = load_documents()
    units_by_doc = load_content_units()

    doc = next(iter(documents.values()))
    source = sources[doc.source_id]
    units = units_by_doc[doc.doc_id]

    result = IngestionResult(
        source=source,
        document=doc,
        units=units
    )

    output_pdf = Path("verification/ingestion_verification.pdf")
    export_ingestion_result_to_pdf(result, output_pdf)

    print(f"✅ Exported verification PDF to: {output_pdf.resolve()}")