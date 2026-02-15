from __future__ import annotations

from enum import Enum
from typing import Optional, Any, Dict, List
from uuid import uuid4
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ConfigDict


# Enums

class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    WEB = "web"
    GOOGLE_DRIVE = "google_drive"
    YOUTUBE = "youtube"


class UnitType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class ExtractionMethod(str, Enum):
    """
    How the content was extracted.
    Orthogonal to UnitType.
    """
    NATIVE = "native"    # digital text & tables
    OCR = "ocr"          # scanned PDFs
    VISION = "vision"    # diagrams & charts


# Source (Ingestion Gate)

class Source(BaseModel):
    """
    Raw origin of data.
    Used for deduplication and reload safety.
    """
    source_id: str = Field(
        default_factory=lambda: f"src_{uuid4().hex}"
    )

    source_type: SourceType = Field(
        ...,
        description="Type of the input source"
    )

    source_uri: str = Field(
        ...,
        description="Path or URL identifying the source"
    )

    file_hash: str = Field(
        ...,
        description="sha256 hash of raw source bytes"
    )

    checksum_algo: str = Field(
        default="sha256"
    )

    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = ConfigDict(extra="forbid")


# Document (Logical, Versioned)

class Document(BaseModel):
    """
    Logical document derived from a source.
    Versioned when extracted content changes.
    """
    doc_id: str = Field(
        default_factory=lambda: f"doc_{uuid4().hex}"
    )

    source_id: str = Field(
        ...,
        description="Parent source identifier"
    )

    title: Optional[str] = None
    language: Optional[str] = "en"

    doc_hash: str = Field(
        ...,
        description="sha256 hash of normalized extracted text"
    )

    version: int = Field(
        default=1
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )

    model_config = ConfigDict(extra="forbid")


# Content Unit (Atomic RAG Primitive)

class ContentUnit(BaseModel):
    """
    Smallest retrievable, chunkable unit.
    OCR text and OCR tables are represented via:
    unit_type + extraction_method
    """
    unit_id: str = Field(
        default_factory=lambda: f"unit_{uuid4().hex}"
    )

    doc_id: str = Field(
        ...,
        description="Parent document identifier"
    )

    unit_type: UnitType = Field(
        ...,
        description="Semantic content type"
    )

    extraction_method: ExtractionMethod = Field(
        ...,
        description="How the content was extracted"
    )

    content: Any = Field(
        ...,
        description="Raw content (text, table structure, image ref)"
    )

    content_hash: str = Field(
        ...,
        description="sha256 hash of normalized content + type + extraction"
    )

    page_start: Optional[int] = Field(
        default=None
    )

    page_end: Optional[int] = Field(
        default=None
    )

    bbox: Optional[List[int]] = Field(
        default=None,
        description="Bounding box [x1, y1, x2, y2]"
    )

    confidence: Optional[float] = Field(
        default=None,
        description="OCR / vision confidence score"
    )

    order_index: int = Field(
        ...,
        description="Deterministic order within document"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    model_config = ConfigDict(extra="forbid")


# Optional Convenience Wrapper

class IngestionResult(BaseModel):
    """
    Returned by ingestion pipelines.
    Not intended for direct persistence.
    """
    source: Source
    document: Document
    units: List[ContentUnit]

    model_config = ConfigDict(extra="forbid")