"""Artifact models passed between ZenML steps.

Keeping these as pydantic models means ZenML can serialize them
automatically and the dashboard renders them nicely.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class PDFCandidate(BaseModel):
    """A PDF file discovered on disk, before any processing."""
    filepath: str
    filename: str


class DedupedPDF(BaseModel):
    """Result of the dedup stage. If is_duplicate=True, downstream steps skip it."""
    filepath: str
    filename: str
    sha256: str
    is_duplicate: bool


class RoutedPDF(BaseModel):
    """A non-duplicate PDF tagged with which parser to use."""
    filepath: str
    filename: str
    sha256: str
    parser: str  # "PyMuPDFParser" | "LlamaParser"


class ParsedPDF(BaseModel):
    """A successfully parsed PDF with its markdown content in memory."""
    filepath: str
    filename: str
    sha256: str
    parser: str
    markdown: str


class ParseFailure(BaseModel):
    """A PDF that exhausted retries during parsing."""
    filepath: str
    filename: str
    sha256: str
    parser: str
    error: str


class PersistedPDF(BaseModel):
    """Final per-file record covering every file the pipeline saw."""
    filename: str
    status: str           # "inserted" | "skipped" | "failed"
    pages: int
    doc_id: Optional[str] = None
    parser: Optional[str] = None
    version: Optional[int] = None
    error: Optional[str] = None


class IngestionReport(BaseModel):
    """The final aggregated summary, returned by the pipeline."""
    total_found: int
    inserted: int
    skipped_duplicates: int
    failed: int
    documents: List[PersistedPDF]







