from abc import ABC
from typing import Optional, List
from pydantic import Field
from models.ovm import VectorBaseDocument
from RAG.base.constants import DataCategory


class EmbeddedChunk(VectorBaseDocument, ABC):
    """
    Base class for embedded data chunks.
    """
    content: str
    embedding: List[float] | None
    sparse_embedding: dict | None = Field(default=None)
    filename: str
    sha256: str
    bank_type: Optional[str] = Field(default=None)
    month: Optional[str] = Field(default=None)
    year: Optional[int] = Field(default=None)
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        """
        Combine multiple chunks into a single text context.
        """
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Filename: {chunk.filename}
            Month: {chunk.month}
            Year: {chunk.year}
            Content: {chunk.content}\n
            """
        return context


class EmbeddedPDFChunk(EmbeddedChunk):
    """
    Embedded representation of a PDF chunk.
    """

    class Config:
        name = "embedded_pdfs"
        category = DataCategory.PDF
        use_vector_index = True
        use_hybrid_index = True

    @classmethod
    def get_payload_indexes(cls) -> dict:
        return {
            "bank_type": "keyword",
            "month": "keyword",
            "year": "integer",
        }