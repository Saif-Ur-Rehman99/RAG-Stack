from abc import ABC
from pydantic import Field
from models.ovm import VectorBaseDocument
from RAG.base.constants import DataCategory



# Chunking Data Models
class Chunk(VectorBaseDocument, ABC):
    """
    Base class for all chunked document types.
    """
    content: str
    filename: str
    sha256: str
    bank_type: str
    month: str
    year: str
    metadata: dict = Field(default_factory=dict)


class PDFChunk(Chunk):
    """
    Represents a single chunk from a cleaned PDF document.
    """
    pdf: str | None = None
    class Config:
        category = DataCategory.PDF

# class ImageChunk(Chunk):
#     image: Optional[str] = None

#     class Config:
#         category = DataCategory.IMAGES
