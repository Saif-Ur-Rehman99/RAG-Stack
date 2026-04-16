from abc import ABC
from typing import Optional
from models.ovm import VectorBaseDocument
from RAG.base.constants import DataCategory



class CleanedDocument(VectorBaseDocument, ABC):
    """
    Base class for all cleaned documents.
    """
    content: str


class PDFCleanedDocument(CleanedDocument):
    """
    Cleaned PDF document model ready for embedding.
    """
    filename: Optional[str] = None
    path: Optional[str] = None
    sha256: Optional[str] = None
    bank_type: str = None
    month: str = None
    year: str = None

    class Config:
        name = "cleaned_pdfs"
        category = DataCategory.PDF
        use_vector_index = False


# class ImageCleanedDocument(CleanedDocument):
#     image: Optional[str] = None

#     class Config:
#         name = "cleaned_images"
#         category = DataCategory.IMAGES
#         use_vector_index = False