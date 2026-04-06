from enum import Enum
from typing import Dict, Any
from pydantic import Field
from models.odm import NoSQLBaseDocument


class IngestionStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    INGESTED   = "ingested"
    FAILED     = "failed"


class IndexingStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED  = "failed"


class PDFDocument(NoSQLBaseDocument):
    filename        : str
    path            : str
    sha256          : str
    markdown_path   : str
    metadata        : Dict[str, Any]  = Field(default_factory=dict)
    version         : int             = 1
    status          : IngestionStatus = IngestionStatus.PENDING
    indexing_status : IndexingStatus  = IndexingStatus.PENDING

    class Settings:
        name = "raw_documents"


class ImageDocument(NoSQLBaseDocument):
    pass
