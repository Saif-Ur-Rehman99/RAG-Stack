from abc import ABC, abstractmethod
from typing import Any
from pydantic import Field

from models.ovm import VectorBaseDocument
from RAG.base.constants import DataCategory


class Query(VectorBaseDocument):
    content: str
    metadata: dict = Field(default_factory=dict)
    
    class Config:
        category = DataCategory.QUERIES

    @classmethod
    def from_str(cls, query: str) -> "Query":
        return Query(content=query.strip("\n "))

    def replace_content(self, new_content: str) -> "Query":
        """Update the content while preserving ID and metadata."""
        return Query(
            id=self.id,
            content=new_content,
            metadata=self.metadata,
        )

class EmbeddedQuery(Query):
    embedding: list[float]
    
    class Config:
        category = DataCategory.QUERIES





class RAGPipeline(ABC):
    def __init__(self, development: bool = False) -> None:
        # development = True: Use mock LLM for development environment
        self._development = development

    @abstractmethod
    def generate(self, query: Query, *args, **kwargs) -> Any:
        pass