from abc import ABC, abstractmethod
from typing import List
from models.ingestion_models import PDFDocument
from models.cleaning_models import PDFCleanedDocument
from models.chunking_models import Chunk, PDFChunk
from models.embedding_models import EmbeddedChunk, EmbeddedPDFChunk

from RAG.base.queries import EmbeddedQuery
from RAG.base.models import get_embedding_model, SparseEmbeddingModelSingleton
from RAG.indexing.chunking import chunk_text, clean_text, clean_markdown, chunk_markdown_by_heading
from config import settings


embedding_model = get_embedding_model()
sparse_embedding_model = SparseEmbeddingModelSingleton()



class CleaningDataHandler(ABC):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """
    @abstractmethod
    def clean(self, data_model: object) -> object:
        pass

class PDFCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: PDFDocument) -> PDFCleanedDocument:
        return PDFCleanedDocument(
            content=clean_markdown(data_model.page_content),
            filename=data_model.metadata.get("filename"),
            path=data_model.metadata.get("path"),
            sha256=data_model.metadata.get("sha256"),
            bank_type=data_model.metadata.get("bank_type"),
            month=data_model.metadata.get("month"),
            year=data_model.metadata.get("year"),
        )
    




class ChunkingDataHandler(ABC):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @property
    def metadata(self) -> dict:
        return {
            "chunking_version": "v1.0",
            "chunking_strategy": "fixed_size",
            "chunking_size": 500,
            "chunking_overlap": 50,
            "chunking_status": "pending",
        }

    @abstractmethod
    def chunk(self, data_model: object) -> list[object]:
        pass


class PDFChunkingHandler(ChunkingDataHandler):

    @property
    def metadata(self) -> dict:
        return {
            "chunking_version"  : "v2.0",
            "chunking_strategy" : "markdown_heading",
            "chunking_size"     : None,
            "chunking_overlap"  : None,
            "chunking_status"   : "pending",
        }

    def chunk(self, data_model: PDFCleanedDocument) -> List[PDFChunk]:
        chunks = chunk_markdown_by_heading(data_model.content)

        return [
            PDFChunk(
                id=data_model.id,
                content=chunk,
                filename=data_model.filename,
                sha256=data_model.sha256,
                bank_type=data_model.bank_type,
                month=data_model.month,
                year=data_model.year,
                metadata={**self.metadata, "chunking_status": "success"},
            )
            for chunk in chunks
        ]






class EmbeddingDataHandler(ABC):
    """
    Abstract class for all embedding data handlers.
    Handles data transformation logic for the embedding step.
    """

    def embed(self, data_model: Chunk) -> EmbeddedChunk:
        """Embed a single data model."""
        return self.embed_batch([data_model])[0]

    def embed_batch(self, data_models: list[Chunk]) -> List[EmbeddedChunk]:
        """Embed a batch of data models."""
        embedding_model_input = [data_model.content for data_model in data_models]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        embedded_chunks = [
            self.map_model(data_model, embedding)
            for data_model, embedding in zip(data_models, embeddings, strict=False)
        ]

        return embedded_chunks

    @abstractmethod
    def map_model(self, data_model: Chunk, embedding: List[float]) -> EmbeddedChunk:
        """Map raw embeddings back into the EmbeddedChunk model."""
        pass


_EMBEDDING_VERSION = "v1.0"


class PDFEmbeddingHandler(EmbeddingDataHandler):
    """
    Handles embedding for PDF chunks — produces both dense and sparse vectors.
    """

    def embed_batch(self, data_models: list[PDFChunk]) -> List[EmbeddedPDFChunk]:
        texts = [dm.content for dm in data_models]
        dense_embeddings = embedding_model(texts, to_list=True)
        sparse_embeddings = sparse_embedding_model(texts)

        return [
            self.map_model(dm, dense_emb, sparse_emb)
            for dm, dense_emb, sparse_emb in zip(data_models, dense_embeddings, sparse_embeddings)
        ]

    def map_model(self, data_model: PDFChunk, embedding: List[float], sparse_embedding: dict | None = None) -> EmbeddedPDFChunk:
        return EmbeddedPDFChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            filename=data_model.filename,
            sha256=data_model.sha256,
            bank_type=data_model.bank_type,
            month=data_model.month,
            year=data_model.year,
            metadata={
                **data_model.metadata,
                "embedding_model": embedding_model.model_id,
                "embedding_dimensions": embedding_model.embedding_size,
                "embedding_version": _EMBEDDING_VERSION,
                "embedding_status": "success",
            },
        )

class QueryEmbeddingHandler(EmbeddingDataHandler):
    """
    Handles embedding for query-type data.
    """

    def map_model(self, data_model: Chunk, embedding: list[float]) -> EmbeddedChunk:
        return EmbeddedQuery(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            metadata={
                **data_model.metadata,
                "embedding_model": embedding_model.model_id,
                "embedding_dimensions": embedding_model.embedding_size,
                "embedding_version": _EMBEDDING_VERSION,
                "embedding_status": "success",
            },
        )

