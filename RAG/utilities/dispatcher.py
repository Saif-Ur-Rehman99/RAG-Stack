import logging as logger
from typing import Union, List
from models.ovm import VectorBaseDocument
from RAG.base.constants import DataCategory
from RAG.utilities.data_handlers import *



class CleaningHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> CleaningDataHandler:
        if data_category == DataCategory.PDF:
            return PDFCleaningHandler()
        # elif data_category == DataCategory.POSTS:
        #     return ImageCleaningHandler()
        else:
            raise ValueError(f"Unsupported data type: {data_category}")

class CleaningDispatcher:
    factory = CleaningHandlerFactory()

    @classmethod
    def dispatch(cls, data_model) -> VectorBaseDocument:
        """Cleaning based on document type (PDF, image, etc.)."""

        if hasattr(data_model, "metadata") and "category" in data_model.metadata:
            data_category = data_model.metadata["category"]
        else:
            raise ValueError("Document metadata must include 'category' field")

        # Create and use the proper handler
        handler = cls.factory.create_handler(data_category)
        clean_model = handler.clean(data_model)

        logger.info("Document cleaned successfully.")

        return clean_model




class ChunkingHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> ChunkingDataHandler:
        if data_category == DataCategory.PDF:
            return PDFChunkingHandler()
        else:
            raise ValueError("Unsupported data type")

class ChunkingDispatcher:
    factory = ChunkingHandlerFactory

    @classmethod
    def dispatch(cls, data_model: VectorBaseDocument) -> List[VectorBaseDocument]:
        data_category = data_model.get_category()
        handler = cls.factory.create_handler(data_category)
        chunk_models = handler.chunk(data_model)

        if not chunk_models:
            logger.warning(f"No chunks returned for category: {data_category}")
        else:
            logger.info(f"Document chunked successfully. Num chunks: {len(chunk_models)}, Category: {data_category}")

        return chunk_models




class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> EmbeddingDataHandler:
        
        if data_category == DataCategory.PDF:
            return PDFEmbeddingHandler()
        elif data_category == DataCategory.QUERIES:
            return QueryEmbeddingHandler()
        else:
            raise ValueError("Unsupported data type")

class EmbeddingDispatcher:
    factory = EmbeddingHandlerFactory

    @classmethod
    def dispatch(
            cls, data_model: Union[VectorBaseDocument, List[VectorBaseDocument]]
        ) -> Union[VectorBaseDocument, List[VectorBaseDocument]]:
        
        is_list = isinstance(data_model, list)
        if not is_list:
            data_model = [data_model]

        if len(data_model) == 0:
            return []

        data_category = data_model[0].get_category()
        
        assert all(dm.get_category() == data_category for dm in data_model), \
        "Data models must be of the same category."

        handler = cls.factory.create_handler(data_category)
        embedded_chunk_model = handler.embed_batch(data_model)

        if not is_list:
            embedded_chunk_model = embedded_chunk_model[0]

        logger.info(f"Data embedded successfully. {data_category}")

        return embedded_chunk_model