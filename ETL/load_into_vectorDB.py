import logging as logger
from zenml import step
from typing_extensions import Annotated, List

from RAG.utilities import utils
from models.ovm import VectorBaseDocument
from models.ingestion_models import PDFDocument, IndexingStatus
from db.mongo import MongoDatabaseConnector
from config import settings


def _mark_indexed(sha256: str) -> None:
    db = MongoDatabaseConnector().get_database(settings.DATABASE_NAME)
    db[PDFDocument.get_collection_name()].update_one(
        {"sha256": sha256},
        {"$set": {"indexing_status": IndexingStatus.INDEXED}},
    )


@step
def load_into_vectorDB(documents: Annotated[List, "documents"]) -> Annotated[bool, "successful"]:
    logger.info(f"Loading {len(documents)} documents into the vector database.")

    grouped_documents = VectorBaseDocument.group_by_class(documents)

    for document_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {document_class.get_collection_name()}")

        for documents_batch in utils.batch(documents, size=4):
            try:
                document_class.bulk_insert(documents_batch)
                for doc in documents_batch:
                    if hasattr(doc, "sha256") and doc.sha256:
                        _mark_indexed(doc.sha256)
            except Exception:
                logger.error(f"Failed to insert documents into {document_class.get_collection_name()}")
                return False

    return True
