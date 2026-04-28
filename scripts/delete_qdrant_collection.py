from db.qdrant import connection
from db.mongo import MongoDatabaseConnector
from models.ingestion_models import PDFDocument, IndexingStatus
from config import settings


def reset_indexing_status() -> int:
    """Reset all INDEXED documents in MongoDB back to PENDING."""
    db = MongoDatabaseConnector().get_database(settings.DATABASE_NAME)
    result = db[PDFDocument.get_collection_name()].update_many(
        {"indexing_status": IndexingStatus.INDEXED},
        {"$set": {"indexing_status": IndexingStatus.PENDING}},
    )
    return result.modified_count


collection_name = "embedded_pdfs"

connection.delete_collection(collection_name=collection_name)
print(f"✅ Qdrant collection '{collection_name}' deleted.")

n = reset_indexing_status()
print(f"✅ Reset {n} document(s) in MongoDB → indexing_status = PENDING.")
