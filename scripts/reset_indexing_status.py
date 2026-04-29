from db.mongo import MongoDatabaseConnector
from models.ingestion_models import PDFDocument, IndexingStatus
from config import settings

db = MongoDatabaseConnector().get_database(settings.DATABASE_NAME)
result = db[PDFDocument.get_collection_name()].update_many(
    {"indexing_status": IndexingStatus.INDEXED},
    {"$set": {"indexing_status": IndexingStatus.PENDING}},
)
print(f"✅ Reset {result.modified_count} document(s) → indexing_status = PENDING.")
