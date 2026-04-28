from db.mongo import MongoDatabaseConnector
from config import settings

COLLECTION_NAME = "raw_documents"

client = MongoDatabaseConnector()
db     = client[settings.DATABASE_NAME]

db.drop_collection(COLLECTION_NAME)

print(f"✅ Collection '{COLLECTION_NAME}' deleted from database '{settings.DATABASE_NAME}'.")
