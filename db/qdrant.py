from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
# from qdrant_client.http import 
from config import settings
import logging as logger
import traceback
import json

class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __new__(cls, *args, **kwargs) -> QdrantClient:
        if cls._instance is None:
            try:
                if settings.USE_QDRANT_CLOUD:
                    logger.info("Making Connection - on Cloud")
                    cls._instance = QdrantClient(
                        url=settings.QDRANT_CLOUD_URL,
                        api_key=settings.QDRANT_API_KEY,
                    )

                    uri = settings.QDRANT_CLOUD_URL
                else:
                    logger.info("Making Connection - Locally")
                    cls._instance = QdrantClient(
                        host=settings.QDRANT_DATABASE_HOST,
                        port=settings.QDRANT_DATABASE_PORT,
                    )

                    uri = f"{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}"

                logger.info(f"Connection to Qdrant DB with URI successful: {uri}")
            except Exception as e:
                logger.error("Failed to connect to Qdrant:")
                logger.error(traceback.format_exc())
                raise e

        return cls._instance

def safe_qdrant_call(func_name: str, *args, **kwargs):
    """Wrapper to catch and print detailed Qdrant API errors."""
    func = getattr(connection, func_name)

    try:
        result = func(*args, **kwargs)
        if isinstance(result):
            logger.info(f"✅ Qdrant call '{func_name}' succeeded.")
        else:
            logger.info(f"✅ Qdrant call '{func_name}' returned data.")
        return result

    except UnexpectedResponse as e:
        logger.error(f"❌ Qdrant call '{func_name}' failed: UnexpectedResponse")
        logger.error(f"Status Code: {getattr(e, 'status_code', 'N/A')}")
        logger.error(f"Response: {getattr(e, 'response', None)}")
        logger.error(traceback.format_exc())
        raise e

    except Exception as e:
        logger.error(f"❌ Qdrant call '{func_name}' raised {type(e).__name__}: {str(e)}")
        logger.error("Args: " + json.dumps(kwargs, indent=2, default=str))
        logger.error(traceback.format_exc())
        raise e



connection = QdrantDatabaseConnector()