import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from config import settings


class MongoDatabaseConnector:
    _instance: MongoClient | None = None
    _pid: int | None = None

    def __new__(cls, *args, **kwargs) -> MongoClient:
        current_pid = os.getpid()

        # After a fork the inherited socket is broken — create a fresh client.
        if cls._instance is not None and cls._pid != current_pid:
            cls._instance = None

        if cls._instance is None:
            try:
                cls._instance = MongoClient(settings.DATABASE_HOST)
                cls._pid = current_pid
                print(f"Connection to MongoDB with URI successful: {settings.DATABASE_HOST}")
            except ConnectionFailure as e:
                print(f"Couldn't connect to the database: {e!s}")
                raise

        return cls._instance
