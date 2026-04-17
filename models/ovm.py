import uuid
import numpy as np
from abc import ABC
from uuid import UUID
from typing import Any, Callable, Dict
from pydantic import UUID4, BaseModel, Field
import logging as logger
import traceback

from RAG.base.constants import DataCategory
from RAG.base.models import get_embedding_model

from qdrant_client.http import exceptions
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Record, PointStruct, CollectionInfo, SparseVector, SparseVectorParams
from db.qdrant import connection


# logger = logging.getLogger(__name__)

class VectorBaseDocument(BaseModel, ABC):
    # When you create an instance without specifying id, it automatically generates a new random UUID.
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False
        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_record(cls, point: Record) -> "VectorBaseDocument":
        """Converts Qdrant PointStruct to Pydantic Document"""
        # convert qdrant_id to UUID (Pydantic models expect UUID4 type for the id field)
        _id = UUID(str(point.id), version=4)
        payload = point.payload or {}

        attributes = {
            "id": _id,
            **payload,
        }
        if cls._has_class_attribute("embedding"):
            vector = point.vector
            if isinstance(vector, dict):
                attributes["embedding"] = vector.get("dense")
            else:
                attributes["embedding"] = vector or None

        return cls(**attributes) # Document(id=UUID(...), title="...", embedding=[...])

    
    def to_point(self, **kwargs) -> PointStruct:
        """Converts Pydantic Document to Qdrant PointStruct"""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        payload = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        _id = str(uuid.uuid4())

        dense_vector = payload.pop("embedding", None)
        if dense_vector is not None and isinstance(dense_vector, np.ndarray):
            dense_vector = dense_vector.tolist()

        sparse_raw = payload.pop("sparse_embedding", None)

        if sparse_raw is not None:
            vector = {
                "dense": dense_vector,
                "sparse": SparseVector(
                    indices=sparse_raw["indices"],
                    values=sparse_raw["values"],
                ),
            }
        else:
            vector = dense_vector or {}

        return PointStruct(id=_id, vector=vector, payload=payload)

    def model_dump(self, **kwargs) -> dict:
        dict_ = super().model_dump(**kwargs)
        dict_ = self._uuid_to_str(dict_)

        return dict_

    def _uuid_to_str(self, item: Any) -> Any:
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, UUID):
                    item[key] = str(value)
                elif isinstance(value, list):
                    item[key] = [self._uuid_to_str(v) for v in value]
                elif isinstance(value, dict):
                    item[key] = {k: self._uuid_to_str(v) for k, v in value.items()}

        return item

    
    @classmethod
    def bulk_insert(cls, documents: list["VectorBaseDocument"]) -> bool:
        try:
            cls._bulk_insert(documents)
        except exceptions.UnexpectedResponse as e:
            logger.warning(
                f"Collection '{cls.get_collection_name()}' may not exist or insertion failed: {e}"
            )
            # logger.error(traceback.format_exc())

            # Try creating the collection and re-inserting
            cls.create_collection()
            try:
                cls._bulk_insert(documents)
            except exceptions.UnexpectedResponse as e2:
                logger.error(
                    f"Second insert attempt failed for '{cls.get_collection_name()}': {e2}"
                )
                # logger.error(traceback.format_exc())
                # raise
            return True
        except Exception as e:
            logger.error(f"Unexpected error in bulk_insert: {e}")
            # logger.error(traceback.format_exc())
            # raise
        return True

    @classmethod
    def _bulk_insert(cls, documents: list["VectorBaseDocument"]) -> None:
        # logger.info(f"_Bulk Insert Method Document: {documents[0]}")
        points = [doc.to_point() for doc in documents]
        # logger.info(f"_Bulk Insert Method Point: {points[0].payload}")
        # logger.info(f"Collection Name {cls.get_collection_name()}")
        connection.upsert(collection_name=cls.get_collection_name(), points=points)



    @classmethod
    def bulk_find(cls, limit: int = 10, **kwargs) -> tuple[list["VectorBaseDocument"], UUID | None]:
        try:
            documents, next_offset = cls._bulk_find(limit=limit, **kwargs)
        except exceptions.UnexpectedResponse:
            print(f"Failed to search documents in '{cls.get_collection_name()}'.")
            documents, next_offset = [], None

        return documents, next_offset

    @classmethod
    def _bulk_find(cls, limit: int = 10, **kwargs) -> tuple[list["VectorBaseDocument"], UUID | None]:
        collection_name = cls.get_collection_name()

        offset = kwargs.pop("offset", None)
        offset = str(offset) if offset else None

        records, next_offset = connection.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=kwargs.pop("with_payload", True),
            with_vectors=kwargs.pop("with_vectors", False),
            offset=offset,
            **kwargs,
        )
        documents = [cls.from_record(record) for record in records]
        if next_offset is not None:
            next_offset = UUID(next_offset, version=4)

        return documents, next_offset



    @classmethod
    def search(cls, query_vector: list, limit: int = 10, **kwargs) -> list["VectorBaseDocument"]:
        try:
            documents = cls._search(query_vector=query_vector, limit=limit, **kwargs)
        except exceptions.UnexpectedResponse:
            print(f"Failed to search documents in '{cls.get_collection_name()}'.")
            documents = []

        return documents

    @classmethod
    def _search(cls, query_vector: list, limit: int = 10, **kwargs) -> list["VectorBaseDocument"]:
        collection_name = cls.get_collection_name()

        try:
            print(f"[DEBUG] Searching in collection: {collection_name}")
            print(f"🔹 Vector length: {len(query_vector)}")
            print(f"🔹 Vector sample: {query_vector[:5]} ...")
            print(f"🔹 Limit: {limit}")
            print(f"🔹 Extra kwargs: {kwargs}")

            response = connection.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=kwargs.pop("with_payload", True),
                with_vectors=kwargs.pop("with_vectors", False),
                **kwargs,
            )
            records = response.points

            print(f"✅ [DEBUG] Qdrant returned {len(records)} records")
            documents = [cls.from_record(record) for record in records]
            return documents

        except Exception as e:
            import traceback
            print(f"[ERROR] Qdrant search failed for {collection_name}")
            print(f"Type: {type(e)}")
            print(f"Error message: {e}")
            print("Full traceback:\n", traceback.format_exc())

            # If it’s an HTTP error from Qdrant, print its response too
            if hasattr(e, "response") and hasattr(e.response, "json"):
                try:
                    print("📜 Qdrant Error JSON:", e.response.json())
                except Exception:
                    pass

            raise



    @classmethod
    def get_payload_indexes(cls) -> dict:
        """Override in subclasses to declare payload indexes: {field_name: schema_type}.
        Valid schema_type values: 'keyword', 'integer', 'float', 'bool', 'geo', 'text'.
        """
        return {}

    @classmethod
    def ensure_payload_indexes(cls) -> None:
        collection_name = cls.get_collection_name()
        indexes = cls.get_payload_indexes()
        for field_name, field_schema in indexes.items():
            try:
                connection.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
                # logger.info(f"Payload index created for '{field_name}' in '{collection_name}'.")
            except Exception as e:
                # Index may already exist — not a fatal error
                logger.warning(f"Payload index for '{field_name}' in '{collection_name}': {e}")

    @classmethod
    def get_or_create_collection(cls) -> CollectionInfo:
        collection_name = cls.get_collection_name()

        try:
            info = connection.get_collection(collection_name=collection_name)
            cls.ensure_payload_indexes()
            return info
        except exceptions.UnexpectedResponse:
            use_vector_index = cls.get_use_vector_index()
            collection_created = cls._create_collection(collection_name=collection_name, use_vector_index=use_vector_index)
            if collection_created is False:
                raise RuntimeError(f"Couldn't create collection {collection_name}") from None

            return connection.get_collection(collection_name=collection_name)

    @classmethod
    def create_collection(cls) -> bool:
        collection_name = cls.get_collection_name()
        use_vector_index = cls.get_use_vector_index()
        return cls._create_collection(collection_name=collection_name, use_vector_index=use_vector_index)

    @classmethod
    def _create_collection(cls, collection_name: str, use_vector_index: bool = True) -> bool:
        try:
            logger.info(f"Creating collection '{collection_name}' in Qdrant...")

            use_hybrid_index = cls.get_use_hybrid_index()

            if use_hybrid_index:
                connection.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=get_embedding_model().embedding_size,
                            distance=Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={"sparse": SparseVectorParams()},
                )
            else:
                connection.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=get_embedding_model().embedding_size,
                        distance=Distance.COSINE,
                    ),
                )

            cls.ensure_payload_indexes()
            logger.info(f"Collection '{collection_name}' created successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            return False


    @classmethod
    def get_category(cls) -> DataCategory:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "category"):
            raise AttributeError(
                f"{cls.__name__} must define a Config class with a 'category' attribute."
            )
        return cls.Config.category

    @classmethod
    def get_collection_name(cls) -> str:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "name"):
            logger.info("Get Collection Name is not working")
            raise ValueError(
                "The class should define a Config class with the 'name' property that reflects the collection's name."
            )
        return cls.Config.name

    @classmethod
    def get_use_vector_index(cls) -> bool:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "use_vector_index"):
            return True
        return cls.Config.use_vector_index

    @classmethod
    def get_use_hybrid_index(cls) -> bool:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "use_hybrid_index"):
            return False
        return cls.Config.use_hybrid_index


    @classmethod
    def group_by_class(cls, documents: list["VectorBaseDocument"]) -> Dict["VectorBaseDocument", list["VectorBaseDocument"]]:
        return cls._group_by(documents, selector=lambda doc: doc.__class__)

    @classmethod
    def group_by_category(cls, documents: list["VectorBaseDocument"]) -> Dict[DataCategory, list["VectorBaseDocument"]]:
        return cls._group_by(documents, selector=lambda doc: doc.get_category())

    @classmethod
    def _group_by(cls, documents: list["VectorBaseDocument"], selector: Callable[["VectorBaseDocument"], Any]) -> Dict[Any, list["VectorBaseDocument"]]:
        grouped = {}
        for doc in documents:
            key = selector(doc)

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(doc)

        return grouped

    @classmethod
    def collection_name_to_class(cls, collection_name: str) -> type["VectorBaseDocument"]:
        for subclass in cls.__subclasses__():
            try:
                if subclass.get_collection_name() == collection_name:
                    return subclass
            except ValueError:
                pass

            try:
                return subclass.collection_name_to_class(collection_name)
            except ValueError:
                continue

        raise ValueError(f"No subclass found for collection name: {collection_name}")


    @classmethod
    def _has_class_attribute(cls, attribute_name: str) -> bool:
        if attribute_name in cls.__annotations__:
            return True

        for base in cls.__bases__:
            if hasattr(base, "_has_class_attribute") and base._has_class_attribute(attribute_name):
                return True

        return False
    