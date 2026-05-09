from qdrant_client.models import Filter, Prefetch, FusionQuery, Fusion, SparseVector

from RAG.base.models import get_embedding_model, SparseEmbeddingModelSingleton
from RAG.base.queries import Query
from models.embedding_models import EmbeddedChunk, EmbeddedPDFChunk
from db.qdrant import connection
from utils.logger import logging as logger


class HybridSearcher:
    def __init__(self) -> None:
        self._dense_model = get_embedding_model()
        self._sparse_model = SparseEmbeddingModelSingleton()

    def search(
        self,
        query: Query,
        k: int,
        query_filter: Filter | None = None,
    ) -> list[EmbeddedChunk]:
        dense_vector = self._dense_model(query.content, to_list=True)
        sparse_result = self._sparse_model([query.content])[0]

        response = connection.query_points(
            collection_name=EmbeddedPDFChunk.get_collection_name(),
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=query_filter,
                    limit=k * 2,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_result["indices"],
                        values=sparse_result["values"],
                    ),
                    using="sparse",
                    filter=query_filter,
                    limit=k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        logger.info(f"Hybrid search returned {len(response.points)} results for: {query.content}")
        return [EmbeddedPDFChunk.from_record(point) for point in response.points]
