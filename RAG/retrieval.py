import concurrent.futures
import opik
from qdrant_client.models import FieldCondition, Filter, MatchValue
from typing_extensions import List
from RAG.base.queries import Query
from RAG.query_optimization.self_query import SelfQuery
from RAG.query_optimization.query_expansion import QueryExpansion
from models.embedding_models import EmbeddedChunk, EmbeddedPDFChunk
from utils.logger import logging as logger
from RAG.reranking.cross_encoder import Reranker
from RAG.filtering.hybrid_search import HybridSearcher




class ContextRetriever:
    def __init__(self, development: bool = False) -> None:
        self._query_expander = QueryExpansion(development=development)
        self._metadata_extractor = SelfQuery(development=development)
        self._reranker = Reranker(development=development)
        self._hybrid_searcher = HybridSearcher()
        EmbeddedPDFChunk.ensure_payload_indexes()

    def search(self, query: str, k: int = 3, expand_to_n_queries: int = 2) -> List[EmbeddedChunk]:
        """
        High-level search interface:
        1. Extract metadata (month, year, etc.)
        2. Expand query (if needed)
        3. Embed and search all queries in parallel
        """

        # --- Step 1: Extract metadata ---
        query_model = Query.from_str(query)
        query_model = self._metadata_extractor.generate(query_model)
        # print(f"Extracted metadata: {query_model}")

        # --- Step 2: Expand query ---
        expanded_queries = self._query_expander.generate(query_model, n=expand_to_n_queries)
        print(f"Generated {len(expanded_queries)} expanded queries")

        # --- Step 3: Search all queries in parallel ---
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = [executor.submit(self._search_single_query, q, k) for q in expanded_queries]
            all_results = [t.result() for t in concurrent.futures.as_completed(tasks)]

        # Deduplicates Results:
        documents = {doc.id: doc for batch in all_results for doc in batch}.values()
        print(f"Retrieved {len(documents)} total unique documents")

        return list(documents)

    def _search_single_query(self, query: Query, k: int = 3) -> List[EmbeddedChunk]:
        if k < 3:
            raise ValueError("k must be >= 3")

        # --- Prepare metadata filters ---
        query_filter = None
        if query.metadata:
            must_filters = [
                FieldCondition(key=field, match=MatchValue(value=query.metadata[field]))
                for field in ["bank_type", "month", "year"]
                if query.metadata.get(field)
            ]
            if must_filters:
                query_filter = Filter(must=must_filters)
                # logger.info(f"Using metadata filters: {must_filters}")

        # --- Hybrid search (dense + sparse via RRF) ---
        try:
            results = self._hybrid_searcher.search(query=query, k=k, query_filter=query_filter)
            logger.info(f"Retrieved {len(results)} results for query: {query.content}")
            return results
        except Exception as e:
            logger.error(f"Search failed for query '{query.content}': {e}")
            return []



    def rerank(self, query: str | Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
        if isinstance(query, str):
            query = Query.from_str(query)

        reranked_documents = self._reranker.generate(query=query, chunks=chunks, keep_top_k=keep_top_k)

        print(f"{len(reranked_documents)} documents reranked successfully.")

        return reranked_documents