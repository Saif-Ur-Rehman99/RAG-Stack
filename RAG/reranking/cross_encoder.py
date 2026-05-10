# import opik
from models.embedding_models import EmbeddedChunk
from RAG.base.queries import Query, RAGPipeline
from RAG.base.models import CrossEncoderModelSingleton


class Reranker(RAGPipeline):
    def __init__(self, development: bool = False) -> None:
        super().__init__(development=development)

        self._model = CrossEncoderModelSingleton()

    # @opik.track(name="Reranker.generate")
    def generate(self, query: Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
        if self._development or not chunks:
            return chunks

        query_doc_tuples = [(query.content, chunk.content) for chunk in chunks]
        scores = self._model(query_doc_tuples)

        scored_query_doc_tuples = list(zip(scores, chunks, strict=False))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]

        return reranked_documents
    

if __name__ == "__main__":
    ranker = Reranker()