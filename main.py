from utils.logger import setup_logging
setup_logging()

from RAG.retrieval import ContextRetriever
from RAG.augment import ContextAugmenter
from RAG.generate import RAGGenerator
from RAG.base.queries import Query


_retriever = None
_augmenter = None
_generator = None


def _get_pipeline():
    global _retriever, _augmenter, _generator
    if _retriever is None:
        _retriever = ContextRetriever(development=False)
        _augmenter = ContextAugmenter(development=False)
        _generator = RAGGenerator(development=False)
    return _retriever, _augmenter, _generator


def run(query_text: str, k: int = 5, top_k: int = 3) -> str:
    retriever, augmenter, generator = _get_pipeline()
    query = Query.from_str(query_text)

    # Step 1: Retrieve
    chunks = retriever.search(query_text, k=k)

    # Step 2: Rerank
    reranked = retriever.rerank(query, chunks, keep_top_k=top_k)

    # Step 3: Augment
    prompt = augmenter.generate(query, reranked)

    # Step 4: Generate
    return generator.generate(query, prompt)


if __name__ == "__main__":
    """tell me about fund report of Conventional fund for Jan 2025"""
    print("\nRAG Pipeline — Alfalah Investments Fund Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        query_text = input("Question: ").strip()
        if not query_text or query_text.lower() == "exit":
            break

        answer = run(query_text)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)
